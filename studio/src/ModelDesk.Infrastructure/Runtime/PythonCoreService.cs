using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using ModelDesk.Core;
using ModelDesk.Infrastructure.Settings;

namespace ModelDesk.Infrastructure.Runtime;

public sealed class PythonCoreService : IPythonCoreService
{
    public const int MaximumLogBytes = 8 * 1024 * 1024;
    private const int MaximumReportBytes = 4 * 1024 * 1024;
    private const int MaximumLineCharacters = 16384;
    public PythonCoreService(string? stateDirectory = null) { _ = stateDirectory; }

    public IReadOnlyList<ModelProfile> GetProfiles(string projectRoot)
    {
        var root = Root(projectRoot);
        var directory = LocalFiles.Within(Path.Combine(root, "config", "models"), root);
        if (!Directory.Exists(directory)) return [];
        var profiles = new List<ModelProfile>();
        foreach (var path in Directory.EnumerateFiles(directory, "*.json", SearchOption.TopDirectoryOnly).Take(128))
        {
            LocalFiles.CheckPath(path);
            if (new FileInfo(path).Length > 65536) throw new InvalidDataException("A model profile exceeds 64 KiB.");
            using var document = JsonDocument.Parse(LocalFiles.ReadBoundedAsync(path, 65536).GetAwaiter().GetResult());
            var json = document.RootElement;
            var id = RequiredString(json, "model_id");
            var revision = RequiredString(json, "revision");
            var filename = Path.GetFileNameWithoutExtension(path);
            var key = filename switch { "abliterated-nvfp4" => "nvfp4", "cybersecurity-fp8" => "fp8", _ => filename };
            var label = key switch { "nvfp4" => "GLM · NVFP4", "fp8" => "GLM · FP8", _ => id };
            var modelPath = RequiredString(json, "model_directory");
            modelPath = Path.GetFullPath(Path.IsPathRooted(modelPath) ? modelPath : Path.Combine(root, modelPath));
            profiles.Add(new ModelProfile(key, label, id, revision, path, modelPath, ReportDirectory(root, json)));
        }
        return profiles.OrderBy(profile => profile.Key == "nvfp4" ? 0 : 1).ThenBy(profile => profile.Key, StringComparer.Ordinal).ToArray();
    }

    public async Task<CoreRunResult> RunAsync(CoreRunRequest request, IProgress<CoreOutput>? output = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        request.Settings.Validate();
        var operation = CoreOperations.Get(request.Operation);
        if (request.Arguments.Any(value => value is null || value.Contains('\0')) || request.Arguments.Count > 256)
            throw new ArgumentException("Core arguments contain an invalid value or exceed 256 entries.");
        if (request.Arguments.Sum(value => (long)Encoding.UTF8.GetByteCount(value)) > 1024 * 1024)
            throw new ArgumentException("Core argument text exceeds the 1 MiB limit.");
        var root = Root(request.Settings.ProjectRoot);
        var profiles = GetProfiles(root);
        var configPath = request.ConfigPath ?? profiles.FirstOrDefault(profile => profile.Key == request.Settings.Profile)?.ConfigPath
            ?? throw new ArgumentException("The selected model profile does not exist.");
        configPath = LocalFiles.Within(Path.GetFullPath(configPath, root), Path.Combine(root, "config"));
        var configBytes = await LocalFiles.ReadBoundedAsync(configPath, 65536, cancellationToken);
        var config = JsonNode.Parse(configBytes) as JsonObject ?? throw new InvalidDataException("Core profile must be a JSON object.");
        config["ram_budget_bytes"] = request.Settings.RamBudgetBytes;
        config["cpu_job_percent"] = request.Settings.CpuLimitPercent;
        config["gpu_index"] = request.Settings.GpuIndex;
        config["gpu_average_target"] = request.Settings.GpuTargetPercent / 100;
        config["gpu_window_seconds"] = request.Settings.GpuWindowSeconds;
        if (!string.IsNullOrWhiteSpace(request.Settings.RuntimeModelDirectory))
            config["model_directory"] = Path.GetFullPath(request.Settings.RuntimeModelDirectory, root);
        using var parsedConfig = JsonDocument.Parse(config.ToJsonString());
        var reportDirectory = ReportDirectory(root, parsedConfig.RootElement);
        var id = DateTimeOffset.UtcNow.ToString("yyyyMMddTHHmmssfffZ") + "-" + Guid.NewGuid().ToString("N")[..8];
        var runDirectory = LocalFiles.Within(Path.Combine(root, "reports", "modeldesk", "runs", id), root);
        Directory.CreateDirectory(runDirectory);
        var generatedConfig = Path.Combine(runDirectory, "config.json");
        await LocalFiles.AtomicWriteAsync(generatedConfig, Encoding.UTF8.GetBytes(config.ToJsonString(LocalFiles.JsonOptions)), cancellationToken);
        var logPath = Path.Combine(runDirectory, "output.log");
        var expected = ExpectedReports(root, reportDirectory, operation.Id).ToArray();
        var prior = new Dictionary<string, (DateTime LastWrite, long Length)>();
        foreach (var path in expected)
            if (File.Exists(path)) { var info = new FileInfo(LocalFiles.CheckPath(path)); prior[path] = (info.LastWriteTimeUtc, info.Length); }
        var reportHints = new ConcurrentQueue<string>();
        var startInfo = CreateStartInfo(root, request.Settings, operation, generatedConfig, request.Arguments);
        var started = DateTimeOffset.UtcNow;
        var cancelled = false;
        int exitCode;
        await using (var log = new FileStream(logPath, FileMode.CreateNew, FileAccess.Write, FileShare.Read, 65536, FileOptions.Asynchronous))
        using (var logLock = new SemaphoreSlim(1, 1))
        using (var process = new Process { StartInfo = startInfo })
        using (var lifetime = new WindowsProcessJob())
        {
            long written = 0;
            var truncated = false;
            async Task Publish(string line, bool error)
            {
                output?.Report(new CoreOutput(DateTimeOffset.UtcNow, line, error));
                if ((line.StartsWith("Report:", StringComparison.Ordinal) || line.StartsWith("Saved:", StringComparison.Ordinal))
                    && reportHints.Count < 32) reportHints.Enqueue(line[(line.IndexOf(':') + 1)..].Trim().Trim('"', '\''));
                await logLock.WaitAsync();
                try
                {
                    if (truncated) return;
                    var bytes = Encoding.UTF8.GetBytes((error ? "[stderr] " : "") + line + Environment.NewLine);
                    if (written + bytes.Length > MaximumLogBytes - 128)
                    {
                        bytes = Encoding.UTF8.GetBytes("[ModelDesk: remaining output omitted from bounded disk log]" + Environment.NewLine);
                        truncated = true;
                    }
                    await log.WriteAsync(bytes);
                    written += bytes.Length;
                }
                finally { logLock.Release(); }
            }
            try
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (!process.Start()) throw new InvalidOperationException("The Python core process could not start.");
                // Assign before starting pipe consumers or awaiting user work.
                // Unlike a process snapshot, a Job tracks future descendants.
                lifetime.Attach(process);
                using var registration = cancellationToken.Register(() =>
                {
                    try { lifetime.Stop(); }
                    catch (System.ComponentModel.Win32Exception) { }
                    try { if (!process.HasExited) process.Kill(entireProcessTree: true); }
                    catch (InvalidOperationException) { }
                    catch (System.ComponentModel.Win32Exception) { }
                });
                var stdout = Task.Run(() => Pump(process.StandardOutput, false, Publish));
                var stderr = Task.Run(() => Pump(process.StandardError, true, Publish));
                await process.WaitForExitAsync(CancellationToken.None);
                await Task.WhenAll(stdout, stderr);
                cancelled = cancellationToken.IsCancellationRequested;
                exitCode = cancelled ? 130 : process.ExitCode;
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                cancelled = true;
                exitCode = 130;
                await Publish("Operation cancelled.", false);
            }
            catch (Exception exception) when (exception is System.ComponentModel.Win32Exception or IOException or InvalidOperationException)
            {
                try { lifetime.Stop(); }
                catch (System.ComponentModel.Win32Exception) { }
                try { if (process.Id > 0 && !process.HasExited) process.Kill(entireProcessTree: true); }
                catch (InvalidOperationException) { }
                catch (System.ComponentModel.Win32Exception) { }
                exitCode = 1;
                await Publish("Core process failed: " + exception.Message, true);
            }
        }
        var hints = reportHints.Reverse().ToArray();
        var reportPath = await CaptureFreshReport(root, reportDirectory, runDirectory,
            cancelled ? hints : hints.Concat(expected), prior, started, operation.Id,
            RequiredString(parsedConfig.RootElement, "model_id"), RequiredString(parsedConfig.RootElement, "revision"), cancelled, expected);
        var result = new CoreRunResult(id, exitCode, cancelled, logPath, reportPath, started, DateTimeOffset.UtcNow);
        await LocalFiles.AtomicWriteAsync(Path.Combine(runDirectory, "run.json"), JsonSerializer.SerializeToUtf8Bytes(result, LocalFiles.JsonOptions));
        return result;
    }

    private static ProcessStartInfo CreateStartInfo(string root, AppSettings settings, CoreOperation operation,
        string configPath, IReadOnlyList<string> arguments)
    {
        var info = new ProcessStartInfo { WorkingDirectory = root, UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardOutput = true, RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8 };
        info.Environment["PYTHONUTF8"] = "1";
        info.Environment["PYTHONIOENCODING"] = "utf-8";
        if (operation.Kind == CoreOperationKind.Python)
        {
            info.FileName = string.IsNullOrWhiteSpace(settings.PythonExecutable) ? "python" : settings.PythonExecutable;
            foreach (var argument in new[] { "-u", "-m", "glm_local", "--config", configPath, operation.Id }.Concat(arguments))
                info.ArgumentList.Add(argument);
            return info;
        }
        if (arguments.Count != 0) throw new ArgumentException("Build/setup/test operations use their fixed core wrapper without extra shell arguments.");
        info.FileName = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System), "WindowsPowerShell", "v1.0", "powershell.exe");
        foreach (var option in new[] { "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass" }) info.ArgumentList.Add(option);
        if (operation.Kind == CoreOperationKind.TestReference)
        {
            var script = LocalFiles.Within(Path.Combine(root, "test-reference.bat"), root);
            if (!File.Exists(script)) throw new FileNotFoundException("The core test wrapper is missing.", script);
            info.ArgumentList.Add("-Command");
            info.ArgumentList.Add("& '" + script.Replace("'", "''", StringComparison.Ordinal) + "'; exit $LASTEXITCODE");
        }
        else
        {
            var script = LocalFiles.Within(Path.Combine(root, operation.Kind == CoreOperationKind.BuildNative ? "build-native.ps1" : "setup-reference.ps1"), root);
            if (!File.Exists(script)) throw new FileNotFoundException("The core PowerShell wrapper is missing.", script);
            info.ArgumentList.Add("-File");
            info.ArgumentList.Add(script);
            if (operation.Kind == CoreOperationKind.SetupReference)
            {
                info.ArgumentList.Add("-Python");
                info.ArgumentList.Add(string.IsNullOrWhiteSpace(settings.PythonExecutable) ? "python" : settings.PythonExecutable);
            }
        }
        return info;
    }

    private static async Task Pump(StreamReader stream, bool error, Func<string, bool, Task> publish)
    {
        // Read the pipe's available bytes. StreamReader.ReadAsync(charBuffer)
        // can wait to fill its requested character count, delaying short lines.
        var bytes = new byte[4096];
        var buffer = new char[Encoding.UTF8.GetMaxCharCount(bytes.Length)];
        var decoder = Encoding.UTF8.GetDecoder();
        var line = new StringBuilder();
        var oversized = false;
        while (true)
        {
            var received = await stream.BaseStream.ReadAsync(bytes);
            var count = decoder.GetChars(bytes, 0, received, buffer, 0, flush: received == 0);
            for (var index = 0; index < count; index++)
            {
                var character = buffer[index];
                if (character == '\n')
                {
                    await publish(line.ToString().TrimEnd('\r') + (oversized ? " [line truncated]" : ""), error);
                    line.Clear(); oversized = false;
                }
                else if (line.Length < MaximumLineCharacters) line.Append(character);
                else oversized = true;
            }
            if (received == 0) break;
        }
        if (line.Length > 0 || oversized) await publish(line.ToString() + (oversized ? " [line truncated]" : ""), error);
    }

    private static async Task<string?> CaptureFreshReport(string root, string reportDirectory, string runDirectory, IEnumerable<string> paths,
        Dictionary<string, (DateTime LastWrite, long Length)> prior, DateTimeOffset started, string operation,
        string modelId, string revision, bool cancelled, string[] sharedLatestPaths)
    {
        foreach (var candidate in paths.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            try
            {
                var path = Path.ChangeExtension(Path.GetFullPath(candidate, root), ".json");
                var synthetic = operation is "probe" or "mini" or "parity" or "storage-check";
                LocalFiles.Within(path, synthetic ? Path.Combine(root, "reports") : reportDirectory);
                if (cancelled && sharedLatestPaths.Contains(path, StringComparer.OrdinalIgnoreCase)) continue;
                if (!File.Exists(path)) continue;
                var info = new FileInfo(path);
                if (info.LastWriteTimeUtc < started.UtcDateTime || info.Length > MaximumReportBytes ||
                    prior.TryGetValue(path, out var old) && old == (info.LastWriteTimeUtc, info.Length)) continue;
                var bytes = await LocalFiles.ReadBoundedAsync(path, MaximumReportBytes);
                using var document = JsonDocument.Parse(bytes);
                if (document.RootElement.ValueKind != JsonValueKind.Object) continue;
                var report = document.RootElement;
                var needsIdentity = operation is "doctor" or "metadata-check" or "architecture-check" or "runtime-plan" or "generate" or "projection-check" or "tokenizer-check";
                if ((report.TryGetProperty("model_id", out var id) ? id.ValueKind != JsonValueKind.String || id.GetString() != modelId : needsIdentity) ||
                    (report.TryGetProperty("revision", out var rev) ? rev.ValueKind != JsonValueKind.String || rev.GetString() != revision : needsIdentity)) continue;
                var action = operation switch { "runtime-plan" => "plan", "projection-check" => "projection", "tokenizer-check" => "tokenizer", _ => operation };
                if (report.TryGetProperty("action", out var reportedAction) &&
                    (reportedAction.ValueKind != JsonValueKind.String || reportedAction.GetString() != action)) continue;
                var saved = Path.Combine(runDirectory, "report.json");
                await LocalFiles.AtomicWriteAsync(saved, bytes);
                return saved;
            }
            catch (Exception exception) when (exception is IOException or ArgumentException or JsonException or UnauthorizedAccessException) { }
        }
        return null;
    }

    private static IEnumerable<string> ExpectedReports(string root, string directory, string operation)
    {
        var filename = operation switch { "doctor" => "latest.json", "policy-check" => "policy-check.json",
            "monitor" => "gpu-observation.json", "runtime-plan" => "plan-latest.json", "metadata-check" => "metadata-latest.json",
            "architecture-check" => "architecture-latest.json", "projection-check" => "projection-latest.json",
            "tokenizer-check" => "tokenizer-latest.json", "generate" => "generate-latest.json",
            "probe" => "backend-probe-latest.json", "mini" => "mini-latest.json", "parity" => "parity-latest.json",
            "storage-check" => "storage-latest.json", _ => null };
        if (filename is not null)
        {
            yield return Path.Combine(directory, filename);
            if (operation is "probe" or "mini" or "parity" or "storage-check")
                if (directory != Path.Combine(root, "reports")) yield return Path.Combine(root, "reports", filename);
        }
    }

    private static string Root(string root)
    {
        if (string.IsNullOrWhiteSpace(root)) throw new DirectoryNotFoundException("Choose the folder containing glm_local/__main__.py in Settings.");
        root = LocalFiles.CheckPath(root);
        if (!File.Exists(Path.Combine(root, "glm_local", "__main__.py"))) throw new DirectoryNotFoundException("The selected project folder does not contain the Python core.");
        return root;
    }

    private static string RequiredString(JsonElement json, string key) => json.TryGetProperty(key, out var value) &&
        value.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(value.GetString()) ? value.GetString()!
        : throw new InvalidDataException($"Model profile is missing {key}.");

    private static string ReportDirectory(string root, JsonElement json)
    {
        var directory = Path.Combine(root, "reports");
        if (json.TryGetProperty("report_namespace", out var value))
        {
            var name = value.GetString() ?? "";
            if (name.Length is < 1 or > 64 || name.Any(character => !(char.IsAsciiLetterLower(character) || char.IsAsciiDigit(character) || character == '-')))
                throw new InvalidDataException("Profile report_namespace must be a portable lowercase slug.");
            directory = Path.Combine(directory, name);
        }
        return LocalFiles.Within(directory, Path.Combine(root, "reports"));
    }
}
