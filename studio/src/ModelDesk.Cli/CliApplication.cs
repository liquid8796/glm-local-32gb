using System.Globalization;
using System.Text.Json;
using ModelDesk.Core;

namespace ModelDesk.Cli;

public sealed class CliApplication(ISettingsStore settingsStore, IHuggingFaceClient hub, IModelDownloader downloader,
    IPythonCoreService core, IReportService reports, TextWriter? stdout = null, TextWriter? stderr = null)
{
    private readonly TextWriter _out = stdout ?? Console.Out;
    private readonly TextWriter _error = stderr ?? Console.Error;
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    public async Task<int> RunAsync(string[] args, CancellationToken cancellationToken = default)
    {
        var json = args.Length > 0 && args[0] == "--json";
        if (json) args = args[1..];
        // Everything after `core OPERATION` belongs to Python, including values
        // which happen to spell --json. Other commands accept a trailing flag.
        if (args.Length > 0 && args[0] != "core" && args[^1] == "--json") { json = true; args = args[..^1]; }
        try
        {
            if (args.Length == 0 || args[0] is "help" or "--help" or "-h") { await _out.WriteLineAsync(Help); return 0; }
            if (args[0] == "hub") return await HubAsync(args[1..], json, cancellationToken);
            if (args[0] == "download") return await DownloadAsync(args[1..], json, cancellationToken);
            if (args[0] == "settings") return await SettingsAsync(args[1..], json, cancellationToken);
            if (args[0] == "reports") return await ReportsAsync(args[1..], json, cancellationToken);
            if (args[0] == "profiles")
            {
                var settings = await settingsStore.LoadAsync(cancellationToken);
                await Write(core.GetProfiles(settings.ProjectRoot), json);
                return 0;
            }
            if (args[0] == "core")
            {
                if (args.Length < 2) throw new ArgumentException("Choose a core operation; run help for the list.");
                var settings = await settingsStore.LoadAsync(cancellationToken);
                var index = 1;
                string? profile = null, configPath = null;
                while (index < args.Length && args[index] is "--profile" or "--config")
                {
                    var option = args[index];
                    var value = Next(args, ref index);
                    if (profile is not null || configPath is not null) throw new ArgumentException("Choose one per-run --profile or --config option.");
                    if (option == "--profile") profile = value; else configPath = value;
                    index++;
                }
                if (index >= args.Length) throw new ArgumentException("Choose a core operation after the profile option.");
                if (profile is not null) settings = settings with { Profile = profile };
                CoreOperations.Get(args[index]);
                var progress = new InlineProgress<CoreOutput>(message => (json ? _error : _out).WriteLine(message.Text));
                var result = await core.RunAsync(new CoreRunRequest(settings, args[index], args[(index + 1)..], configPath), progress, cancellationToken);
                await Write(result, json);
                return result.Cancelled ? 130 : result.ExitCode;
            }
            throw new ArgumentException($"Unknown command: {args[0]}. Run help.");
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            await _error.WriteLineAsync("Cancelled.");
            return 130;
        }
        catch (Exception exception) when (exception is ArgumentException or InvalidOperationException or IOException or
                                           HttpRequestException or UnauthorizedAccessException or JsonException or TimeoutException or System.ComponentModel.Win32Exception)
        {
            if (json) await _out.WriteLineAsync(JsonSerializer.Serialize(new { error = exception.Message }, JsonOptions));
            else await _error.WriteLineAsync("Error: " + exception.Message);
            return 1;
        }
    }

    private async Task<int> HubAsync(string[] args, bool json, CancellationToken cancellationToken)
    {
        if (args.Length < 2) throw new ArgumentException("Usage: hub search QUERY | hub details ID [--revision REV]");
        if (args[0] == "search")
        {
            var result = await hub.SearchAsync(string.Join(' ', args[1..]), cancellationToken: cancellationToken);
            if (json) await Write(result, true);
            else foreach (var model in result.Models) await _out.WriteLineAsync($"{model.Id}\t{model.Downloads:N0} downloads\t{model.PipelineTag}");
            return 0;
        }
        if (args[0] == "details")
        {
            var revision = "main";
            for (var index = 2; index < args.Length; index++)
            {
                if (args[index] != "--revision") throw new ArgumentException("Only --revision is accepted for hub details.");
                revision = Next(args, ref index);
            }
            var result = await hub.GetModelAsync(args[1], revision, cancellationToken);
            if (json) await Write(result, true);
            else
            {
                await _out.WriteLineAsync($"{result.Id} @ {result.Revision}\n{result.Files.Count} files · {DisplayFormat.Bytes(result.TotalBytes)}");
                foreach (var file in result.Files) await _out.WriteLineAsync($"{DisplayFormat.Bytes(file.Size),12}  {file.Path}");
            }
            return 0;
        }
        throw new ArgumentException("Unknown hub command.");
    }

    private async Task<int> DownloadAsync(string[] args, bool json, CancellationToken cancellationToken)
    {
        if (args.Length == 0) throw new ArgumentException("Specify a model ID and --all or at least one --file.");
        var settings = await settingsStore.LoadAsync(cancellationToken);
        var folder = settings.DefaultDownloadDirectory;
        var revision = "main";
        var parallel = settings.ConcurrentDownloads;
        var connections = settings.DownloadConnectionsPerFile;
        var limit = settings.DownloadBytesPerSecond;
        var all = false;
        var names = new HashSet<string>(StringComparer.Ordinal);
        for (var index = 1; index < args.Length; index++)
        {
            switch (args[index])
            {
                case "--folder": folder = Next(args, ref index); break;
                case "--revision": revision = Next(args, ref index); break;
                case "--parallel": parallel = Integer(Next(args, ref index), 1, 8, "parallel"); break;
                case "--connections":
                case "--connections-per-file": connections = Integer(Next(args, ref index), 1, 4, "connections"); break;
                case "--limit-mib": limit = Rate(Next(args, ref index)); break;
                case "--all": all = true; break;
                case "--file": names.Add(Next(args, ref index)); break;
                default: throw new ArgumentException($"Unknown download option: {args[index]}");
            }
        }
        if (all == (names.Count > 0)) throw new ArgumentException("Choose either --all or one or more --file options.");
        if (string.IsNullOrWhiteSpace(folder)) throw new ArgumentException("Choose a destination with --folder.");
        folder = Path.GetFullPath(folder);
        var model = await hub.GetModelAsync(args[0], revision, cancellationToken);
        var selected = all ? model.Files : model.Files.Where(file => names.Contains(file.Path)).ToArray();
        if (!all && selected.Count != names.Count) throw new ArgumentException("A selected file is absent from the pinned model revision.");
        downloader.BytesPerSecondLimit = limit;
        if (downloader is IAdvancedDownloadOptions advanced) advanced.ConnectionsPerFile = connections;
        var requests = selected.Select(file => new DownloadRequest(model.Id, model.Revision, file, folder)).ToArray();
        if (downloader is IModelDownloadPlanner planner) await planner.ValidateBatchAsync(requests, cancellationToken);
        var results = new System.Collections.Concurrent.ConcurrentBag<DownloadResult>();
        var progress = new InlineProgress<DownloadProgress>(value =>
        {
            lock (_error) _error.WriteLine($"{value.FilePath}: {DisplayFormat.Bytes(value.DownloadedBytes)} / {DisplayFormat.Bytes(value.TotalBytes)} · {DisplayFormat.Speed(value.BytesPerSecond)} · {value.Stage}");
        });
        await Parallel.ForEachAsync(requests, new ParallelOptions { MaxDegreeOfParallelism = parallel, CancellationToken = cancellationToken },
            async (request, token) => results.Add(await downloader.DownloadAsync(request, progress, token)));
        await Write(results.OrderBy(result => result.FilePath, StringComparer.Ordinal).ToArray(), json);
        return 0;
    }

    private async Task<int> SettingsAsync(string[] args, bool json, CancellationToken cancellationToken)
    {
        var settings = await settingsStore.LoadAsync(cancellationToken);
        if (args.Length == 0 || args is ["show"]) { await Write(settings, true); return 0; }
        if (args.Length != 3 || args[0] != "set") throw new ArgumentException("Usage: settings show | settings set KEY VALUE");
        var value = args[2];
        settings = args[1] switch
        {
            "root" => settings with { ProjectRoot = Path.GetFullPath(value) },
            "python" => settings with { PythonExecutable = string.IsNullOrWhiteSpace(value)
                ? File.Exists(Path.Combine(settings.ProjectRoot, ".venv-reference", "Scripts", "python.exe"))
                    ? Path.Combine(settings.ProjectRoot, ".venv-reference", "Scripts", "python.exe") : "python"
                : value },
            "profile" => settings with { Profile = value },
            "download-folder" => settings with { DefaultDownloadDirectory = Path.GetFullPath(value) },
            "runtime-folder" => settings with { RuntimeModelDirectory = string.IsNullOrWhiteSpace(value) ? "" : Path.GetFullPath(value) },
            "parallel" => settings with { ConcurrentDownloads = Integer(value, 1, 8, "parallel") },
            "connections" => settings with { DownloadConnectionsPerFile = Integer(value, 1, 4, "connections") },
            "limit-mib" => settings with { DownloadBytesPerSecond = Rate(value) },
            "ram-mib" => settings with { RamBudgetBytes = checked((long)Integer(value, 1, 30517, "ram-mib") * 1024 * 1024) },
            "cpu-percent" => settings with { CpuLimitPercent = Integer(value, 1, 70, "cpu-percent") },
            "gpu-index" => settings with { GpuIndex = Integer(value, 0, 64, "gpu-index") },
            "gpu-percent" => settings with { GpuTargetPercent = Number(value, 1, 60, "gpu-percent") },
            "gpu-window" => settings with { GpuWindowSeconds = Integer(value, 1, 3600, "gpu-window") },
            "theme" => settings with { Theme = value },
            _ => throw new ArgumentException("Unknown settings key. Tokens are configured in the GUI or HF_TOKEN environment variable, never command-line values.")
        };
        settings.Validate();
        await settingsStore.SaveAsync(settings, cancellationToken);
        await Write(settings, json);
        return 0;
    }

    private async Task<int> ReportsAsync(string[] args, bool json, CancellationToken cancellationToken)
    {
        var settings = await settingsStore.LoadAsync(cancellationToken);
        var entries = reports.List(settings.ProjectRoot);
        if (args.Length == 0 || args is ["list"]) { await Write(entries, json); return 0; }
        if (args is ["read", var path])
        {
            path = Path.GetFullPath(path, settings.ProjectRoot);
            var text = await reports.ReadAsync(path, cancellationToken);
            if (json) await Write(new { path, text }, true); else await _out.WriteLineAsync(text);
            return 0;
        }
        throw new ArgumentException("Usage: reports list | reports read PATH");
    }

    private Task Write<T>(T value, bool json) => _out.WriteLineAsync(JsonSerializer.Serialize(value, JsonOptions));
    private static string Next(string[] args, ref int index) => ++index < args.Length ? args[index] : throw new ArgumentException("An option value is missing.");
    private static int Integer(string value, int minimum, int maximum, string name) => int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var number)
        && number >= minimum && number <= maximum ? number : throw new ArgumentException($"{name} must be {minimum}..{maximum}.");
    private static double Number(string value, double minimum, double maximum, string name) => double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var number)
        && double.IsFinite(number) && number >= minimum && number <= maximum ? number : throw new ArgumentException($"{name} must be {minimum}..{maximum}.");
    private static long Rate(string value) => (long)(Number(value, 0, 1_048_576, "limit-mib") * 1024 * 1024);

    private sealed class InlineProgress<T>(Action<T> callback) : IProgress<T> { public void Report(T value) => callback(value); }

    public const string Help = """
        ModelDesk CLI — Hugging Face downloads and the existing local Python core

        hub search QUERY
        hub details OWNER/MODEL [--revision REV]
        download OWNER/MODEL --folder DIR (--all | --file FILE [...])
                 [--revision REV] [--limit-mib MiB_PER_SECOND] [--parallel 1..8] [--connections 1..4]
        settings show
        settings set KEY VALUE
          Keys: root, python, profile, download-folder, runtime-folder, parallel, connections,
                limit-mib, ram-mib, cpu-percent, gpu-index, gpu-percent, gpu-window, theme
        profiles
        core [--profile KEY | --config FILE] OPERATION [PYTHON_ARGUMENTS...]
          doctor, policy-check, monitor, runtime-plan, generate, metadata-check,
          architecture-check, projection-check, tokenizer-check, probe, mini,
          parity, storage-check, build-native, setup-reference, test-reference
        reports list
        reports read PATH

        Prefix --json for structured output; progress goes to stderr. Ctrl+C cancels.
        Exit 2 from the core means review/blocked readiness and is preserved.
        Tokens: configure in the GUI or use HF_TOKEN; never put a token in arguments.
        Downloading a repository does not establish Python inference compatibility.
        """;
}
