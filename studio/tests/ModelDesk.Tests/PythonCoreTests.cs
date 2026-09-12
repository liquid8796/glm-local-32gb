using System.Diagnostics;
using System.Text.Json;
using ModelDesk.Core;
using ModelDesk.Infrastructure.Runtime;

namespace ModelDesk.Tests;

public sealed class PythonCoreTests : IDisposable
{
    private readonly Xunit.Abstractions.ITestOutputHelper _output;
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ModelDesk tests", Guid.NewGuid().ToString("N"), "core & quoted'");
    private readonly PythonCoreService _service = new();
    private readonly AppSettings _settings;
    private readonly string _profilePath;
    private readonly string _originalProfile;

    public PythonCoreTests(Xunit.Abstractions.ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(Path.Combine(_root, "glm_local"));
        Directory.CreateDirectory(Path.Combine(_root, "config", "models"));
        File.WriteAllText(Path.Combine(_root, "glm_local", "__init__.py"), "");
        File.WriteAllText(Path.Combine(_root, "glm_local", "__main__.py"), FakeCore);
        _profilePath = Path.Combine(_root, "config", "models", "abliterated-nvfp4.json");
        _originalProfile = """
            {"model_id":"example/synthetic","revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
             "model_directory":"models/original","report_namespace":"nvfp4","ram_budget_bytes":32000000000,
             "cpu_job_percent":70,"gpu_index":0,"gpu_average_target":0.6,"gpu_window_seconds":10,
             "disk_reserve_bytes":20000000000,"backend_status":"experimental_streaming_unverified"}
            """;
        File.WriteAllText(_profilePath, _originalProfile);
        _settings = new AppSettings { ProjectRoot = _root, PythonExecutable = FindPython(), Profile = "nvfp4" };
    }

    [Fact]
    public void ProfilesDiscoverFutureFilesAndUseDeclaredReportNamespace()
    {
        File.WriteAllText(Path.Combine(_root, "config", "models", "future-model.json"),
            _originalProfile.Replace("example/synthetic", "other/future", StringComparison.Ordinal));
        var profiles = _service.GetProfiles(_root);
        Assert.Equal("nvfp4", profiles[0].Key);
        Assert.Contains(profiles, profile => profile.Key == "future-model" && profile.ModelId == "other/future");
        Assert.Equal(Path.Combine(_root, "reports", "nvfp4"), profiles[0].ReportDirectory);
    }

    [Fact]
    public async Task ArgumentsResourcesUtf8AndExitTwoArePreservedWithoutChangingCoreConfig()
    {
        string[] arguments = ["--prompt", "Tiếng Việt ' \"quoted\" & | $() %NAME%", "--path", "C:\\model path\\", "--value", "a;b"];
        var settings = _settings with { RamBudgetBytes = 2_000_000_000, CpuLimitPercent = 25, GpuTargetPercent = 35,
            RuntimeModelDirectory = Path.Combine(_root, "different model") };
        var messages = new List<CoreOutput>();
        var result = await _service.RunAsync(new CoreRunRequest(settings, "doctor", arguments), new ProgressSink<CoreOutput>(messages.Add));
        Assert.Equal(2, result.ExitCode);
        Assert.Equal("Needs review", result.Status);
        Assert.False(result.Cancelled);
        Assert.Equal(_originalProfile, File.ReadAllText(_profilePath));
        Assert.NotNull(result.ReportPath);
        using var report = JsonDocument.Parse(await File.ReadAllTextAsync(result.ReportPath));
        Assert.Equal(arguments.Prepend("doctor"), report.RootElement.GetProperty("arguments").EnumerateArray().Select(value => value.GetString()));
        var config = report.RootElement.GetProperty("config");
        Assert.Equal(2_000_000_000, config.GetProperty("ram_budget_bytes").GetInt64());
        Assert.Equal(25, config.GetProperty("cpu_job_percent").GetInt32());
        Assert.Equal(0.35, config.GetProperty("gpu_average_target").GetDouble());
        Assert.Equal(settings.RuntimeModelDirectory, config.GetProperty("model_directory").GetString());
        Assert.Contains(messages, value => value.Text.Contains("Xin chào", StringComparison.Ordinal));
        Assert.Contains(messages, value => value.IsError && value.Text.Contains("Lỗi thử", StringComparison.Ordinal));
        Assert.Contains("modeldesk", result.ReportPath, StringComparison.Ordinal);
    }

    [Fact]
    public async Task OldReportIsNeverClaimedWhenProcessWritesNone()
    {
        var path = Path.Combine(_root, "reports", "nvfp4", "latest.json");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        await File.WriteAllTextAsync(path, "{\"status\":\"STALE\"}");
        File.SetLastWriteTimeUtc(path, DateTime.UtcNow.AddHours(-1));
        var result = await _service.RunAsync(new CoreRunRequest(_settings, "doctor", ["--no-report"]));
        Assert.Null(result.ReportPath);
        Assert.Contains("STALE", await File.ReadAllTextAsync(path));
    }

    [Theory]
    [InlineData("--wrong-identity")]
    [InlineData("--wrong-action")]
    [InlineData("--legacy-report")]
    public async Task FreshReportFromAnotherModelOperationOrNamespaceIsRejected(string argument)
    {
        var result = await _service.RunAsync(new CoreRunRequest(_settings, "doctor", [argument]));
        Assert.Equal(2, result.ExitCode);
        Assert.Null(result.ReportPath);
        Assert.Equal(_originalProfile, File.ReadAllText(_profilePath));
    }

    [Fact]
    public async Task TotalArgumentBytesAreBoundedBeforeStartingProcess()
    {
        await Assert.ThrowsAsync<ArgumentException>(() => _service.RunAsync(new CoreRunRequest(_settings, "doctor", [new string('語', 400000)])));
    }

    [Fact]
    public async Task CancellationKillsTheOwnedProcessTree()
    {
        using var cancellation = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        var watch = Stopwatch.StartNew();
        using var cancellationTrace = cancellation.Token.Register(() => _output.WriteLine($"Cancellation requested at {watch.Elapsed.TotalSeconds:F3}s"));
        var childId = 0;
        var progress = new ProgressSink<CoreOutput>(message =>
        {
            if (message.Text.StartsWith("CHILD:", StringComparison.Ordinal))
            {
                _output.WriteLine($"Child output received at {watch.Elapsed.TotalSeconds:F3}s");
                childId = int.Parse(message.Text[6..], System.Globalization.CultureInfo.InvariantCulture);
                cancellation.Cancel();
            }
        });
        var result = await _service.RunAsync(new CoreRunRequest(_settings, "doctor", ["--wait"]), progress, cancellation.Token);
        _output.WriteLine($"Run returned at {watch.Elapsed.TotalSeconds:F3}s, exit={result.ExitCode}, cancelled={result.Cancelled}");
        Assert.True(childId > 0);
        Assert.True(result.Cancelled);
        Assert.Equal(130, result.ExitCode);
        Assert.True(watch.Elapsed < TimeSpan.FromSeconds(8), "Cancellation must stop live child processes promptly, not await their natural exit.");
        try
        {
            using var child = Process.GetProcessById(childId);
            Assert.True(child.HasExited || child.WaitForExit(5000));
        }
        catch (ArgumentException) { /* Process no longer exists. */ }
    }

    [Fact]
    public async Task OversizedLinesAndDiskLogHaveFiniteBounds()
    {
        var result = await _service.RunAsync(new CoreRunRequest(_settings, "doctor", ["--large-log"]));
        var info = new FileInfo(result.LogPath);
        Assert.InRange(info.Length, 1, PythonCoreService.MaximumLogBytes);
        var text = await File.ReadAllTextAsync(result.LogPath);
        Assert.Contains("line truncated", text);
        Assert.Contains("bounded disk log", text);
    }

    [Fact]
    public async Task MissingPythonProducesAnActionableFailedRun()
    {
        var result = await _service.RunAsync(new CoreRunRequest(_settings with { PythonExecutable = Path.Combine(_root, "absent-python.exe") }, "doctor", []));
        Assert.Equal(1, result.ExitCode);
        Assert.Null(result.ReportPath);
        Assert.Contains("Core process failed", await File.ReadAllTextAsync(result.LogPath));
    }

    [Fact]
    public async Task ToolWrappersRejectUnrequestedShellArguments()
    {
        await Assert.ThrowsAsync<ArgumentException>(() => _service.RunAsync(new CoreRunRequest(_settings, "build-native", ["& malicious"]))) ;
        Assert.Equal(_originalProfile, File.ReadAllText(_profilePath));
    }

    [Fact]
    public void EscapingProfileReportNamespaceIsRejected()
    {
        File.WriteAllText(_profilePath, _originalProfile.Replace("\"nvfp4\"", "\"../outside\"", StringComparison.Ordinal));
        Assert.Throws<InvalidDataException>(() => _service.GetProfiles(_root));
    }

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, true);
    }

    private static string FindPython()
    {
        for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory is not null; directory = directory.Parent)
        {
            var executable = Path.Combine(directory.FullName, ".venv-reference", "Scripts", "python.exe");
            if (File.Exists(executable)) return executable;
        }
        return "python";
    }

    private sealed class ProgressSink<T>(Action<T> callback) : IProgress<T>
    {
        private readonly object _gate = new();
        public void Report(T value) { lock (_gate) callback(value); }
    }

    private const string FakeCore = """
        import json, os, pathlib, subprocess, sys, time
        config_path = pathlib.Path(sys.argv[sys.argv.index('--config') + 1])
        config = json.loads(config_path.read_text(encoding='utf-8'))
        arguments = sys.argv[sys.argv.index('--config') + 2:]
        print('Xin chào UTF-8', flush=True)
        print('Lỗi thử trên stderr', file=sys.stderr, flush=True)
        if '--wait' in arguments:
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(12)'])
            print('CHILD:' + str(child.pid), flush=True)
            time.sleep(12)
        if '--large-log' in arguments:
            for _ in range(530): print('A' * 20000)
        report = pathlib.Path.cwd() / 'reports' / 'nvfp4' / 'latest.json'
        if '--legacy-report' in arguments: report = pathlib.Path.cwd() / 'reports' / 'latest.json'
        if '--no-report' not in arguments:
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps({'status':'BLOCKED','arguments':arguments,'config':config,
                'model_id':'other/model' if '--wrong-identity' in arguments else config['model_id'],
                'revision':config['revision'],'action':'generate' if '--wrong-action' in arguments else 'doctor'}), encoding='utf-8')
        print('Report: ' + str(report), flush=True)
        sys.exit(2)
        """;
}
