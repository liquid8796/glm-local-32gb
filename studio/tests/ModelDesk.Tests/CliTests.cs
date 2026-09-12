using System.Text.Json;
using ModelDesk.Cli;
using ModelDesk.Core;

namespace ModelDesk.Tests;

public sealed class CliTests
{
    [Fact]
    public async Task CoreOverridePreservesLiteralArgumentsAndReadinessExitWithoutSavingSettings()
    {
        var fixtures = new Fixtures();
        var result = await fixtures.App.RunAsync(["--json", "core", "--profile", "fp8", "generate", "--prompt", "--json", "--profile", "literal"]);
        Assert.Equal(2, result);
        Assert.NotNull(fixtures.Core.Request);
        Assert.Equal("fp8", fixtures.Core.Request.Settings.Profile);
        Assert.Equal(new[] { "--prompt", "--json", "--profile", "literal" }, fixtures.Core.Request.Arguments);
        Assert.Equal("nvfp4", fixtures.Settings.Value.Profile);
        Assert.Equal(0, fixtures.Settings.Saves);
        using var document = JsonDocument.Parse(fixtures.Out.ToString());
        Assert.Equal(2, document.RootElement.GetProperty("ExitCode").GetInt32());
        Assert.Contains("core progress", fixtures.Error.ToString());
    }

    [Fact]
    public async Task CoreExplicitConfigAndLegacySyntaxAreBothAccepted()
    {
        var fixtures = new Fixtures();
        await fixtures.App.RunAsync(["core", "--config", "config/custom.json", "doctor"]);
        Assert.Equal("config/custom.json", fixtures.Core.Request!.ConfigPath);
        await fixtures.App.RunAsync(["core", "doctor", "--refresh"]);
        Assert.Equal(new[] { "--refresh" }, fixtures.Core.Request!.Arguments);
    }

    [Fact]
    public async Task DownloadPinsRevisionPreflightsWholeBatchAndKeepsProgressOutOfJson()
    {
        var fixtures = new Fixtures();
        var code = await fixtures.App.RunAsync(["--json", "download", "owner/model", "--folder", Path.GetTempPath(),
            "--all", "--limit-mib", "0.5", "--parallel", "1", "--connections-per-file", "3"]);
        Assert.Equal(0, code);
        Assert.Equal(524288, fixtures.Downloader.BytesPerSecondLimit);
        Assert.Equal(3, fixtures.Downloader.ConnectionsPerFile);
        Assert.Equal(2, fixtures.Downloader.Requests.Count);
        Assert.True(fixtures.Downloader.Validated);
        Assert.All(fixtures.Downloader.Requests, request => Assert.Equal(new string('a', 40), request.Revision));
        using var document = JsonDocument.Parse(fixtures.Out.ToString());
        Assert.Equal(2, document.RootElement.GetArrayLength());
        Assert.Contains("Downloading", fixtures.Error.ToString());
    }

    [Fact]
    public async Task BatchSpaceFailureDoesNotStartAnyFile()
    {
        var fixtures = new Fixtures();
        fixtures.Downloader.FailPreflight = true;
        Assert.Equal(1, await fixtures.App.RunAsync(["download", "owner/model", "--folder", Path.GetTempPath(), "--all"]));
        Assert.Empty(fixtures.Downloader.Requests);
    }

    [Theory]
    [InlineData("--all", "--file", "config.json")]
    [InlineData("--parallel", "9")]
    [InlineData("--connections", "5")]
    [InlineData("--limit-mib", "NaN")]
    public async Task InvalidDownloadOptionsFailBeforeModelOrPayloadRequests(params string[] options)
    {
        var fixtures = new Fixtures();
        Assert.Equal(1, await fixtures.App.RunAsync(["download", "owner/model", .. options]));
        Assert.Equal(0, fixtures.Hub.DetailCalls);
        Assert.Empty(fixtures.Downloader.Requests);
    }

    [Fact]
    public async Task UnknownSelectedFileCannotBeSilentlyOmitted()
    {
        var fixtures = new Fixtures();
        Assert.Equal(1, await fixtures.App.RunAsync(["download", "owner/model", "--folder", Path.GetTempPath(), "--file", "absent.json"]));
        Assert.Empty(fixtures.Downloader.Requests);
    }

    [Fact]
    public async Task RuntimeOverrideCanBeClearedAndTokenCommandLineIsRejected()
    {
        var fixtures = new Fixtures();
        fixtures.Settings.Value = fixtures.Settings.Value with { RuntimeModelDirectory = "D:\\custom-model" };
        Assert.Equal(0, await fixtures.App.RunAsync(["settings", "set", "runtime-folder", ""]));
        Assert.Equal("", fixtures.Settings.Value.RuntimeModelDirectory);
        Assert.Equal(1, await fixtures.App.RunAsync(["settings", "set", "token", "synthetic-private-value"]));
        Assert.DoesNotContain("synthetic-private-value", fixtures.Out.ToString());
        Assert.DoesNotContain("synthetic-private-value", fixtures.Error.ToString());
    }

    [Fact]
    public async Task ConnectionSettingPersistsAndRejectsValuesOutsideSupportedRange()
    {
        var fixtures = new Fixtures();
        Assert.Equal(0, await fixtures.App.RunAsync(["settings", "set", "connections", "2"]));
        Assert.Equal(2, fixtures.Settings.Value.DownloadConnectionsPerFile);
        var saves = fixtures.Settings.Saves;
        Assert.Equal(1, await fixtures.App.RunAsync(["settings", "set", "connections", "5"]));
        Assert.Equal(saves, fixtures.Settings.Saves);
        Assert.Equal(2, fixtures.Settings.Value.DownloadConnectionsPerFile);
    }

    [Fact]
    public async Task CancellationHasDefinedExitCode()
    {
        var fixtures = new Fixtures();
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        Assert.Equal(130, await fixtures.App.RunAsync(["hub", "search", "model"], cancellation.Token));
    }

    private sealed class Fixtures
    {
        public FakeSettings Settings { get; } = new();
        public FakeHub Hub { get; } = new();
        public FakeDownloader Downloader { get; } = new();
        public FakeCore Core { get; } = new();
        public StringWriter Out { get; } = new();
        public StringWriter Error { get; } = new();
        public CliApplication App => new(Settings, Hub, Downloader, Core, new FakeReports(), Out, Error);
    }

    private sealed class FakeSettings : ISettingsStore
    {
        public AppSettings Value { get; set; } = new() { ProjectRoot = Path.GetTempPath(), DefaultDownloadDirectory = Path.GetTempPath() };
        public int Saves { get; private set; }
        public string FilePath => "unused-settings.json";
        public Task<AppSettings> LoadAsync(CancellationToken cancellationToken = default) => Task.FromResult(Value);
        public Task SaveAsync(AppSettings settings, CancellationToken cancellationToken = default) { Value = settings; Saves++; return Task.CompletedTask; }
    }

    private sealed class FakeHub : IHuggingFaceClient
    {
        public int DetailCalls { get; private set; }
        public Task<HubSearchResult> SearchAsync(string query, string? nextPage = null, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Task.FromResult(new HubSearchResult([]));
        }
        public Task<HubModelDetail> GetModelAsync(string modelId, string revision = "main", CancellationToken cancellationToken = default)
        {
            DetailCalls++;
            return Task.FromResult(new HubModelDetail(modelId, new string('a', 40), [new("config.json", 10), new("tokenizer.json", 20)],
                "", "text-generation", "transformers", false, false, 1, 1));
        }
    }

    private sealed class FakeDownloader : IModelDownloader, IModelDownloadPlanner, IAdvancedDownloadOptions
    {
        public long BytesPerSecondLimit { get; set; }
        public int ConnectionsPerFile { get; set; }
        public long ParallelThresholdBytes { get; set; } = 64 * 1024 * 1024;
        public long SegmentSizeBytes { get; set; } = 16 * 1024 * 1024;
        public List<DownloadRequest> Requests { get; } = [];
        public bool Validated { get; private set; }
        public bool FailPreflight { get; set; }
        public Task<DownloadBatchEstimate> ValidateBatchAsync(IReadOnlyList<DownloadRequest> requests, CancellationToken cancellationToken = default)
        {
            if (FailPreflight) throw new IOException("Insufficient free disk space for the selected batch.");
            Validated = true;
            return Task.FromResult(new DownloadBatchEstimate([], requests.Count));
        }
        public Task<DownloadResult> DownloadAsync(DownloadRequest request, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
        {
            Assert.True(Validated);
            lock (Requests) Requests.Add(request);
            progress?.Report(new(request.File.Path, 1, request.File.Size, 10, null, "Downloading"));
            return Task.FromResult(new DownloadResult(request.File.Path, request.File.Size, true, false, 0, TimeSpan.Zero));
        }
    }

    private sealed class FakeCore : IPythonCoreService
    {
        public CoreRunRequest? Request { get; private set; }
        public IReadOnlyList<ModelProfile> GetProfiles(string projectRoot) => [];
        public Task<CoreRunResult> RunAsync(CoreRunRequest request, IProgress<CoreOutput>? output = null, CancellationToken cancellationToken = default)
        {
            Request = request;
            output?.Report(new(DateTimeOffset.UtcNow, "core progress"));
            return Task.FromResult(new CoreRunResult("id", 2, false, "output.log", null, DateTimeOffset.UtcNow, DateTimeOffset.UtcNow));
        }
    }

    private sealed class FakeReports : IReportService
    {
        public IReadOnlyList<ReportEntry> List(string projectRoot, string? profileReportDirectory = null) => [];
        public Task<string> ReadAsync(string path, CancellationToken cancellationToken = default) => Task.FromResult("{}");
    }
}
