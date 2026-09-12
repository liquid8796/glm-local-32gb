using System.Text.Json;
using System.Windows.Threading;
using ModelDesk.Core;
using ModelDesk.Desktop.ViewModels;
using ModelDesk.Infrastructure.Downloads;
using ModelDesk.Infrastructure.Reports;
using ModelDesk.Infrastructure.Runtime;
using ModelDesk.Infrastructure.Settings;

namespace ModelDesk.Tests;

/// <summary>
/// Exercises the real bounded-file services on a WPF dispatcher. No window is
/// created or shown. The former fake-profile GUI tests could not reproduce the
/// sync-over-async deadlock that hid the application before its main window.
/// </summary>
public sealed class StartupRegressionTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ModelDesk-startup-tests", Guid.NewGuid().ToString("N"));

    public StartupRegressionTests()
    {
        Directory.CreateDirectory(Path.Combine(_root, "glm_local"));
        File.WriteAllText(Path.Combine(_root, "glm_local", "__main__.py"), "# Unexecuted fixture marker.\n");
        var profiles = Path.Combine(_root, "config", "models");
        Directory.CreateDirectory(profiles);
        // Still below the public 64-KiB profile bound. Multiple actual async
        // file reads make the previously context-dependent deadlock observable.
        for (var index = 0; index < 16; index++)
        {
            var name = index == 0 ? "abliterated-nvfp4" : $"future-{index:00}";
            File.WriteAllText(Path.Combine(profiles, name + ".json"), JsonSerializer.Serialize(new
            {
                model_id = "example/startup-fixture", revision = new string('a', 40),
                model_directory = "models/fixture", report_namespace = "nvfp4", padding = new string('x', 60000)
            }));
        }
        var reportDirectory = Path.Combine(_root, "reports", "nvfp4");
        Directory.CreateDirectory(reportDirectory);
        for (var index = 0; index < 16; index++)
            File.WriteAllText(Path.Combine(reportDirectory, index == 0 ? "latest.json" : $"report-{index:00}.json"),
                JsonSerializer.Serialize(new { status = "BLOCKED", padding = new string('r', 60000) }));
    }

    [Fact]
    public async Task RealProfileDiscoveryCompletesOnPumpedStaDispatcher()
    {
        var profiles = await OnDispatcherAsync(() =>
        {
            Assert.IsType<DispatcherSynchronizationContext>(SynchronizationContext.Current);
            return Task.FromResult(new PythonCoreService().GetProfiles(_root));
        });
        Assert.Equal(16, profiles.Count);
        Assert.Equal("nvfp4", profiles[0].Key);
        Assert.Equal("example/startup-fixture", profiles[0].ModelId);
    }

    [Fact]
    public async Task RealReportListingCompletesOnPumpedStaDispatcher()
    {
        var reports = await OnDispatcherAsync(() =>
        {
            Assert.IsType<DispatcherSynchronizationContext>(SynchronizationContext.Current);
            return Task.FromResult(new ReportService().List(_root, Path.Combine(_root, "reports", "nvfp4")));
        });
        Assert.Equal(16, reports.Count);
        Assert.All(reports, report => Assert.Equal("BLOCKED", report.Status));
    }

    [Fact]
    public async Task WorkspaceInitializesRealSettingsProfilesAndReportsWithoutLeavingDispatcher()
    {
        var state = Path.Combine(_root, "state");
        var settings = new JsonSettingsStore(state);
        await settings.SaveAsync(new AppSettings { ProjectRoot = _root, Profile = "nvfp4", PythonExecutable = "python",
            DefaultDownloadDirectory = Path.Combine(_root, "models") });
        var snapshot = await OnDispatcherAsync(async () =>
        {
            var ownerThread = Environment.CurrentManagedThreadId;
            var changedThreads = new List<int>();
            var shell = new ShellViewModel(settings, new NoCredentials(), new PythonCoreService(), new ReportService(),
                new NoNetworkHub(), new NoNetworkDownloader(), new JsonDownloadQueueStore(state));
            var initializationStates = new List<bool>();
            shell.PropertyChanged += (_, args) =>
            {
                if (args.PropertyName == nameof(shell.IsInitializing)) initializationStates.Add(shell.IsInitializing);
            };
            shell.Profiles.CollectionChanged += (_, _) => changedThreads.Add(Environment.CurrentManagedThreadId);
            shell.Reports.Items.CollectionChanged += (_, _) => changedThreads.Add(Environment.CurrentManagedThreadId);
            await shell.InitializeAsync();
            Assert.Equal([true, false], initializationStates);
            Assert.False(shell.IsInitializing);
            Assert.IsType<DispatcherSynchronizationContext>(SynchronizationContext.Current);
            Assert.Equal(ownerThread, Environment.CurrentManagedThreadId);
            Assert.All(changedThreads, thread => Assert.Equal(ownerThread, thread));
            Assert.NotEmpty(changedThreads);
            Assert.Equal("", shell.Error);
            var result = (Profiles: shell.Profiles.Count, Reports: shell.Reports.Items.Count,
                ModelId: shell.ModelId, Readiness: shell.Readiness, Available: shell.Task.IsAvailable);
            await shell.CloseAsync();
            return result;
        });
        Assert.Equal(16, snapshot.Profiles);
        Assert.Equal(16, snapshot.Reports);
        Assert.Equal("example/startup-fixture", snapshot.ModelId);
        Assert.Equal("BLOCKED", snapshot.Readiness);
        Assert.True(snapshot.Available);
    }

    private static async Task<T> OnDispatcherAsync<T>(Func<Task<T>> body)
    {
        var completion = new TaskCompletionSource<T>(TaskCreationOptions.RunContinuationsAsynchronously);
        Dispatcher? dispatcher = null;
        var thread = new Thread(() =>
        {
            dispatcher = Dispatcher.CurrentDispatcher;
            SynchronizationContext.SetSynchronizationContext(new DispatcherSynchronizationContext(dispatcher));
            dispatcher.UnhandledException += (_, eventArgs) =>
            {
                eventArgs.Handled = true;
                completion.TrySetException(eventArgs.Exception);
                dispatcher.BeginInvokeShutdown(DispatcherPriority.Send);
            };
            dispatcher.BeginInvoke(new Action(async () =>
            {
                try { completion.TrySetResult(await body()); }
                catch (Exception exception) { completion.TrySetException(exception); }
                finally { dispatcher.BeginInvokeShutdown(DispatcherPriority.Send); }
            }));
            Dispatcher.Run();
        }) { IsBackground = true, Name = "ModelDesk real-service dispatcher regression" };
        thread.SetApartmentState(ApartmentState.STA);
        thread.Start();
        try { return await completion.Task.WaitAsync(TimeSpan.FromSeconds(5)); }
        catch (TimeoutException)
        {
            throw new Xunit.Sdk.XunitException("Real startup service blocked a pumped STA dispatcher for five seconds. Check for GetAwaiter().GetResult() over context-capturing file I/O.");
        }
        finally
        {
            // A regressed dispatcher may be blocked inside the service. Queue
            // shutdown but never join/abort that background thread and hang the
            // test host. Successful callbacks shut their dispatcher down above.
            if (dispatcher is { HasShutdownStarted: false }) dispatcher.BeginInvokeShutdown(DispatcherPriority.Send);
        }
    }

    public void Dispose()
    {
        var expected = Path.GetFullPath(Path.Combine(Path.GetTempPath(), "ModelDesk-startup-tests")) + Path.DirectorySeparatorChar;
        var target = Path.GetFullPath(_root);
        if (!target.StartsWith(expected, StringComparison.OrdinalIgnoreCase)) throw new IOException("Startup test cleanup escaped its private fixture root.");
        if (Directory.Exists(target)) Directory.Delete(target, true);
    }

    private sealed class NoCredentials : ICredentialStore
    {
        public Task<string?> GetTokenAsync(CancellationToken cancellationToken = default) => Task.FromResult<string?>(null);
        public Task SetTokenAsync(string? token, CancellationToken cancellationToken = default) => throw new InvalidOperationException("Startup must not set credentials.");
    }
    private sealed class NoNetworkHub : IHuggingFaceClient
    {
        public Task<HubSearchResult> SearchAsync(string query, string? nextPage = null, CancellationToken cancellationToken = default) =>
            throw new InvalidOperationException("Startup must not query the network.");
        public Task<HubModelDetail> GetModelAsync(string modelId, string revision = "main", CancellationToken cancellationToken = default) =>
            throw new InvalidOperationException("Startup must not query the network.");
    }
    private sealed class NoNetworkDownloader : IModelDownloader
    {
        public long BytesPerSecondLimit { get; set; }
        public Task<DownloadResult> DownloadAsync(DownloadRequest request, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default) =>
            throw new InvalidOperationException("Startup must not download files.");
    }
}
