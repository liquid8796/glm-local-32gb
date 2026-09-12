using System.Windows.Threading;
using System.IO;
using ModelDesk.Core;
using ModelDesk.Desktop.ViewModels;

namespace ModelDesk.Tests;

public sealed class DesktopViewModelTests
{
    [Fact]
    public Task Restored_active_downloads_remain_paused_without_requests() => OnDispatcher(async () =>
    {
        var store = new MemoryQueue([new(Guid.NewGuid(), Request("a"), DownloadState.Downloading, 12), new(Guid.NewGuid(), Request("b"), DownloadState.Queued)]);
        var downloader = new ControlledDownloader();
        var model = new DownloadsViewModel(downloader, store, () => new());
        await model.InitializeAsync();
        Assert.All(model.Items, item => Assert.Equal(DownloadState.Paused, item.State));
        Assert.Equal(12, model.Items[0].DownloadedBytes); Assert.Empty(downloader.Started);
        Assert.All(store.Saved, item => Assert.Equal(DownloadState.Paused, item.State));
        await model.StopAsync();
    });

    [Fact]
    public Task Queue_respects_parallelism_and_runtime_speed_settings() => OnDispatcher(async () =>
    {
        var settings = new AppSettings { ConcurrentDownloads = 2, DownloadBytesPerSecond = 12345 };
        var downloader = new ControlledDownloader();
        var model = new DownloadsViewModel(downloader, new MemoryQueue(), () => settings);
        try
        {
            await model.EnqueueAsync([Request("a"), Request("b"), Request("c")]);
            await Until(() => downloader.Started.Count == 2);
            Assert.Equal(2, downloader.MaximumActive); Assert.Equal(12345, downloader.BytesPerSecondLimit);
            settings = settings with { ConcurrentDownloads = 1, DownloadBytesPerSecond = 6789 };
            await model.ApplySettingsAsync(); Assert.Equal(6789, downloader.BytesPerSecondLimit);
            downloader.Complete("a"); await Until(() => model.Items[0].State == DownloadState.Completed);
            Assert.Equal(2, downloader.Started.Count);
            downloader.Complete("b"); await Until(() => downloader.Started.Count == 3);
            downloader.Complete("c"); await Until(() => model.Items.All(item => item.State == DownloadState.Completed));
        }
        finally { await model.StopAsync(); }
    });

    [Fact]
    public Task Failed_disk_preflight_does_not_enqueue_or_download() => OnDispatcher(async () =>
    {
        var downloader = new ControlledDownloader { RejectBatch = true };
        var model = new DownloadsViewModel(downloader, new MemoryQueue(), () => new());
        await Assert.ThrowsAsync<IOException>(() => model.EnqueueAsync([Request("a"), Request("b")]));
        Assert.Empty(model.Items); Assert.Empty(downloader.Started);
        await model.StopAsync();
    });

    [Fact]
    public Task Pause_resume_commands_only_allow_valid_transitions() => OnDispatcher(async () =>
    {
        var downloader = new ControlledDownloader();
        var model = new DownloadsViewModel(downloader, new MemoryQueue(), () => new());
        try
        {
            await model.EnqueueAsync([Request("a")]); await Until(() => downloader.Started.Count == 1);
            var item = Assert.Single(model.Items);
            Assert.False(model.ResumeCommand.CanExecute(item)); Assert.True(model.PauseCommand.CanExecute(item));
            await model.PauseAllAsync(); await Until(() => downloader.Active == 0);
            Assert.True(model.ResumeCommand.CanExecute(item)); Assert.False(model.PauseCommand.CanExecute(item));
            downloader.RejectBatch = true;
            await Assert.ThrowsAsync<IOException>(() => model.ResumeAsync([item]));
            Assert.Equal(DownloadState.Paused, item.State); Assert.Single(downloader.Started);
        }
        finally { await model.StopAsync(); }
    });

    [Fact]
    public Task Cancel_all_does_not_start_queued_files_while_cancelling() => OnDispatcher(async () =>
    {
        var downloader = new ControlledDownloader();
        var model = new DownloadsViewModel(downloader, new MemoryQueue(), () => new() { ConcurrentDownloads = 1 });
        await model.EnqueueAsync([Request("a"), Request("b"), Request("c")]);
        await Until(() => downloader.Started.Count == 1);
        await model.CancelAllAsync(); await Until(() => downloader.Active == 0);
        Assert.Single(downloader.Started); Assert.All(model.Items, item => Assert.Equal(DownloadState.Cancelled, item.State));
        await model.StopAsync();
    });

    [Fact]
    public Task Shutdown_waits_for_cancelled_downloader_cleanup() => OnDispatcher(async () =>
    {
        var downloader = new ControlledDownloader { DelayCancellation = true };
        var model = new DownloadsViewModel(downloader, new MemoryQueue(), () => new());
        await model.EnqueueAsync([Request("a")]); await Until(() => downloader.Started.Count == 1);
        var stopping = model.StopAsync(); await downloader.CancellationObserved.Task.WaitAsync(TimeSpan.FromSeconds(2));
        Assert.False(stopping.IsCompleted);
        downloader.ReleaseCancellation.TrySetResult(); await stopping;
        Assert.Equal(0, downloader.Active); Assert.Equal(DownloadState.Paused, model.Items[0].State);
    });

    [Fact]
    public Task Duplicate_enqueue_requests_create_one_queue_item() => OnDispatcher(async () =>
    {
        var downloader = new ControlledDownloader();
        var model = new DownloadsViewModel(downloader, new MemoryQueue(), () => new());
        try
        {
            await Task.WhenAll(model.EnqueueAsync([Request("a"), Request("a")]), model.EnqueueAsync([Request("a")]));
            await Until(() => downloader.Started.Count == 1);
            Assert.Single(model.Items); Assert.Single(downloader.Started);
        }
        finally { await model.StopAsync(); }
    });

    [Fact]
    public Task Hub_bulk_selection_updates_totals_once_and_does_not_download() => OnDispatcher(async () =>
    {
        var downloader = new ControlledDownloader();
        var queue = new DownloadsViewModel(downloader, new MemoryQueue(), () => new());
        var hub = new HubViewModel(new FakeHub(), queue, () => new() { DefaultDownloadDirectory = Path.GetTempPath() });
        hub.OpenCommand.Execute(null); await Until(() => hub.Files.Count == 1000 && !hub.IsBusy);
        int notifications = 0;
        hub.PropertyChanged += (_, args) => { if (args.PropertyName == nameof(HubViewModel.SelectionSummary)) notifications++; };
        hub.SelectAllCommand.Execute(null);
        Assert.All(hub.Files, file => Assert.True(file.Selected)); Assert.Equal(1, notifications);
        hub.SelectNoneCommand.Execute(null); Assert.Equal(2, notifications); Assert.Contains("0 / 1000", hub.SelectionSummary);
        Assert.Empty(downloader.Started); await queue.StopAsync();
    });

    [Fact]
    public Task Workspace_initialization_loads_profiles_before_any_network_work() => OnDispatcher(async () =>
    {
        var root = Path.GetTempPath();
        var settings = new AppSettings { ProjectRoot = root, DefaultDownloadDirectory = root, PythonExecutable = "custom-python.exe", Profile = "nvfp4" };
        var downloader = new ControlledDownloader();
        var shell = new ShellViewModel(new MemorySettings(settings), new EmptyCredentials(), new FakeCore(), new EmptyReports(), new FakeHub(), downloader, new MemoryQueue());
        await shell.InitializeAsync();
        Assert.Equal(2, shell.Profiles.Count); Assert.Equal("nvfp4", shell.SelectedProfile?.Key);
        Assert.Equal("owner/nvfp4", shell.ModelId); Assert.Equal("custom-python.exe", shell.PythonPath);
        Assert.EndsWith("weights-nvfp4", shell.Run.ModelDirectory); Assert.Empty(shell.Error);
        Assert.Empty(downloader.Started); await shell.CloseAsync();
    });

    [Fact]
    public Task Settings_discovery_preserves_a_custom_python_executable() => OnDispatcher(async () =>
    {
        var current = new AppSettings { ProjectRoot = Path.Combine(Path.GetTempPath(), "not-a-core"), PythonExecutable = @"C:\custom-runtime\python.exe" };
        var found = new AppSettings { ProjectRoot = Path.GetTempPath(), PythonExecutable = "auto-python.exe", DefaultDownloadDirectory = Path.GetTempPath() };
        var model = new SettingsViewModel(new MemorySettings(current), new EmptyCredentials(), () => current, _ => Task.CompletedTask, () => found);
        model.Load(current); model.DiscoverCommand.Execute(null);
        Assert.Equal(current.PythonExecutable, model.PythonExecutable); Assert.Equal(found.ProjectRoot, model.ProjectRoot);
        await Task.CompletedTask;
    });

    private static DownloadRequest Request(string name) => new("owner/model", new string('a', 40), new(name, 100), Path.GetTempPath());
    private static async Task Until(Func<bool> predicate)
    {
        var deadline = DateTime.UtcNow.AddSeconds(3);
        while (!predicate()) { if (DateTime.UtcNow > deadline) throw new TimeoutException("View model did not reach the expected state."); await Task.Delay(10); }
    }
    private static async Task OnDispatcher(Func<Task> test)
    {
        var result = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var thread = new Thread(() =>
        {
            var dispatcher = Dispatcher.CurrentDispatcher;
            SynchronizationContext.SetSynchronizationContext(new DispatcherSynchronizationContext(dispatcher));
            dispatcher.BeginInvoke(async () =>
            {
                try { await test(); result.TrySetResult(); }
                catch (Exception exception) { result.TrySetException(exception); }
                finally { dispatcher.BeginInvokeShutdown(DispatcherPriority.Background); }
            });
            Dispatcher.Run();
        }) { IsBackground = true };
        thread.SetApartmentState(ApartmentState.STA); thread.Start();
        await result.Task.WaitAsync(TimeSpan.FromSeconds(12));
    }
    private sealed class MemoryQueue(IReadOnlyList<SavedDownload>? initial = null) : IDownloadQueueStore
    {
        public IReadOnlyList<SavedDownload> Saved { get; private set; } = initial ?? [];
        public Task<IReadOnlyList<SavedDownload>> LoadAsync(CancellationToken cancellationToken = default) => Task.FromResult(Saved);
        public Task SaveAsync(IReadOnlyList<SavedDownload> downloads, CancellationToken cancellationToken = default) { Saved = downloads.ToArray(); return Task.CompletedTask; }
    }
    private sealed class ControlledDownloader : IModelDownloader, IModelDownloadPlanner, IAdvancedDownloadOptions
    {
        private readonly Dictionary<string, TaskCompletionSource<DownloadResult>> pending = [];
        public List<string> Started { get; } = [];
        public int Active { get; private set; }
        public int MaximumActive { get; private set; }
        public bool RejectBatch { get; set; }
        public bool DelayCancellation { get; set; }
        public long BytesPerSecondLimit { get; set; }
        public int ConnectionsPerFile { get; set; } = 4;
        public long ParallelThresholdBytes { get; set; } = 64 * 1024 * 1024;
        public long SegmentSizeBytes { get; set; } = 16 * 1024 * 1024;
        public TaskCompletionSource CancellationObserved { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource ReleaseCancellation { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public Task<DownloadBatchEstimate> ValidateBatchAsync(IReadOnlyList<DownloadRequest> requests, CancellationToken cancellationToken = default)
            => RejectBatch ? Task.FromException<DownloadBatchEstimate>(new IOException("Insufficient batch disk space")) : Task.FromResult(new DownloadBatchEstimate([], requests.Count));
        public async Task<DownloadResult> DownloadAsync(DownloadRequest request, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
        {
            Started.Add(request.File.Path); Active++; MaximumActive = Math.Max(MaximumActive, Active);
            var completion = new TaskCompletionSource<DownloadResult>(TaskCreationOptions.RunContinuationsAsynchronously); pending[request.File.Path] = completion;
            try { return await completion.Task.WaitAsync(cancellationToken); }
            catch (OperationCanceledException) { CancellationObserved.TrySetResult(); if (DelayCancellation) await ReleaseCancellation.Task; throw; }
            finally { Active--; }
        }
        public void Complete(string name) => pending[name].TrySetResult(new(name, 100, true, false, 0, TimeSpan.FromSeconds(1)));
    }
    private sealed class FakeHub : IHuggingFaceClient
    {
        public Task<HubSearchResult> SearchAsync(string query, string? nextPage = null, CancellationToken cancellationToken = default) => Task.FromResult(new HubSearchResult([]));
        public Task<HubModelDetail> GetModelAsync(string modelId, string revision = "main", CancellationToken cancellationToken = default)
            => Task.FromResult(new HubModelDetail(modelId, new string('a', 40), Enumerable.Range(0, 1000).Select(index => new HubFile($"file-{index}.json", 100)).ToArray(), "Model metadata", null, null, false, false, 0, 0));
    }
    private sealed class MemorySettings(AppSettings settings) : ISettingsStore
    {
        public string FilePath => Path.Combine(Path.GetTempPath(), "modeldesk-test-settings.json");
        public Task<AppSettings> LoadAsync(CancellationToken cancellationToken = default) => Task.FromResult(settings);
        public Task SaveAsync(AppSettings value, CancellationToken cancellationToken = default) => Task.CompletedTask;
    }
    private sealed class EmptyCredentials : ICredentialStore
    {
        public Task<string?> GetTokenAsync(CancellationToken cancellationToken = default) => Task.FromResult<string?>(null);
        public Task SetTokenAsync(string? token, CancellationToken cancellationToken = default) => Task.CompletedTask;
    }
    private sealed class EmptyReports : IReportService
    {
        public IReadOnlyList<ReportEntry> List(string projectRoot, string? profileReportDirectory = null) => [];
        public Task<string> ReadAsync(string path, CancellationToken cancellationToken = default) => Task.FromResult("");
    }
    private sealed class FakeCore : IPythonCoreService
    {
        public IReadOnlyList<ModelProfile> GetProfiles(string projectRoot) => [
            new("nvfp4", "NVFP4", "owner/nvfp4", new string('a', 40), "config.json", "weights-nvfp4", "reports/nvfp4"),
            new("fp8", "FP8", "owner/fp8", new string('b', 40), "config.json", "weights-fp8", "reports")];
        public Task<CoreRunResult> RunAsync(CoreRunRequest request, IProgress<CoreOutput>? output = null, CancellationToken cancellationToken = default) => throw new NotSupportedException("Initialization must not launch Python.");
    }
}
