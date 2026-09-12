using System.ComponentModel;
using System.Windows.Data;
using System.Windows.Threading;
using ModelDesk.Core;
using ModelDesk.Desktop.ViewModels;

namespace ModelDesk.Tests;

public sealed class QueueManagementTests
{
    [Fact]
    public Task Clear_during_startup_load_cannot_be_undone_by_the_old_saved_snapshot() => OnDispatcher(async () =>
    {
        var store = new DelayedLoadQueue([Saved("old-partial", DownloadState.Downloading, 35)]);
        var downloader = new ControlledDownloader();
        var model = new DownloadsViewModel(downloader, store, () => new());
        var initializing = model.InitializeAsync();
        try
        {
            await store.LoadEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));
            var clearing = model.ClearAsync();
            await Drain();
            Assert.True(model.IsClearing);
            Assert.False(clearing.IsCompleted);

            store.ReleaseLoad.TrySetResult();
            await Task.WhenAll(initializing, clearing); await Drain();
            Assert.Empty(model.Items); Assert.Empty(store.Saved); Assert.True(model.View.IsEmpty);
            Assert.Equal("0 / 0 tệp hoàn tất · 0 B / 0 B", model.Summary);
            Assert.Equal("0 B/s", model.Speed); Assert.False(model.IsClearing);
            Assert.Empty(downloader.Started);

            await model.EnqueueAsync([Request("new-after-startup-clear")]);
            await Until(() => downloader.Started.Count == 1);
            Assert.Equal("new-after-startup-clear", Assert.Single(model.Items).FileName);
        }
        finally { store.ReleaseLoad.TrySetResult(); await initializing; await model.StopAsync(); }
    });

    [Fact]
    public Task Clear_before_initialization_is_preserved_when_the_store_is_later_loaded() => OnDispatcher(async () =>
    {
        var store = new MemoryQueue([Saved("old", DownloadState.Paused, 35)]);
        var downloader = new ControlledDownloader();
        var model = new DownloadsViewModel(downloader, store, () => new());
        try
        {
            await model.ClearAsync(); await model.InitializeAsync();
            Assert.Empty(model.Items); Assert.Empty(store.Saved); Assert.True(model.View.IsEmpty);
            Assert.Empty(downloader.Started); Assert.Equal("0 B/s", model.Speed);
        }
        finally { await model.StopAsync(); }
    });

    [Fact]
    public Task Startup_load_finishing_after_stop_does_not_restore_rows_or_start_downloads() => OnDispatcher(async () =>
    {
        var store = new DelayedLoadQueue([Saved("old", DownloadState.Downloading, 35)]);
        var downloader = new ControlledDownloader();
        var model = new DownloadsViewModel(downloader, store, () => new());
        var initializing = model.InitializeAsync();
        try
        {
            await store.LoadEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));
            await model.StopAsync();
            store.ReleaseLoad.TrySetResult(); await initializing; await Drain();
            Assert.Empty(model.Items); Assert.Empty(store.Saved); Assert.Empty(downloader.Started);
            Assert.False(model.IsClearing);
        }
        finally { store.ReleaseLoad.TrySetResult(); await initializing; await model.StopAsync(); }
    });

    [Fact]
    public Task Clear_resets_queue_stats_selection_and_filter_without_touching_files() => OnDispatcher(async () =>
    {
        using var temporary = new TestDirectory();
        var final = Path.Combine(temporary.Path, "ready.bin");
        var partial = Path.Combine(temporary.Path, "incomplete.bin.part");
        var sidecar = partial + ".json";
        await File.WriteAllBytesAsync(final, [1, 2, 3, 4]);
        await File.WriteAllBytesAsync(partial, [5, 6, 7]);
        await File.WriteAllTextAsync(sidecar, "{\"fixture\":\"must remain unchanged\"}");
        var originals = new[] { final, partial, sidecar }.ToDictionary(path => path, File.ReadAllBytes);
        var downloader = new ControlledDownloader { RejectBatch = true };
        var store = new MemoryQueue([
            Saved("ready.bin", DownloadState.Completed, 4, temporary.Path),
            Saved("incomplete.bin", DownloadState.Paused, 3, temporary.Path)
        ]);
        var model = new DownloadsViewModel(downloader, store, () => new());
        try
        {
            await model.InitializeAsync();
            var paused = model.Items[1]; model.SelectedItem = paused;
            model.ResumeCommand.Execute(paused); await Until(() => model.Error.Contains("disk", StringComparison.Ordinal));
            model.StatusFilter = Filter(model, DownloadState.Failed);
            model.View.SortDescriptions.Clear(); model.View.SortDescriptions.Add(new(nameof(DownloadItemViewModel.ModelId), ListSortDirection.Descending));
            await model.ClearAsync();

            Assert.Empty(model.Items); Assert.True(model.View.IsEmpty); Assert.Empty(store.Saved);
            Assert.Equal("0 / 0 tệp hoàn tất · 0 B / 0 B", model.Summary);
            Assert.Equal("0 B/s", model.Speed); Assert.Equal("Hiển thị 0 / 0 tệp", model.VisibleSummary);
            Assert.Null(model.SelectedItem); Assert.Null(model.StatusFilter.State); Assert.Empty(model.Error);
            Assert.False(model.IsClearing); Assert.True(model.ClearCommand.CanExecute(null));
            Assert.Equal(nameof(DownloadItemViewModel.ModelId), Assert.Single(model.View.SortDescriptions).PropertyName);
            foreach (var pair in originals) Assert.Equal(pair.Value, await File.ReadAllBytesAsync(pair.Key));
            Assert.Equal(3, Directory.GetFiles(temporary.Path).Length);
            Assert.Empty(downloader.Started);
        }
        finally { await model.StopAsync(); }
    });

    [Fact]
    public Task Clear_awaits_active_cleanup_and_serializes_new_enqueue_and_stale_resume() => OnDispatcher(async () =>
    {
        var downloader = new ControlledDownloader { DelayCancellation = true };
        var store = new MemoryQueue();
        var model = new DownloadsViewModel(downloader, store, () => new() { ConcurrentDownloads = 1 });
        try
        {
            await model.EnqueueAsync([Request("active"), Request("queued")]);
            await Until(() => downloader.Started.Count == 1);
            var staleQueued = model.Items.Single(item => item.FileName == "queued");
            var clearing = model.ClearAsync();
            await downloader.CancellationObserved.Task.WaitAsync(TimeSpan.FromSeconds(2));
            Assert.True(model.IsClearing); Assert.False(clearing.IsCompleted);
            Assert.False(model.ClearCommand.CanExecute(null));
            Assert.False(model.ResumeAllCommand.CanExecute(null));
            var enqueueAfter = model.EnqueueAsync([Request("after-clear")]);
            var resumeStale = model.ResumeAsync([staleQueued]);
            Assert.False(enqueueAfter.IsCompleted); Assert.False(resumeStale.IsCompleted);
            Assert.Single(downloader.Started);

            downloader.ReleaseCancellation.TrySetResult();
            await clearing; await Task.WhenAll(enqueueAfter, resumeStale);
            await Until(() => downloader.Started.Count == 2);
            Assert.Equal(["active", "after-clear"], downloader.Started);
            Assert.Equal("after-clear", Assert.Single(model.Items).FileName);
            Assert.False(model.IsClearing);
            Assert.Contains(store.History, snapshot => snapshot.Count == 0);
        }
        finally { downloader.ReleaseCancellation.TrySetResult(); await model.StopAsync(); }
    });

    [Fact]
    public Task Late_progress_from_a_removed_download_cannot_restore_rows_or_stats() => OnDispatcher(async () =>
    {
        var downloader = new ControlledDownloader();
        var model = new DownloadsViewModel(downloader, new MemoryQueue(), () => new());
        try
        {
            await model.EnqueueAsync([Request("old")]); await Until(() => downloader.Started.Count == 1);
            var removed = Assert.Single(model.Items);
            downloader.Emit("old", 40, 200); await Drain();
            Assert.Equal(40, removed.DownloadedBytes);
            await model.ClearAsync();
            var reset = model.Summary;
            var updates = 0;
            model.PropertyChanged += (_, args) => { if (args.PropertyName is nameof(model.Summary) or nameof(model.Speed)) updates++; };
            // Even a stale external owner changing its row does not make it part of the queue again.
            removed.State = DownloadState.Downloading;
            downloader.Emit("old", 100, 9999); await Drain();
            Assert.Empty(model.Items); Assert.Equal(reset, model.Summary); Assert.Equal("0 B/s", model.Speed);
            Assert.Equal(0, updates); Assert.Equal(40, removed.DownloadedBytes);
        }
        finally { await model.StopAsync(); }
    });

    [Fact]
    public Task Concurrent_clear_calls_share_one_cleanup_operation() => OnDispatcher(async () =>
    {
        var downloader = new ControlledDownloader { DelayCancellation = true };
        var store = new MemoryQueue();
        var model = new DownloadsViewModel(downloader, store, () => new());
        try
        {
            await model.EnqueueAsync([Request("active")]); await Until(() => downloader.Started.Count == 1);
            var first = model.ClearAsync(); var second = model.ClearAsync();
            Assert.Same(first, second); Assert.False(first.IsCompleted);
            downloader.ReleaseCancellation.TrySetResult(); await Task.WhenAll(first, second);
            Assert.Empty(model.Items); Assert.Equal(1, store.History.Count(snapshot => snapshot.Count == 0));
        }
        finally { downloader.ReleaseCancellation.TrySetResult(); await model.StopAsync(); }
    });

    [Fact]
    public Task Clear_also_captures_enqueue_preflight_already_in_progress() => OnDispatcher(async () =>
    {
        var downloader = new ControlledDownloader { HoldFirstPreflight = true };
        var model = new DownloadsViewModel(downloader, new MemoryQueue(), () => new() { ConcurrentDownloads = 1 });
        try
        {
            var preceding = model.EnqueueAsync([Request("before-clear")]);
            await downloader.PreflightEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));
            var clearing = model.ClearAsync();
            var following = model.EnqueueAsync([Request("after-clear")]);
            downloader.ReleasePreflight.TrySetResult();
            await preceding; await clearing; await following;
            await Until(() => downloader.Started.Count == 1);
            Assert.Equal("after-clear", Assert.Single(downloader.Started));
            Assert.Equal("after-clear", Assert.Single(model.Items).FileName);
        }
        finally { downloader.ReleasePreflight.TrySetResult(); await model.StopAsync(); }
    });

    [Fact]
    public Task Every_status_filter_preserves_global_stats_and_reports_visible_count() => OnDispatcher(async () =>
    {
        var model = new DownloadsViewModel(new ControlledDownloader(), new MemoryQueue(), () => new());
        try
        {
            foreach (var state in Enum.GetValues<DownloadState>()) model.Items.Add(new(Saved(state.ToString(), state, 10)));
            var globalSummary = model.Summary;
            Assert.Equal(7, model.StatusFilters.Count);
            foreach (var option in model.StatusFilters)
            {
                model.StatusFilter = option; await Drain();
                var visible = model.View.Cast<DownloadItemViewModel>().ToArray();
                Assert.Equal(option.State is null ? 6 : 1, visible.Length);
                if (option.State is not null) Assert.All(visible, item => Assert.Equal(option.State.Value, item.State));
                Assert.Equal(globalSummary, model.Summary);
                Assert.Equal($"Hiển thị {visible.Length} / 6 tệp", model.VisibleSummary);
            }
        }
        finally { await model.StopAsync(); }
    });

    [Fact]
    public Task Status_changes_update_the_live_filter_without_manual_refresh() => OnDispatcher(async () =>
    {
        var model = new DownloadsViewModel(new ControlledDownloader(), new MemoryQueue(), () => new());
        try
        {
            var queued = new DownloadItemViewModel(Saved("queued", DownloadState.Queued));
            var active = new DownloadItemViewModel(Saved("active", DownloadState.Downloading));
            model.Items.Add(queued); model.Items.Add(active); model.StatusFilter = Filter(model, DownloadState.Downloading);
            Assert.Equal("active", Assert.Single(model.View.Cast<DownloadItemViewModel>()).FileName);
            queued.State = DownloadState.Downloading;
            await Until(() => model.View.Cast<DownloadItemViewModel>().Count() == 2);
            active.State = DownloadState.Completed;
            await Until(() => model.View.Cast<DownloadItemViewModel>().Count() == 1);
            Assert.Same(queued, Assert.Single(model.View.Cast<DownloadItemViewModel>()));
            Assert.StartsWith("1 / 2", model.Summary);
        }
        finally { await model.StopAsync(); }
    });

    [Fact]
    public Task Filter_refresh_waits_for_an_edit_transaction_to_finish() => OnDispatcher(async () =>
    {
        var model = new DownloadsViewModel(new ControlledDownloader(), new MemoryQueue(), () => new());
        try
        {
            var completed = new DownloadItemViewModel(Saved("completed", DownloadState.Completed));
            var paused = new DownloadItemViewModel(Saved("paused", DownloadState.Paused));
            model.Items.Add(completed); model.Items.Add(paused);
            var editing = Assert.IsAssignableFrom<IEditableCollectionView>(model.View);
            editing.EditItem(paused);
            model.StatusFilter = Filter(model, DownloadState.Completed);
            await Task.Delay(140);
            Assert.True(editing.IsEditingItem);
            editing.CommitEdit();
            await Until(() => model.View.Cast<DownloadItemViewModel>().Count() == 1);
            Assert.Same(completed, Assert.Single(model.View.Cast<DownloadItemViewModel>()));
        }
        finally { await model.StopAsync(); }
    });

    [Fact]
    public Task Filename_and_model_sorting_change_the_view_without_reordering_saved_items() => OnDispatcher(async () =>
    {
        var model = new DownloadsViewModel(new ControlledDownloader(), new MemoryQueue(), () => new());
        try
        {
            model.Items.Add(new(Saved("z.bin", DownloadState.Paused, modelId: "owner/a")));
            model.Items.Add(new(Saved("a.bin", DownloadState.Paused, modelId: "owner/c")));
            model.Items.Add(new(Saved("m.bin", DownloadState.Paused, modelId: "owner/b")));
            Assert.Equal(["a.bin", "m.bin", "z.bin"], model.View.Cast<DownloadItemViewModel>().Select(item => item.FileName));
            model.View.SortDescriptions.Clear(); model.View.SortDescriptions.Add(new(nameof(DownloadItemViewModel.ModelId), ListSortDirection.Ascending));
            Assert.Equal(["owner/a", "owner/b", "owner/c"], model.View.Cast<DownloadItemViewModel>().Select(item => item.ModelId));
            model.View.SortDescriptions.Clear(); model.View.SortDescriptions.Add(new(nameof(DownloadItemViewModel.FileName), ListSortDirection.Descending));
            Assert.Equal(["z.bin", "m.bin", "a.bin"], model.View.Cast<DownloadItemViewModel>().Select(item => item.FileName));
            Assert.Equal(["z.bin", "a.bin", "m.bin"], model.Items.Select(item => item.FileName));
        }
        finally { await model.StopAsync(); }
    });

    [Fact]
    public Task Successful_completion_emits_one_request_and_clear_cancellation_emits_none() => OnDispatcher(async () =>
    {
        var downloader = new ControlledDownloader();
        var model = new DownloadsViewModel(downloader, new MemoryQueue(), () => new() { ConcurrentDownloads = 1 });
        var completed = new List<DownloadRequest>(); model.DownloadCompleted += completed.Add;
        try
        {
            await model.EnqueueAsync([Request("ready"), Request("cancelled")]); await Until(() => downloader.Started.Count == 1);
            downloader.Complete("ready"); await Until(() => completed.Count == 1 && downloader.Started.Count == 2);
            await model.ClearAsync();
            Assert.Equal("ready", Assert.Single(completed).File.Path);
            await model.EnqueueAsync([Request("new")]); await Until(() => downloader.Started.Count == 3);
            downloader.Complete("new"); await Until(() => completed.Count == 2);
            Assert.Equal(["ready", "new"], completed.Select(request => request.File.Path));
        }
        finally { await model.StopAsync(); }
    });

    private static DownloadStatusFilter Filter(DownloadsViewModel model, DownloadState state) => model.StatusFilters.Single(filter => filter.State == state);
    private static DownloadRequest Request(string name, string? directory = null, string modelId = "owner/model") => new(modelId, new string('a', 40), new(name, 100), directory ?? Path.GetTempPath());
    private static SavedDownload Saved(string name, DownloadState state, long bytes = 0, string? directory = null, string modelId = "owner/model") => new(Guid.NewGuid(), Request(name, directory, modelId), state, bytes);
    private static async Task Until(Func<bool> predicate)
    {
        var deadline = DateTime.UtcNow.AddSeconds(4);
        while (!predicate()) { if (DateTime.UtcNow > deadline) throw new TimeoutException("Queue did not reach the expected state."); await Task.Delay(10); }
    }
    private static async Task Drain()
    {
        await Dispatcher.CurrentDispatcher.InvokeAsync(() => { }, DispatcherPriority.ApplicationIdle);
        await Dispatcher.CurrentDispatcher.InvokeAsync(() => { }, DispatcherPriority.ApplicationIdle);
    }
    private static async Task OnDispatcher(Func<Task> action)
    {
        var finished = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var thread = new Thread(() =>
        {
            var dispatcher = Dispatcher.CurrentDispatcher;
            SynchronizationContext.SetSynchronizationContext(new DispatcherSynchronizationContext(dispatcher));
            dispatcher.UnhandledException += (_, args) => { finished.TrySetException(args.Exception); args.Handled = true; dispatcher.BeginInvokeShutdown(DispatcherPriority.Background); };
            dispatcher.BeginInvoke(async () =>
            {
                try { await action(); finished.TrySetResult(); }
                catch (Exception exception) { finished.TrySetException(exception); }
                finally { if (!dispatcher.HasShutdownStarted) dispatcher.BeginInvokeShutdown(DispatcherPriority.Background); }
            });
            Dispatcher.Run();
        }) { IsBackground = true };
        thread.SetApartmentState(ApartmentState.STA); thread.Start();
        await finished.Task.WaitAsync(TimeSpan.FromSeconds(15));
    }
    private sealed class MemoryQueue(IReadOnlyList<SavedDownload>? initial = null) : IDownloadQueueStore
    {
        public IReadOnlyList<SavedDownload> Saved { get; private set; } = initial ?? [];
        public List<IReadOnlyList<SavedDownload>> History { get; } = [];
        public Task<IReadOnlyList<SavedDownload>> LoadAsync(CancellationToken cancellationToken = default) => Task.FromResult(Saved);
        public Task SaveAsync(IReadOnlyList<SavedDownload> items, CancellationToken cancellationToken = default)
        { Saved = items.ToArray(); History.Add(Saved); return Task.CompletedTask; }
    }
    private sealed class DelayedLoadQueue(IReadOnlyList<SavedDownload> snapshot) : IDownloadQueueStore
    {
        public IReadOnlyList<SavedDownload> Saved { get; private set; } = snapshot.ToArray();
        public TaskCompletionSource LoadEntered { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource ReleaseLoad { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public async Task<IReadOnlyList<SavedDownload>> LoadAsync(CancellationToken cancellationToken = default)
        {
            var frozen = snapshot.ToArray(); LoadEntered.TrySetResult();
            await ReleaseLoad.Task.WaitAsync(cancellationToken);
            return frozen;
        }
        public Task SaveAsync(IReadOnlyList<SavedDownload> items, CancellationToken cancellationToken = default)
        { Saved = items.ToArray(); return Task.CompletedTask; }
    }
    private sealed class ControlledDownloader : IModelDownloader, IModelDownloadPlanner
    {
        private readonly Dictionary<string, TaskCompletionSource<DownloadResult>> pending = [];
        private readonly Dictionary<string, IProgress<DownloadProgress>?> progress = [];
        private int preflights;
        public List<string> Started { get; } = [];
        public long BytesPerSecondLimit { get; set; }
        public bool RejectBatch { get; set; }
        public bool DelayCancellation { get; set; }
        public bool HoldFirstPreflight { get; set; }
        public TaskCompletionSource CancellationObserved { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource ReleaseCancellation { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource PreflightEntered { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource ReleasePreflight { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public async Task<DownloadBatchEstimate> ValidateBatchAsync(IReadOnlyList<DownloadRequest> requests, CancellationToken cancellationToken = default)
        {
            if (RejectBatch) throw new IOException("disk preflight failed");
            if (HoldFirstPreflight && ++preflights == 1) { PreflightEntered.TrySetResult(); await ReleasePreflight.Task; }
            return new([], requests.Count);
        }
        public async Task<DownloadResult> DownloadAsync(DownloadRequest request, IProgress<DownloadProgress>? reporter = null, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Started.Add(request.File.Path); progress[request.File.Path] = reporter;
            var result = new TaskCompletionSource<DownloadResult>(TaskCreationOptions.RunContinuationsAsynchronously); pending[request.File.Path] = result;
            try { return await result.Task.WaitAsync(cancellationToken); }
            catch (OperationCanceledException) { CancellationObserved.TrySetResult(); if (DelayCancellation) await ReleaseCancellation.Task; throw; }
        }
        public void Complete(string name) => pending[name].TrySetResult(new(name, 100, true, false, 0, TimeSpan.FromSeconds(1)));
        public void Emit(string name, long bytes, double speed) => progress[name]?.Report(new(name, bytes, 100, speed, TimeSpan.FromSeconds(1), "test progress"));
    }
    private sealed class TestDirectory : IDisposable
    {
        private static readonly string Boundary = System.IO.Path.GetFullPath(System.IO.Path.Combine(System.IO.Path.GetTempPath(), "ModelDesk.QueueManagementTests"));
        public string Path { get; } = System.IO.Path.Combine(Boundary, Guid.NewGuid().ToString("N"));
        public TestDirectory() => Directory.CreateDirectory(Path);
        public void Dispose()
        {
            var resolved = System.IO.Path.GetFullPath(Path);
            if (!resolved.StartsWith(Boundary + System.IO.Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase)) throw new IOException("Test cleanup escaped its fixture directory.");
            Directory.Delete(resolved, recursive: true);
        }
    }
}
