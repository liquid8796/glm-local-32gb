using System.ComponentModel;
using System.Windows.Threading;
using ModelDesk.Core;
using ModelDesk.Desktop.ViewModels;
using ModelDesk.Infrastructure.Downloads;

namespace ModelDesk.Tests;

/// <summary>Hub workflows on an STA dispatcher, without windows or user files.</summary>
public sealed class HubWorkflowTests
{
    [Fact]
    public Task SelectingSearchResultAutomaticallyOpensItsFileList() => OnDispatcher(async () =>
    {
        await using var fixture = new Fixture();
        fixture.Hub.SelectedModel = Summary("owner/selected");
        await Until(() => fixture.Hub.Detail?.Id == "owner/selected" && !fixture.Hub.IsBusy && !fixture.Hub.IsScanning);
        Assert.Single(fixture.Client.Calls);
        Assert.Equal("owner/selected", fixture.Client.Calls[0].Id);
        Assert.Equal("main", fixture.Client.Calls[0].Revision);
        Assert.Equal(5, fixture.Hub.Files.Count);
        Assert.Empty(fixture.Downloader.Calls);
    });

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public Task LatestOpenWinsEvenWhenOldRequestIgnoresCancellation(bool oldFails) => OnDispatcher(async () =>
    {
        await using var fixture = new Fixture();
        fixture.Client.Controlled = true;
        var old = fixture.Hub.OpenModelAsync("owner/A", "old-tag");
        await Until(() => fixture.Client.Calls.Count == 1);
        var latest = fixture.Hub.OpenModelAsync("owner/B", "new-tag");
        await Until(() => fixture.Client.Calls.Count == 2);
        Assert.True(fixture.Client.Calls[0].Cancellation.IsCancellationRequested);
        fixture.Client.Calls[1].Completion.SetResult(Detail("owner/B", [RemoteFile("new.bin", 7)]));
        await latest;
        if (oldFails) fixture.Client.Calls[0].Completion.SetException(new IOException("Old request failed after replacement"));
        else fixture.Client.Calls[0].Completion.SetResult(Detail("owner/A", [RemoteFile("old.bin", 99)]));
        try { await old; }
        catch (OperationCanceledException) { }
        catch (IOException) when (oldFails) { }
        Assert.Equal("owner/B", fixture.Hub.Detail?.Id);
        Assert.Equal("new.bin", Assert.Single(fixture.Hub.Files).Path);
        Assert.Empty(fixture.Hub.Error);
        Assert.False(fixture.Hub.IsBusy);
        Assert.Contains("0 / 1", fixture.Hub.SelectionSummary);
    });

    [Fact]
    public Task OpenCommandCapturesFormIdAndRevisionBeforeAwaiting() => OnDispatcher(async () =>
    {
        await using var fixture = new Fixture();
        fixture.Client.Controlled = true;
        fixture.Hub.ModelId = " owner/form-model ";
        fixture.Hub.Revision = " release-v1 ";
        fixture.Hub.OpenCommand.Execute(null);
        await Until(() => fixture.Client.Calls.Count == 1);
        fixture.Hub.ModelId = "owner/edited-later";
        fixture.Hub.Revision = "release-v2";
        var captured = fixture.Client.Calls[0];
        Assert.Equal("owner/form-model", captured.Id);
        Assert.Equal("release-v1", captured.Revision);
        captured.Completion.SetResult(Detail(captured.Id, [RemoteFile("captured.bin", 4)]));
        await Until(() => fixture.Hub.Detail?.Id == captured.Id && !fixture.Hub.IsBusy);
        Assert.Equal(Revision, fixture.Hub.Detail?.Revision);
    });

    [Fact]
    public Task ChangingFolderCancelsOldScanAndOldCompletionCannotClearNewScanState() => OnDispatcher(async () =>
    {
        await using var fixture = new Fixture();
        await fixture.OpenAsync();
        fixture.Inventory.Controlled = true;
        var count = fixture.Inventory.Calls.Count;
        var old = fixture.Hub.RefreshLocalFilesAsync();
        await Until(() => fixture.Inventory.Calls.Count == count + 1);
        var first = fixture.Inventory.Calls[^1];
        var newDirectory = Path.Combine(fixture.Root, "different-folder");
        fixture.Hub.DownloadDirectory = newDirectory;
        await Until(() => fixture.Inventory.Calls.Count == count + 2);
        var latest = fixture.Inventory.Calls[^1];
        Assert.True(first.Cancellation.IsCancellationRequested);
        Assert.Equal(Path.GetFullPath(newDirectory), Path.GetFullPath(latest.Directory));
        first.Completion.SetResult(Statuses(first.Files, LocalModelFileState.Downloaded));
        try { await old; }
        catch (OperationCanceledException) { }
        Assert.True(fixture.Hub.IsScanning);
        latest.Completion.SetResult(Statuses(latest.Files, LocalModelFileState.Missing));
        await Until(() => !fixture.Hub.IsScanning);
        Assert.Equal(newDirectory, fixture.Hub.DownloadDirectory);
        Assert.All(fixture.Hub.Files, item => Assert.False(item.IsDownloaded));
    });

    [Fact]
    public Task CancellationAfterScanResultCompletionDoesNotLeaveRowsInspecting() => OnDispatcher(async () =>
    {
        await using var fixture = new Fixture();
        await fixture.OpenAsync();
        fixture.Inventory.Controlled = true;
        var scan = fixture.Hub.RefreshLocalFilesAsync();
        var pending = fixture.Inventory.Calls[^1];
        Assert.True(fixture.Hub.IsScanning);
        Assert.All(fixture.Hub.Files, row => Assert.True(row.IsInspecting));
        Assert.True(fixture.Hub.CancelCommand.CanExecute(null));

        // Both actions run in the same dispatcher turn. The asynchronous TCS
        // queues the scan continuation, then cancellation occurs before that
        // continuation can apply an otherwise successful inventory result.
        pending.Completion.SetResult(Statuses(pending.Files, LocalModelFileState.Downloaded));
        fixture.Hub.CancelCommand.Execute(null);
        Assert.True(pending.Cancellation.IsCancellationRequested);
        await scan;

        Assert.False(fixture.Hub.IsScanning);
        Assert.All(fixture.Hub.Files, row =>
        {
            Assert.False(row.IsInspecting);
            Assert.False(row.IsDownloaded);
            Assert.False(row.Selected);
        });

        fixture.Inventory.Controlled = false;
        await fixture.Hub.RefreshLocalFilesAsync();
        Assert.False(fixture.Hub.IsScanning);
        Assert.All(fixture.Hub.Files, row =>
        {
            Assert.False(row.IsInspecting);
            Assert.Equal(LocalModelFileState.Missing, row.LocalState);
            Assert.True(row.CanSelect);
        });
    });

    [Fact]
    public Task RefreshAndReopenDoNotReplaceAnExplicitCustomFolder() => OnDispatcher(async () =>
    {
        await using var fixture = new Fixture();
        await fixture.OpenAsync();
        var custom = Path.Combine(fixture.Root, "custom target");
        fixture.Hub.DownloadDirectory = custom;
        await fixture.Hub.RefreshLocalFilesAsync();
        await fixture.Hub.OpenModelAsync("owner/model", "main");
        Assert.Equal(custom, fixture.Hub.DownloadDirectory);
        Assert.Equal(Path.GetFullPath(custom), Path.GetFullPath(fixture.Inventory.Calls[^1].Directory));
    });

    [Fact]
    public Task ShiftRangeUsesVisibleSortOrderAndSkipsCompletedOrUnavailableRows() => OnDispatcher(async () =>
    {
        await using var fixture = new Fixture();
        fixture.Inventory.States["b.bin"] = LocalModelFileState.Downloaded;
        fixture.Inventory.States["d.bin"] = LocalModelFileState.Unavailable;
        await fixture.OpenAsync();
        fixture.Hub.HideDownloaded = true;
        fixture.Hub.FilesView.SortDescriptions.Clear();
        fixture.Hub.FilesView.SortDescriptions.Add(new(nameof(HubFileViewModel.Path), ListSortDirection.Ascending));
        fixture.Hub.FilesView.Refresh();
        Assert.Equal(new[] { "a.bin", "c.bin", "d.bin", "e.bin" }, fixture.Hub.FilesView.Cast<HubFileViewModel>().Select(item => item.Path));
        fixture.Hub.SelectRange(fixture.Row("a.bin"), true, false);
        fixture.Hub.SelectRange(fixture.Row("e.bin"), true, true);
        Assert.Equal(new[] { "a.bin", "c.bin", "e.bin" }, fixture.Hub.Files.Where(item => item.Selected).Select(item => item.Path).Order());
        Assert.False(fixture.Row("b.bin").Selected);
        Assert.False(fixture.Row("d.bin").Selected);
        Assert.Contains("3 / 5", fixture.Hub.SelectionSummary);
        fixture.Hub.SelectRange(fixture.Row("c.bin"), false, true);
        Assert.False(fixture.Row("a.bin").Selected);
        Assert.False(fixture.Row("c.bin").Selected);
        Assert.True(fixture.Row("e.bin").Selected);
        Assert.Contains("1 / 5", fixture.Hub.SelectionSummary);
    });

    [Fact]
    public Task BulkAndFilteredSelectionsKeepCountsCoherentWhenScanCompletesFiles() => OnDispatcher(async () =>
    {
        await using var fixture = new Fixture();
        await fixture.OpenAsync();
        fixture.Hub.Filter = "a.bin";
        fixture.Hub.SelectVisibleCommand.Execute(null);
        Assert.True(fixture.Row("a.bin").Selected);
        Assert.Contains("1 / 5", fixture.Hub.SelectionSummary);
        fixture.Hub.Filter = "c.bin";
        fixture.Hub.SelectVisibleCommand.Execute(null);
        Assert.Contains("2 / 5", fixture.Hub.SelectionSummary);
        fixture.Inventory.States["a.bin"] = LocalModelFileState.Downloaded;
        fixture.Inventory.States["c.bin"] = LocalModelFileState.Unavailable;
        await fixture.Hub.RefreshLocalFilesAsync();
        Assert.False(fixture.Row("a.bin").Selected);
        Assert.False(fixture.Row("c.bin").Selected);
        Assert.Contains("0 / 5", fixture.Hub.SelectionSummary);
        fixture.Hub.SelectAllCommand.Execute(null);
        Assert.Equal(3, fixture.Hub.Files.Count(item => item.Selected));
        Assert.Contains("3 / 5", fixture.Hub.SelectionSummary);
        fixture.Hub.SelectNoneCommand.Execute(null);
        Assert.Contains("0 / 5", fixture.Hub.SelectionSummary);
    });

    [Fact]
    public Task RemovedRowsCannotChangeNewModelsCachedSelectionTotals() => OnDispatcher(async () =>
    {
        await using var fixture = new Fixture();
        await fixture.OpenAsync();
        var old = fixture.Row("a.bin");
        old.Selected = true;
        fixture.Client.Files = [RemoteFile("new.bin", 10)];
        await fixture.Hub.OpenModelAsync("owner/new-model");
        old.Selected = false;
        old.Selected = true;
        Assert.Contains("0 / 1", fixture.Hub.SelectionSummary);
        Assert.False(Assert.Single(fixture.Hub.Files).Selected);
    });

    [Fact]
    public Task DownloadStatusFiltersKeepHiddenSelectionsAndCachedTotalsCoherent() => OnDispatcher(async () =>
    {
        await using var fixture = new Fixture();
        fixture.Inventory.States["b.bin"] = LocalModelFileState.Downloaded;
        fixture.Inventory.States["d.bin"] = LocalModelFileState.Partial;
        await fixture.OpenAsync();
        fixture.Row("a.bin").Selected = true;
        fixture.Hub.DownloadFilter = "Đã tải";
        fixture.Hub.FilesView.Refresh();
        Assert.Equal("b.bin", Assert.Single(fixture.Hub.FilesView.Cast<HubFileViewModel>()).Path);
        fixture.Hub.SelectVisibleCommand.Execute(null);
        Assert.Contains("1 / 5", fixture.Hub.SelectionSummary);
        fixture.Hub.DownloadFilter = "Chưa tải";
        fixture.Hub.FilesView.Refresh();
        fixture.Hub.SelectVisibleCommand.Execute(null);
        Assert.Contains("4 / 5", fixture.Hub.SelectionSummary);
        fixture.Inventory.States["c.bin"] = LocalModelFileState.Downloaded;
        await fixture.Hub.RefreshLocalFilesAsync();
        Assert.False(fixture.Row("c.bin").Selected);
        Assert.Equal(3, fixture.Hub.FilesView.Cast<HubFileViewModel>().Count());
        Assert.Contains("3 / 5", fixture.Hub.SelectionSummary);
    });

    [Fact]
    public Task EnqueueRefreshSkipsFilesWhichBecameDownloadedOrUnavailableAfterSelection() => OnDispatcher(async () =>
    {
        await using var fixture = new Fixture();
        await fixture.OpenAsync();
        fixture.Hub.SelectAllCommand.Execute(null);
        fixture.Inventory.States["a.bin"] = LocalModelFileState.Downloaded;
        fixture.Inventory.States["d.bin"] = LocalModelFileState.Unavailable;
        fixture.Hub.DownloadSelectedCommand.Execute(null);
        await Until(() => fixture.Queue.Items.Count > 0 && fixture.Queue.Items.All(item => item.State == DownloadState.Completed));
        Assert.Equal(new[] { "b.bin", "c.bin", "e.bin" }, fixture.Downloader.Calls.Select(request => request.File.Path).Order());
        Assert.All(fixture.Downloader.Calls, request =>
        {
            Assert.Equal("owner/model", request.ModelId);
            Assert.Equal(Revision, request.Revision);
            Assert.Equal(Path.GetFullPath(fixture.Hub.DownloadDirectory), request.DestinationDirectory);
        });
        Assert.False(fixture.Row("a.bin").Selected);
        Assert.False(fixture.Row("d.bin").Selected);
    });

    [Fact]
    public Task FailedScanForNewFolderNeverRetainsOldDownloadedMarks() => OnDispatcher(async () =>
    {
        await using var fixture = new Fixture();
        fixture.Inventory.States["a.bin"] = LocalModelFileState.Downloaded;
        await fixture.OpenAsync();
        Assert.True(fixture.Row("a.bin").IsDownloaded);
        fixture.Inventory.Failure = new IOException("Synthetic folder cannot be inspected");
        fixture.Hub.DownloadDirectory = Path.Combine(fixture.Root, "unreadable-folder");
        await fixture.Hub.RefreshLocalFilesAsync();
        Assert.All(fixture.Hub.Files, item => Assert.False(item.IsDownloaded));
        Assert.All(fixture.Hub.Files, item => Assert.False(item.Selected));
        Assert.False(fixture.Hub.IsScanning);
        Assert.Contains("Synthetic folder cannot be inspected", fixture.Hub.ScanError);
    });

    [Fact]
    public Task RealInventoryTreatsOnlyFinalExactSizeFileAsDownloadedAndHideable() => OnDispatcher(async () =>
    {
        await using var fixture = new Fixture(new LocalModelFileInventory());
        fixture.Client.Files = [RemoteFile("complete.bin", 4), RemoteFile("partial.bin", 4), RemoteFile("missing.bin", 4)];
        var directory = Path.Combine(fixture.Root, "real-scan");
        Directory.CreateDirectory(directory);
        await File.WriteAllBytesAsync(Path.Combine(directory, "complete.bin"), [1, 2, 3, 4]);
        await using (var partial = File.Create(Path.Combine(directory, "partial.bin.part"))) partial.SetLength(4);
        await File.WriteAllTextAsync(Path.Combine(directory, "partial.bin.part.json"), "not parsed by the size-only inventory");
        fixture.Hub.DownloadDirectory = directory;
        await fixture.OpenAsync();
        Assert.True(fixture.Row("complete.bin").IsDownloaded);
        Assert.False(fixture.Row("complete.bin").CanSelect);
        Assert.False(fixture.Row("partial.bin").IsDownloaded);
        Assert.True(fixture.Row("partial.bin").CanSelect);
        fixture.Hub.HideDownloaded = true;
        fixture.Hub.FilesView.Refresh();
        Assert.Equal(new[] { "missing.bin", "partial.bin" }, fixture.Hub.FilesView.Cast<HubFileViewModel>().Select(item => item.Path).Order());
        fixture.Hub.SelectVisibleCommand.Execute(null);
        Assert.False(fixture.Row("complete.bin").Selected);
        Assert.True(fixture.Row("partial.bin").Selected);
        Assert.Contains("2 / 3", fixture.Hub.SelectionSummary);
        Assert.Equal(4, new FileInfo(Path.Combine(directory, "partial.bin.part")).Length);
    });

    [Fact]
    public Task QueueCompletionRescansOnlyCurrentPinnedModelAndNormalizedDestination() => OnDispatcher(async () =>
    {
        await using var fixture = new Fixture();
        await fixture.OpenAsync();
        var detail = fixture.Hub.Detail!;
        var file = detail.Files[0];
        var initialScans = fixture.Inventory.Calls.Count;
        await fixture.Queue.EnqueueAsync([
            new("owner/unrelated", detail.Revision, file, fixture.Hub.DownloadDirectory),
            new(detail.Id, new string('c', 40), file, fixture.Hub.DownloadDirectory),
            new(detail.Id, detail.Revision, file, Path.Combine(fixture.Root, "other-destination"))]);
        await Until(() => fixture.Queue.Items.Count == 3 && fixture.Queue.Items.All(item => item.State == DownloadState.Completed));
        // Allow a debounce interval to elapse: unrelated completions must not
        // silently schedule a later scan of the currently displayed model.
        await Task.Delay(500);
        Assert.Equal(initialScans, fixture.Inventory.Calls.Count);
        fixture.Row(file.Path).Selected = true;
        fixture.Inventory.States[file.Path] = LocalModelFileState.Downloaded;
        await fixture.Queue.EnqueueAsync([new(detail.Id, detail.Revision, file, Path.Combine(fixture.Hub.DownloadDirectory, "."))]);
        await Until(() => fixture.Inventory.Calls.Count > initialScans && !fixture.Hub.IsScanning);
        Assert.True(fixture.Row(file.Path).IsDownloaded);
        Assert.False(fixture.Row(file.Path).Selected);
        Assert.Contains("0 / 5", fixture.Hub.SelectionSummary);
    });

    [Fact]
    public Task StopCancelsScheduledFolderScanAndUnsubscribesQueueCompletions() => OnDispatcher(async () =>
    {
        await using var fixture = new Fixture();
        await fixture.OpenAsync();
        fixture.Hub.DownloadDirectory = Path.Combine(fixture.Root, "changed-before-close");
        var detail = fixture.Hub.Detail!;
        var directory = fixture.Hub.DownloadDirectory;
        await fixture.Hub.StopAsync();
        var scansAfterStop = fixture.Inventory.Calls.Count;
        await fixture.Queue.EnqueueAsync([new(detail.Id, detail.Revision, detail.Files[0], directory)]);
        await Until(() => fixture.Queue.Items.Any(item => item.State == DownloadState.Completed));
        await Task.Delay(500);
        Assert.Equal(scansAfterStop, fixture.Inventory.Calls.Count);
        Assert.False(fixture.Hub.IsScanning);
    });

    [Fact]
    public Task StopCancelsPendingOpenAndIgnoresItsLateCompletion() => OnDispatcher(async () =>
    {
        await using var fixture = new Fixture();
        fixture.Client.Controlled = true;
        var opening = fixture.Hub.OpenModelAsync("owner/slow");
        await Until(() => fixture.Client.Calls.Count == 1);
        var pending = fixture.Client.Calls[0];
        var stopping = fixture.Hub.StopAsync();
        Assert.True(pending.Cancellation.IsCancellationRequested);
        pending.Completion.SetResult(Detail("owner/slow", [RemoteFile("late.bin", 1)]));
        try { await opening; }
        catch (OperationCanceledException) { }
        await stopping;
        Assert.Null(fixture.Hub.Detail);
        Assert.Empty(fixture.Hub.Files);
        Assert.False(fixture.Hub.IsBusy);
        Assert.False(fixture.Hub.IsScanning);
    });

    private static HubModelSummary Summary(string id) => new(id, "owner", 1, 1, null, null, false, false, null, []);
    private const string Revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    private static HubFile RemoteFile(string path, long size) => new(path, size, new string('b', 64));
    private static HubModelDetail Detail(string id, IReadOnlyList<HubFile> files) => new(id, Revision, files, "Synthetic model detail", null, null, false, false, 0, 0);
    private static IReadOnlyList<LocalModelFileStatus> Statuses(IReadOnlyList<HubFile> files, LocalModelFileState state) =>
        files.Select(file => new LocalModelFileStatus(file.Path, state, state == LocalModelFileState.Downloaded ? file.Size : null, "Synthetic inventory result")).ToArray();

    private static async Task Until(Func<bool> predicate)
    {
        var deadline = DateTime.UtcNow.AddSeconds(3);
        while (!predicate())
        {
            if (DateTime.UtcNow >= deadline) throw new TimeoutException("Hub workflow did not reach its expected state.");
            await Task.Delay(10);
        }
    }

    private static async Task OnDispatcher(Func<Task> test)
    {
        var completion = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var thread = new Thread(() =>
        {
            var dispatcher = Dispatcher.CurrentDispatcher;
            SynchronizationContext.SetSynchronizationContext(new DispatcherSynchronizationContext(dispatcher));
            dispatcher.BeginInvoke(async () =>
            {
                try { await test(); completion.TrySetResult(); }
                catch (Exception exception) { completion.TrySetException(exception); }
                finally { dispatcher.BeginInvokeShutdown(DispatcherPriority.Background); }
            });
            Dispatcher.Run();
        }) { IsBackground = true, Name = "ModelDesk Hub workflow tests" };
        thread.SetApartmentState(ApartmentState.STA);
        thread.Start();
        await completion.Task.WaitAsync(TimeSpan.FromSeconds(12));
    }

    private sealed class Fixture : IAsyncDisposable
    {
        public string Root { get; } = Path.Combine(Path.GetTempPath(), "ModelDesk-hub-workflows", Guid.NewGuid().ToString("N"));
        public ControlledHub Client { get; } = new();
        public ControlledInventory Inventory { get; } = new();
        public QueueDownloader Downloader { get; } = new();
        public DownloadsViewModel Queue { get; }
        public HubViewModel Hub { get; }
        public Fixture(ILocalModelFileInventory? inventory = null)
        {
            var settings = new AppSettings { DefaultDownloadDirectory = Root };
            Queue = new(Downloader, new MemoryQueue(), () => settings);
            Hub = new(Client, Queue, () => settings, inventory ?? Inventory);
        }
        public Task OpenAsync() => Hub.OpenModelAsync("owner/model");
        public HubFileViewModel Row(string name) => Hub.Files.Single(item => item.Path == name);
        public async ValueTask DisposeAsync()
        {
            foreach (var call in Client.Calls) call.Completion.TrySetCanceled();
            foreach (var call in Inventory.Calls) call.Completion.TrySetCanceled();
            await Hub.StopAsync();
            await Queue.StopAsync();
            var permitted = Path.GetFullPath(Path.Combine(Path.GetTempPath(), "ModelDesk-hub-workflows")) + Path.DirectorySeparatorChar;
            if (!Path.GetFullPath(Root).StartsWith(permitted, StringComparison.OrdinalIgnoreCase)) throw new IOException("Fixture cleanup escaped its private root.");
            if (Directory.Exists(Root)) Directory.Delete(Root, true);
        }
    }
    private sealed record ModelCall(string Id, string Revision, CancellationToken Cancellation)
    { public TaskCompletionSource<HubModelDetail> Completion { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously); }
    private sealed class ControlledHub : IHuggingFaceClient
    {
        public bool Controlled { get; set; }
        public IReadOnlyList<HubFile> Files { get; set; } = [RemoteFile("c.bin", 3), RemoteFile("a.bin", 1), RemoteFile("e.bin", 5), RemoteFile("b.bin", 2), RemoteFile("d.bin", 4)];
        public List<ModelCall> Calls { get; } = [];
        public Task<HubSearchResult> SearchAsync(string query, string? nextPage = null, CancellationToken cancellationToken = default) => Task.FromResult(new HubSearchResult([]));
        public Task<HubModelDetail> GetModelAsync(string modelId, string revision = "main", CancellationToken cancellationToken = default)
        {
            var call = new ModelCall(modelId, revision, cancellationToken);
            Calls.Add(call);
            if (!Controlled) call.Completion.SetResult(Detail(modelId, Files.ToArray()));
            return call.Completion.Task; // Deliberately non-cooperative: generation guards must still work.
        }
    }
    private sealed record ScanCall(string Directory, IReadOnlyList<HubFile> Files, CancellationToken Cancellation)
    { public TaskCompletionSource<IReadOnlyList<LocalModelFileStatus>> Completion { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously); }
    private sealed class ControlledInventory : ILocalModelFileInventory
    {
        public bool Controlled { get; set; }
        public Exception? Failure { get; set; }
        public Dictionary<string, LocalModelFileState> States { get; } = [];
        public List<ScanCall> Calls { get; } = [];
        public Task<IReadOnlyList<LocalModelFileStatus>> ScanAsync(string directory, IReadOnlyList<HubFile> files, CancellationToken cancellationToken = default)
        {
            var call = new ScanCall(directory, files.ToArray(), cancellationToken);
            Calls.Add(call);
            if (!Controlled)
            {
                if (Failure is not null) call.Completion.SetException(Failure);
                else call.Completion.SetResult(files.Select(file =>
                {
                    var state = States.GetValueOrDefault(file.Path, LocalModelFileState.Missing);
                    return new LocalModelFileStatus(file.Path, state,
                        state == LocalModelFileState.Downloaded ? file.Size : state == LocalModelFileState.Missing ? 0 : null,
                        "Synthetic inventory");
                }).ToArray());
            }
            return call.Completion.Task;
        }
    }
    private sealed class MemoryQueue : IDownloadQueueStore
    {
        public Task<IReadOnlyList<SavedDownload>> LoadAsync(CancellationToken cancellationToken = default) => Task.FromResult<IReadOnlyList<SavedDownload>>([]);
        public Task SaveAsync(IReadOnlyList<SavedDownload> downloads, CancellationToken cancellationToken = default) => Task.CompletedTask;
    }
    private sealed class QueueDownloader : IModelDownloader
    {
        public long BytesPerSecondLimit { get; set; }
        public List<DownloadRequest> Calls { get; } = [];
        public Task<DownloadResult> DownloadAsync(DownloadRequest request, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
        {
            Calls.Add(request);
            return Task.FromResult(new DownloadResult(Path.Combine(request.DestinationDirectory, request.File.Path), request.File.Size, true, false, 0, TimeSpan.Zero));
        }
    }
}
