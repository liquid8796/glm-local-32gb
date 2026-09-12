using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Windows.Data;
using System.Windows.Input;
using System.Windows.Threading;
using ModelDesk.Core;
using ModelDesk.Desktop.Presentation;

namespace ModelDesk.Desktop.ViewModels;

public sealed record DownloadStatusFilter(string Label, DownloadState? State);

public sealed class DownloadItemViewModel(SavedDownload saved) : ObservableObject
{
    private DownloadState state = saved.State;
    private long bytes = saved.DownloadedBytes;
    private double speed;
    private string error = saved.Error ?? "", eta = "—";
    public Guid Id { get; } = saved.Id;
    public DownloadRequest Request { get; } = saved.Request;
    public string FileName => Request.File.Path;
    public string ModelId => Request.ModelId;
    public DownloadState State { get => state; set { Set(ref state, value); Raise(nameof(StateLabel)); } }
    public string StateLabel => State switch { DownloadState.Queued => "Đang chờ", DownloadState.Downloading => "Đang tải", DownloadState.Paused => "Tạm dừng", DownloadState.Completed => "Hoàn tất", DownloadState.Cancelled => "Đã hủy", _ => "Có lỗi" };
    public long DownloadedBytes { get => bytes; set { Set(ref bytes, value); Raise(nameof(Progress)); Raise(nameof(SizeLabel)); } }
    public double BytesPerSecond { get => speed; set { Set(ref speed, value); Raise(nameof(SpeedLabel)); } }
    public double Progress => Request.File.Size > 0 ? Math.Clamp(100d * DownloadedBytes / Request.File.Size, 0, 100) : 0;
    public string SizeLabel => $"{DisplayFormat.Bytes(DownloadedBytes)} / {DisplayFormat.Bytes(Request.File.Size)}";
    public string SpeedLabel => DisplayFormat.Speed(BytesPerSecond);
    public string Eta { get => eta; set => Set(ref eta, value); }
    public string Error { get => error; set => Set(ref error, value); }
    public SavedDownload Save() => new(Id, Request, State, DownloadedBytes, string.IsNullOrWhiteSpace(Error) ? null : Error);
}

public sealed class DownloadsViewModel : ObservableObject
{
    private readonly IModelDownloader downloader;
    private readonly IDownloadQueueStore store;
    private readonly Func<AppSettings> settings;
    private readonly Dictionary<Guid, CancellationTokenSource> active = [];
    private readonly Dictionary<Guid, Task> jobs = [];
    private readonly SemaphoreSlim persistence = new(1, 1), dispatch = new(1, 1), mutation = new(1, 1);
    private readonly HashSet<DownloadItemViewModel> observedItems = [];
    private readonly DispatcherTimer filterRefresh;
    private readonly bool liveFiltering;
    private string error = "";
    private bool stopping, clearing, batchingItems;
    private Task? clearingTask;
    private DownloadStatusFilter statusFilter;
    private DownloadItemViewModel? selectedItem;
    public DownloadsViewModel(IModelDownloader downloader, IDownloadQueueStore store, Func<AppSettings> settings)
    {
        this.downloader = downloader; this.store = store; this.settings = settings;
        statusFilter = StatusFilters[0];
        View = CollectionViewSource.GetDefaultView(Items);
        View.Filter = item => item is DownloadItemViewModel download && (StatusFilter.State is null || download.State == StatusFilter.State);
        View.SortDescriptions.Add(new(nameof(DownloadItemViewModel.FileName), ListSortDirection.Ascending));
        if (View is ICollectionViewLiveShaping { CanChangeLiveFiltering: true } live)
        {
            live.LiveFilteringProperties.Add(nameof(DownloadItemViewModel.State));
            live.IsLiveFiltering = true;
            liveFiltering = true;
        }
        View.CollectionChanged += (_, _) => { if (!batchingItems) { Raise(nameof(VisibleSummary)); Raise(nameof(EmptyText)); } };
        Items.CollectionChanged += ItemsChanged;
        filterRefresh = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(100) };
        filterRefresh.Tick += (_, _) => RefreshFilter();
        PauseCommand = new AsyncCommand(async value => { if (value is DownloadItemViewModel item) await SetStateAsync(item, DownloadState.Paused); }, Fail, value => !IsClearing && !stopping && value is DownloadItemViewModel { State: DownloadState.Queued or DownloadState.Downloading });
        ResumeCommand = new AsyncCommand(async value => { if (value is DownloadItemViewModel item) await ResumeAsync([item]); }, Fail, value => !IsClearing && !stopping && value is DownloadItemViewModel { State: DownloadState.Paused or DownloadState.Failed or DownloadState.Cancelled });
        CancelCommand = new AsyncCommand(async value => { if (value is DownloadItemViewModel item) await SetStateAsync(item, DownloadState.Cancelled); }, Fail, value => !IsClearing && !stopping && value is DownloadItemViewModel { State: not (DownloadState.Completed or DownloadState.Cancelled) });
        PauseAllCommand = new AsyncCommand(_ => PauseAllAsync(), Fail, _ => !IsClearing && !stopping);
        ResumeAllCommand = new AsyncCommand(_ => ResumeAsync(Items.Where(x => x.State is DownloadState.Paused or DownloadState.Failed).ToArray()), Fail, _ => !IsClearing && !stopping);
        CancelAllCommand = new AsyncCommand(_ => CancelAllAsync(), Fail, _ => !IsClearing && !stopping);
        ClearCommand = new AsyncCommand(_ => ClearAsync(), Fail, _ => !IsClearing && !stopping);
        OpenFolderCommand = new RelayCommand(value => { if (value is DownloadItemViewModel item) try { DesktopActions.OpenFolder(item.Request.DestinationDirectory); } catch (Exception exception) { Fail(exception); } });
    }
    public ObservableCollection<DownloadItemViewModel> Items { get; } = [];
    public ICollectionView View { get; }
    public IReadOnlyList<DownloadStatusFilter> StatusFilters { get; } = [
        new("Tất cả trạng thái", null), new("Đang chờ", DownloadState.Queued),
        new("Đang tải", DownloadState.Downloading), new("Tạm dừng", DownloadState.Paused),
        new("Hoàn tất", DownloadState.Completed), new("Có lỗi", DownloadState.Failed),
        new("Đã hủy", DownloadState.Cancelled)
    ];
    public DownloadStatusFilter StatusFilter
    {
        get => statusFilter;
        set { if (value is not null && Set(ref statusFilter, value)) RefreshFilter(); }
    }
    public DownloadItemViewModel? SelectedItem { get => selectedItem; set => Set(ref selectedItem, value); }
    public bool IsClearing { get => clearing; private set { Set(ref clearing, value); CommandManager.InvalidateRequerySuggested(); } }
    public string Error { get => error; private set => Set(ref error, value); }
    public string Summary => $"{Items.Count(x => x.State == DownloadState.Completed)} / {Items.Count} tệp hoàn tất · {DisplayFormat.Bytes(Items.Sum(x => x.DownloadedBytes))} / {DisplayFormat.Bytes(Items.Sum(x => x.Request.File.Size))}";
    public string Speed => DisplayFormat.Speed(Items.Sum(x => x.BytesPerSecond));
    public string VisibleSummary => $"Hiển thị {View.Cast<DownloadItemViewModel>().Count()} / {Items.Count} tệp";
    public string EmptyText => Items.Count == 0 ? "Chưa có tệp trong hàng đợi. Chọn model và tệp tại Hugging Face để bắt đầu."
        : View.IsEmpty ? "Không có tệp khớp trạng thái đã chọn." : "";
    public event Action<DownloadRequest>? DownloadCompleted;
    public ICommand PauseCommand { get; }
    public ICommand ResumeCommand { get; }
    public ICommand CancelCommand { get; }
    public ICommand PauseAllCommand { get; }
    public ICommand ResumeAllCommand { get; }
    public ICommand CancelAllCommand { get; }
    public ICommand ClearCommand { get; }
    public ICommand OpenFolderCommand { get; }
    private void Fail(Exception exception) => Error = exception.Message;
    private void Refresh() { Raise(nameof(Summary)); Raise(nameof(Speed)); Raise(nameof(EmptyText)); CommandManager.InvalidateRequerySuggested(); }
    private void ItemsChanged(object? sender, NotifyCollectionChangedEventArgs args)
    {
        if (args.Action == NotifyCollectionChangedAction.Reset)
        {
            foreach (var old in observedItems) old.PropertyChanged -= ItemChanged;
            observedItems.Clear();
            foreach (var item in Items) { observedItems.Add(item); item.PropertyChanged += ItemChanged; }
        }
        else if (args.Action != NotifyCollectionChangedAction.Move)
        {
            if (args.OldItems is not null)
                foreach (DownloadItemViewModel old in args.OldItems)
                    if (observedItems.Remove(old)) old.PropertyChanged -= ItemChanged;
            if (args.NewItems is not null)
                foreach (DownloadItemViewModel item in args.NewItems)
                    if (observedItems.Add(item)) item.PropertyChanged += ItemChanged;
        }
        if (!batchingItems) { Refresh(); Raise(nameof(VisibleSummary)); }
    }
    private void ItemChanged(object? sender, PropertyChangedEventArgs args)
    {
        if (args.PropertyName != nameof(DownloadItemViewModel.State)) return;
        Refresh();
        if (!liveFiltering) RefreshFilter();
    }
    private void RefreshFilter()
    {
        if (View is IEditableCollectionView editable && (editable.IsEditingItem || editable.IsAddingNew))
        { filterRefresh.Start(); return; }
        filterRefresh.Stop();
        View.Refresh(); Raise(nameof(VisibleSummary)); Raise(nameof(EmptyText));
    }
    public async Task InitializeAsync()
    {
        // A clear/enqueue request must not overtake a previously started disk load
        // and then have its result replaced by that load's older snapshot.
        await mutation.WaitAsync();
        try
        {
            if (stopping) return;
            var savedItems = await store.LoadAsync();
            if (stopping) return;
            batchingItems = true;
            try { foreach (var saved in savedItems) Items.Add(new(saved.State is DownloadState.Queued or DownloadState.Downloading ? saved with { State = DownloadState.Paused } : saved)); }
            finally { batchingItems = false; }
            Refresh(); Raise(nameof(VisibleSummary)); await SaveAsync();
        }
        finally { mutation.Release(); }
    }
    public async Task EnqueueAsync(IEnumerable<DownloadRequest> requests)
    {
        await mutation.WaitAsync();
        try
        {
            if (stopping) throw new InvalidOperationException("Ứng dụng đang đóng hàng đợi tải.");
            Error = "";
            var existing = Items.Where(item => item.State is not (DownloadState.Cancelled or DownloadState.Failed)).Select(item => item.Request).ToHashSet();
            var additions = requests.Distinct().Where(request => !existing.Contains(request)).ToArray();
            await ValidateBatchAsync(additions);
            batchingItems = true;
            try { foreach (var request in additions) Items.Add(new(new(Guid.NewGuid(), request, DownloadState.Queued))); }
            finally { batchingItems = false; }
            Refresh(); Raise(nameof(VisibleSummary)); await SaveAsync();
        }
        finally { mutation.Release(); }
        await PumpAsync();
    }
    public async Task ApplySettingsAsync()
    {
        downloader.BytesPerSecondLimit = settings().DownloadBytesPerSecond;
        if (downloader is IAdvancedDownloadOptions options) options.ConnectionsPerFile = settings().DownloadConnectionsPerFile;
        await PumpAsync();
    }
    private async Task PumpAsync()
    {
        await dispatch.WaitAsync();
        try
        {
            if (stopping || IsClearing) return;
            downloader.BytesPerSecondLimit = settings().DownloadBytesPerSecond;
            if (downloader is IAdvancedDownloadOptions options) options.ConnectionsPerFile = settings().DownloadConnectionsPerFile;
            while (active.Count < settings().ConcurrentDownloads)
            {
                var item = Items.FirstOrDefault(x => x.State == DownloadState.Queued && !active.ContainsKey(x.Id));
                if (item is null) break;
                var cancellation = new CancellationTokenSource(); active.Add(item.Id, cancellation); item.State = DownloadState.Downloading;
                jobs[item.Id] = RunAsync(item, cancellation);
            }
        }
        finally { dispatch.Release(); Refresh(); }
    }
    private async Task RunAsync(DownloadItemViewModel item, CancellationTokenSource cancellation)
    {
        await Task.Yield(); // Register the job before even a synchronous fake downloader can finish.
        try
        {
            cancellation.Token.ThrowIfCancellationRequested();
            await SaveAsync();
            cancellation.Token.ThrowIfCancellationRequested();
            long lastUpdate = 0;
            var result = await downloader.DownloadAsync(item.Request, new Progress<DownloadProgress>(progress =>
            {
                if (IsClearing || item.State != DownloadState.Downloading || !active.TryGetValue(item.Id, out var owner)
                    || !ReferenceEquals(owner, cancellation)) return;
                var now = Environment.TickCount64;
                if (now - lastUpdate < 150 && progress.DownloadedBytes != progress.TotalBytes) return;
                lastUpdate = now;
                item.DownloadedBytes = progress.DownloadedBytes; item.BytesPerSecond = progress.BytesPerSecond;
                item.Eta = progress.Remaining is { } remaining ? (remaining.TotalHours >= 1 ? remaining.ToString(@"d\.hh\:mm\:ss") : remaining.ToString(@"mm\:ss")) : "—";
                Refresh();
            }), cancellation.Token);
            item.DownloadedBytes = result.Bytes; item.State = DownloadState.Completed;
            item.Error = result.HashVerified ? "Đã xác minh hash" : "Đã hoàn tất";
            try { DownloadCompleted?.Invoke(item.Request); } catch (Exception exception) { Fail(exception); }
        }
        catch (OperationCanceledException) { if (item.State == DownloadState.Downloading) item.State = DownloadState.Paused; }
        catch (Exception exception) { item.State = DownloadState.Failed; item.Error = exception.Message; }
        finally
        {
            item.BytesPerSecond = 0; item.Eta = "—"; active.Remove(item.Id); cancellation.Dispose(); Refresh();
            try { await SaveAsync(); await PumpAsync(); } catch (Exception exception) { Fail(exception); }
            finally { jobs.Remove(item.Id); }
        }
    }
    private async Task SetStateAsync(DownloadItemViewModel item, DownloadState state)
    {
        if (IsClearing || item.State == DownloadState.Completed || !Items.Contains(item)) return;
        item.State = state;
        if (active.TryGetValue(item.Id, out var cancellation)) cancellation.Cancel();
        Refresh(); await SaveAsync();
    }
    public async Task PauseAllAsync()
    {
        if (IsClearing && !stopping) return;
        foreach (var item in Items.Where(x => x.State is DownloadState.Queued or DownloadState.Downloading).ToArray())
        { item.State = DownloadState.Paused; if (active.TryGetValue(item.Id, out var cancellation)) cancellation.Cancel(); }
        Refresh(); await SaveAsync();
    }
    public async Task CancelAllAsync()
    {
        if (IsClearing) return;
        var selected = Items.Where(x => x.State is not (DownloadState.Completed or DownloadState.Cancelled)).ToArray();
        foreach (var item in selected) item.State = DownloadState.Cancelled;
        foreach (var item in selected) if (active.TryGetValue(item.Id, out var cancellation)) cancellation.Cancel();
        Refresh(); await SaveAsync();
    }
    public async Task ResumeAsync(IReadOnlyList<DownloadItemViewModel> selected)
    {
        await mutation.WaitAsync();
        try
        {
            if (stopping) return;
            var resumable = selected.Where(item => Items.Contains(item) && item.State is DownloadState.Paused or DownloadState.Failed or DownloadState.Cancelled).ToArray();
            if (resumable.Length == 0) return;
            await ValidateBatchAsync(resumable.Select(item => item.Request));
            foreach (var item in resumable) { item.State = DownloadState.Queued; item.Error = ""; }
            Refresh(); await SaveAsync();
        }
        finally { mutation.Release(); }
        await PumpAsync();
    }
    private async Task ValidateBatchAsync(IEnumerable<DownloadRequest> additions)
    {
        if (downloader is IModelDownloadPlanner planner)
            await planner.ValidateBatchAsync(Items.Where(item => item.State is DownloadState.Queued or DownloadState.Downloading).Select(item => item.Request).Concat(additions).Distinct().ToArray());
    }
    public Task ClearAsync()
    {
        if (clearingTask is { IsCompleted: false }) return clearingTask;
        IsClearing = true;
        return clearingTask = ClearCoreAsync();
    }
    private async Task ClearCoreAsync()
    {
        await mutation.WaitAsync();
        try
        {
            await dispatch.WaitAsync();
            try
            {
                foreach (var item in Items.Where(item => item.State is DownloadState.Queued or DownloadState.Downloading))
                    item.State = DownloadState.Cancelled;
                foreach (var cancellation in active.Values.ToArray()) cancellation.Cancel();
            }
            finally { dispatch.Release(); }
            while (jobs.Count > 0) await Task.WhenAll(jobs.Values.ToArray());
            filterRefresh.Stop();
            if (View is IEditableCollectionView editable)
            {
                if (editable.IsAddingNew) editable.CancelNew();
                if (editable.IsEditingItem) editable.CommitEdit();
            }
            Items.Clear(); SelectedItem = null; StatusFilter = StatusFilters[0];
            Error = ""; Refresh(); Raise(nameof(VisibleSummary));
            await SaveAsync();
        }
        finally { mutation.Release(); IsClearing = false; }
    }
    public async Task StopAsync()
    {
        stopping = true;
        if (clearingTask is { IsCompleted: false }) await clearingTask;
        await PauseAllAsync();
        while (jobs.Count > 0) await Task.WhenAll(jobs.Values.ToArray());
        filterRefresh.Stop();
        await SaveAsync();
    }
    private async Task SaveAsync()
    {
        await persistence.WaitAsync();
        try { await store.SaveAsync(Items.Select(item => item.Save()).ToArray()); }
        finally { persistence.Release(); }
    }
}
