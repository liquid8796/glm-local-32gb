using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Windows.Data;
using System.Windows.Input;
using ModelDesk.Core;
using ModelDesk.Desktop.Presentation;

namespace ModelDesk.Desktop.ViewModels;

public sealed class HubFileViewModel(HubFile file) : ObservableObject
{
    private bool selected, inspecting;
    private LocalModelFileState localState;
    private string localMessage = "Chưa quét trạng thái tệp cục bộ.";
    public HubFile File { get; } = file;
    public string Path => File.Path;
    public string Size => DisplayFormat.Bytes(File.Size);
    public string Hash => File.LfsSha256 ?? File.BlobId ?? "Không có hash";
    public bool Selected { get => selected; set => Set(ref selected, value && CanSelect); }
    public LocalModelFileState LocalState => localState;
    public bool IsDownloaded => localState == LocalModelFileState.Downloaded;
    public bool IsInspecting => inspecting;
    public bool CanSelect => !inspecting && !IsDownloaded && localState != LocalModelFileState.Unavailable;
    public bool ShowSelection => !IsDownloaded;
    public string LocalMessage => localMessage;
    public string LocalStateLabel => inspecting ? "Đang quét…" : localState switch
    {
        LocalModelFileState.Downloaded => "Đã tải",
        LocalModelFileState.Partial => "Tải dở",
        LocalModelFileState.SizeMismatch => "Sai dung lượng",
        LocalModelFileState.Unavailable => "Không kiểm tra được",
        _ => "Chưa tải"
    };
    internal void Inspect(bool invalidate)
    {
        inspecting = true;
        if (invalidate) { localState = LocalModelFileState.Missing; Selected = false; }
        NotifyLocal();
    }
    internal void Apply(LocalModelFileStatus status)
    {
        localState = status.State; inspecting = false;
        localMessage = status.State switch
        {
            LocalModelFileState.Downloaded => "File đích đã có, đúng dung lượng. Lần quét này không đọc nội dung hoặc xác minh lại hash.",
            LocalModelFileState.Partial => "Có dữ liệu tải dở; file .part có thể đã cấp phát trước và chưa được xem là hoàn tất.",
            LocalModelFileState.SizeMismatch => "File đích khác dung lượng của revision đang mở. Tệp hiện có không được ghi đè tự động.",
            LocalModelFileState.Unavailable => status.Message ?? "Không thể kiểm tra đường dẫn này.",
            _ => "Chưa tìm thấy file hoàn chỉnh trong thư mục đang chọn."
        };
        if (!CanSelect) Selected = false;
        NotifyLocal();
    }
    private void NotifyLocal()
    {
        Raise(nameof(LocalState)); Raise(nameof(IsDownloaded)); Raise(nameof(IsInspecting));
        Raise(nameof(CanSelect)); Raise(nameof(ShowSelection)); Raise(nameof(LocalMessage)); Raise(nameof(LocalStateLabel));
    }
}

public sealed class HubViewModel : ObservableObject
{
    private readonly IHuggingFaceClient client;
    private readonly DownloadsViewModel downloads;
    private readonly Func<AppSettings> settings;
    private readonly ILocalModelFileInventory? inventory;
    private readonly Dictionary<string, string> modelFolders = new(StringComparer.Ordinal);
    private CancellationTokenSource? cancellation, scanCancellation, folderCancellation;
    private int requestVersion, scanVersion;
    private bool busy, scanning, stopped, selecting;
    private string query = "GLM-5.3", modelId = "dealignai/GLM-5.3-ABLITERATED-NVFP4", revision = "main";
    private string filter = "", downloadFilter = "Tất cả", directory = "", error = "", scanError = "";
    private string status = "Tìm theo tên hoặc nhập owner/model để mở trực tiếp.";
    private string? nextPage;
    private HubModelSummary? selectedModel;
    private HubModelDetail? detail;
    private string? pendingFolder;
    private HubFileViewModel? selectionAnchor;
    private int selectedCount;
    private long selectedBytes;

    public HubViewModel(IHuggingFaceClient client, DownloadsViewModel downloads, Func<AppSettings> settings,
        ILocalModelFileInventory? inventory = null)
    {
        this.client = client; this.downloads = downloads; this.settings = settings; this.inventory = inventory;
        FilesView = CollectionViewSource.GetDefaultView(Files);
        FilesView.Filter = item => item is HubFileViewModel file && file.Path.Contains(Filter, StringComparison.OrdinalIgnoreCase)
            && (DownloadFilter == "Tất cả" || file.IsDownloaded == (DownloadFilter == "Đã tải"));
        SearchCommand = new AsyncCommand(_ => SearchAsync(false), Fail, _ => !IsBusy && !stopped);
        MoreCommand = new AsyncCommand(_ => SearchAsync(true), Fail, _ => !IsBusy && !stopped && nextPage is not null);
        OpenCommand = new AsyncCommand(_ => OpenModelAsync(ModelId.Trim(), Revision.Trim()), Fail, _ => !stopped);
        OpenSelectedCommand = new AsyncCommand(_ => SelectedModel is { } model ? OpenModelAsync(model.Id) : Task.CompletedTask,
            Fail, _ => SelectedModel is not null && !stopped);
        CancelCommand = new RelayCommand(_ => CancelRequests(), _ => IsBusy || IsScanning);
        SelectVisibleCommand = new RelayCommand(_ => SelectFiles(FilesView.Cast<HubFileViewModel>(), true), _ => !IsScanning && !IsBusy);
        SelectAllCommand = new RelayCommand(_ => SelectFiles(Files, true), _ => !IsScanning && !IsBusy);
        SelectNoneCommand = new RelayCommand(_ => SelectFiles(Files, false));
        BrowseCommand = new RelayCommand(_ => DownloadDirectory = DesktopActions.ChooseFolder("Chọn thư mục tải model", DownloadDirectory) ?? DownloadDirectory);
        RefreshFilesCommand = new AsyncCommand(_ => RefreshLocalFilesAsync(), Fail, _ => Detail is not null && !IsBusy && !stopped);
        DownloadSelectedCommand = new AsyncCommand(_ => EnqueueAsync(false), Fail, _ => CanDownload && selectedCount > 0);
        DownloadAllCommand = new AsyncCommand(_ => EnqueueAsync(true), Fail, _ => CanDownload && Files.Any(file => file.CanSelect));
        downloads.DownloadCompleted += DownloadFinished;
    }

    public ObservableCollection<HubModelSummary> Models { get; } = [];
    public ObservableCollection<HubFileViewModel> Files { get; } = [];
    public ICollectionView FilesView { get; }
    public IReadOnlyList<string> DownloadFilters { get; } = ["Tất cả", "Đã tải", "Chưa tải"];
    public string Query { get => query; set { if (Set(ref query, value)) { nextPage = null; CommandManager.InvalidateRequerySuggested(); } } }
    public string ModelId { get => modelId; set => Set(ref modelId, value); }
    public string Revision { get => revision; set => Set(ref revision, value); }
    public string Filter { get => filter; set { if (Set(ref filter, value)) { selectionAnchor = null; RefreshView(); } } }
    public string DownloadFilter
    {
        get => downloadFilter;
        set { if (value is not null && DownloadFilters.Contains(value) && Set(ref downloadFilter, value)) { selectionAnchor = null; Raise(nameof(HideDownloaded)); RefreshView(); } }
    }
    public bool HideDownloaded { get => DownloadFilter == "Chưa tải"; set => DownloadFilter = value ? "Chưa tải" : "Tất cả"; }
    public string DownloadDirectory
    {
        get => directory;
        set
        {
            if (!Set(ref directory, value) || stopped) return;
            if (Detail is not null) modelFolders[Detail.Id] = value;
            else pendingFolder = value;
            InvalidateScan(true);
            if (Detail is null) return;
            var pending = new CancellationTokenSource(); folderCancellation = pending;
            _ = ScanFolderAfterDelayAsync(pending);
        }
    }
    public string Error { get => error; private set => Set(ref error, value); }
    public string ScanError { get => scanError; private set => Set(ref scanError, value); }
    public string Status { get => status; private set => Set(ref status, value); }
    public bool IsBusy { get => busy; private set { Set(ref busy, value); CommandManager.InvalidateRequerySuggested(); } }
    public bool IsScanning { get => scanning; private set { Set(ref scanning, value); Raise(nameof(ScanStatus)); CommandManager.InvalidateRequerySuggested(); } }
    public string ScanStatus => IsScanning ? "Đang quét thư mục…" : $"Đã tải {Files.Count(file => file.IsDownloaded)} / {Files.Count} tệp";
    public bool CanDownload => Detail is not null && !IsBusy && !IsScanning && !stopped;
    public HubModelSummary? SelectedModel
    {
        get => selectedModel;
        set
        {
            if (!Set(ref selectedModel, value)) return;
            CommandManager.InvalidateRequerySuggested();
            if (value is not null && !stopped) _ = OpenModelAsync(value.Id);
        }
    }
    public HubModelDetail? Detail { get => detail; private set { Set(ref detail, value); Raise(nameof(DetailSummary)); CommandManager.InvalidateRequerySuggested(); } }
    public string DetailSummary => Detail is null ? "Chưa mở model" : $"{Files.Count} tệp · {DisplayFormat.Bytes(Detail.TotalBytes)} · revision {Detail.Revision}";
    public string SelectionSummary => $"Đã chọn {selectedCount} / {Files.Count} tệp · {DisplayFormat.Bytes(selectedBytes)}";
    public ICommand SearchCommand { get; }
    public ICommand MoreCommand { get; }
    public ICommand OpenCommand { get; }
    public ICommand OpenSelectedCommand { get; }
    public ICommand CancelCommand { get; }
    public ICommand SelectVisibleCommand { get; }
    public ICommand SelectAllCommand { get; }
    public ICommand SelectNoneCommand { get; }
    public ICommand BrowseCommand { get; }
    public ICommand RefreshFilesCommand { get; }
    public ICommand DownloadSelectedCommand { get; }
    public ICommand DownloadAllCommand { get; }

    private void Fail(Exception exception) { Error = exception.Message; Status = "Không thể hoàn tất yêu cầu. Bạn có thể thử lại."; }
    private (int Version, CancellationTokenSource Token) BeginRequest()
    {
        cancellation?.Cancel();
        var token = new CancellationTokenSource(); cancellation = token;
        var version = ++requestVersion;
        IsBusy = true; Error = ""; Status = "Đang đọc Hugging Face…";
        return (version, token);
    }
    private bool Current(int version) => !stopped && version == requestVersion;
    private void EndRequest(int version, CancellationTokenSource token)
    {
        if (version == requestVersion) { cancellation = null; IsBusy = false; }
        token.Dispose();
    }
    private async Task SearchAsync(bool more)
    {
        var search = Query.Trim(); var page = more ? nextPage : null;
        var (version, token) = BeginRequest();
        try
        {
            var result = await client.SearchAsync(search, page, token.Token);
            if (!Current(version) || token.IsCancellationRequested) return;
            if (!more) { SelectedModel = null; Models.Clear(); }
            foreach (var item in result.Models) Models.Add(item);
            nextPage = result.NextPage;
            Status = Models.Count == 0 ? "Không tìm thấy model. Thử tên khác hoặc nhập owner/model." : $"{Models.Count} kết quả · chọn model để xem các tệp";
        }
        catch (OperationCanceledException) { if (Current(version)) Status = "Đã dừng tìm kiếm"; }
        catch (Exception exception) { if (Current(version)) Fail(exception); }
        finally { EndRequest(version, token); }
    }
    public async Task OpenModelAsync(string id, string revision = "main")
    {
        if (stopped) return;
        id = id.Trim(); revision = string.IsNullOrWhiteSpace(revision) ? "main" : revision.Trim();
        var (version, token) = BeginRequest();
        InvalidateScan(true);
        foreach (var row in Files) row.PropertyChanged -= RowChanged;
        Files.Clear(); selectedCount = 0; selectedBytes = 0; selectionAnchor = null; Detail = null;
        ModelId = id; Revision = revision; NotifyFiles();
        try
        {
            var result = await client.GetModelAsync(id, revision, token.Token);
            if (!Current(version) || token.IsCancellationRequested) return;
            Detail = result;
            foreach (var file in result.Files)
            {
                var row = new HubFileViewModel(file); row.PropertyChanged += RowChanged; Files.Add(row);
            }
            var remembered = pendingFolder ?? modelFolders.GetValueOrDefault(result.Id)
                ?? downloads.Items.LastOrDefault(item => item.Request.ModelId == result.Id)?.Request.DestinationDirectory
                ?? Path.Combine(settings().DefaultDownloadDirectory, result.Id.Split('/').Last());
            directory = remembered; Raise(nameof(DownloadDirectory));
            pendingFolder = null;
            modelFolders[result.Id] = remembered;
            NotifyFiles();
            await RefreshLocalFilesAsync();
            if (Current(version)) Status = "Đã ghim revision và quét thư mục. File đã tải được ẩn ô chọn.";
        }
        catch (OperationCanceledException) { if (Current(version)) Status = "Đã dừng mở model"; }
        catch (Exception exception) { if (Current(version)) Fail(exception); }
        finally { EndRequest(version, token); }
    }
    private async Task ScanFolderAfterDelayAsync(CancellationTokenSource pending)
    {
        try
        {
            await Task.Delay(180, pending.Token);
            if (!stopped && !pending.IsCancellationRequested) await RefreshLocalFilesAsync();
        }
        catch (OperationCanceledException) { }
        finally { if (ReferenceEquals(folderCancellation, pending)) folderCancellation = null; pending.Dispose(); }
    }
    private void InvalidateScan(bool invalidateRows)
    {
        folderCancellation?.Cancel(); folderCancellation = null;
        scanCancellation?.Cancel(); scanCancellation = null; scanVersion++;
        if (invalidateRows)
        {
            selecting = true;
            try { foreach (var row in Files) row.Inspect(true); }
            finally { selecting = false; }
        }
        IsScanning = false; ScanError = ""; NotifyFiles();
    }
    public async Task RefreshLocalFilesAsync()
    {
        if (stopped || Detail is null) return;
        folderCancellation?.Cancel(); folderCancellation = null;
        scanCancellation?.Cancel();
        var token = new CancellationTokenSource(); scanCancellation = token;
        var version = ++scanVersion;
        var source = Detail; var folder = DownloadDirectory.Trim(); var rows = Files.ToArray();
        IsScanning = true; ScanError = "";
        foreach (var row in rows) row.Inspect(false);
        try
        {
            // Tests/third-party hosts can omit the optional adapter. Production App always injects it.
            var statuses = inventory is null
                ? rows.Select(row => new LocalModelFileStatus(row.Path, LocalModelFileState.Missing, Message: "Inventory adapter not configured.")).ToArray()
                : await inventory.ScanAsync(folder, rows.Select(row => row.File).ToArray(), token.Token);
            if (!CurrentScan(version, source, folder)) return;
            token.Token.ThrowIfCancellationRequested();
            var byPath = statuses.ToDictionary(item => item.Path, StringComparer.Ordinal);
            selecting = true;
            try
            {
                foreach (var row in rows) row.Apply(byPath.GetValueOrDefault(row.Path)
                    ?? new(row.Path, LocalModelFileState.Unavailable, Message: "Bộ quét không trả về trạng thái tệp."));
            }
            finally { selecting = false; }
        }
        catch (OperationCanceledException)
        {
            if (CurrentScan(version, source, folder))
                foreach (var row in rows) row.Apply(new(row.Path, LocalModelFileState.Unavailable, Message: "Đã dừng quét thư mục; bấm Quét lại để tiếp tục."));
        }
        catch (Exception exception)
        {
            if (CurrentScan(version, source, folder))
            {
                ScanError = exception.Message;
                foreach (var row in rows) row.Apply(new(row.Path, LocalModelFileState.Unavailable, Message: exception.Message));
            }
        }
        finally
        {
            if (version == scanVersion) { scanCancellation = null; IsScanning = false; NotifyFiles(); RefreshView(); }
            token.Dispose();
        }
    }
    private bool CurrentScan(int version, HubModelDetail source, string folder) => !stopped && version == scanVersion
        && ReferenceEquals(Detail, source) && folder.Equals(DownloadDirectory.Trim(), StringComparison.Ordinal);
    private void RowChanged(object? sender, PropertyChangedEventArgs args)
    {
        if (args.PropertyName != nameof(HubFileViewModel.Selected) || sender is not HubFileViewModel row) return;
        selectedCount += row.Selected ? 1 : -1; selectedBytes += row.Selected ? row.File.Size : -row.File.Size;
        if (!selecting) { Raise(nameof(SelectionSummary)); CommandManager.InvalidateRequerySuggested(); }
    }
    private void NotifyFiles()
    {
        Raise(nameof(SelectionSummary)); Raise(nameof(DetailSummary)); Raise(nameof(ScanStatus));
        CommandManager.InvalidateRequerySuggested();
    }
    private void RefreshView()
    {
        if (FilesView is IEditableCollectionView editable && editable.IsEditingItem) editable.CancelEdit();
        FilesView.Refresh();
    }
    private void SelectFiles(IEnumerable<HubFileViewModel> rows, bool selected)
    {
        selectionAnchor = null; SelectBatch(rows.ToArray(), selected);
    }
    private void SelectBatch(IEnumerable<HubFileViewModel> rows, bool selected)
    {
        selecting = true;
        try { foreach (var row in rows) if (!selected || row.CanSelect) row.Selected = selected; }
        finally { selecting = false; Raise(nameof(SelectionSummary)); CommandManager.InvalidateRequerySuggested(); }
    }
    public void SelectRange(HubFileViewModel target, bool selected, bool extend)
    {
        if (stopped || IsScanning || IsBusy) return;
        var visible = FilesView.Cast<HubFileViewModel>().ToArray();
        var end = Array.IndexOf(visible, target);
        if (end < 0) return;
        var start = extend && selectionAnchor is not null ? Array.IndexOf(visible, selectionAnchor) : -1;
        if (start < 0)
        {
            if (!target.CanSelect) return;
            selectionAnchor = target; SelectBatch([target], selected); return;
        }
        SelectBatch(visible.Skip(Math.Min(start, end)).Take(Math.Abs(start - end) + 1), selected);
    }
    private async Task EnqueueAsync(bool all)
    {
        if (!CanDownload || Detail is null) return;
        var source = Detail; var folder = DownloadDirectory.Trim();
        await RefreshLocalFilesAsync();
        if (!ReferenceEquals(source, Detail) || !SameDirectory(folder, DownloadDirectory) || stopped || ScanError.Length > 0) return;
        var requests = Files.Where(row => row.CanSelect && (all || row.Selected))
            .Select(row => new DownloadRequest(source.Id, source.Revision, row.File, Path.GetFullPath(folder))).ToArray();
        await downloads.EnqueueAsync(requests); Status = $"Đã thêm {requests.Length} tệp vào Tải xuống.";
    }
    private void DownloadFinished(DownloadRequest request) => _ = RefreshForDownloadAsync(request);
    public Task RefreshForDownloadAsync(DownloadRequest request)
    {
        if (stopped || Detail is null || request.ModelId != Detail.Id
            || !request.Revision.Equals(Detail.Revision, StringComparison.OrdinalIgnoreCase)
            || !SameDirectory(request.DestinationDirectory, DownloadDirectory)) return Task.CompletedTask;
        return RefreshLocalFilesAsync();
    }
    private static bool SameDirectory(string left, string right)
    {
        try { return Path.TrimEndingDirectorySeparator(Path.GetFullPath(left.Trim())).Equals(Path.TrimEndingDirectorySeparator(Path.GetFullPath(right.Trim())), StringComparison.OrdinalIgnoreCase); }
        catch (Exception exception) when (exception is ArgumentException or NotSupportedException or PathTooLongException) { return false; }
    }
    private void CancelRequests()
    {
        cancellation?.Cancel(); scanCancellation?.Cancel(); folderCancellation?.Cancel();
    }
    public Task StopAsync()
    {
        if (stopped) return Task.CompletedTask;
        stopped = true; requestVersion++; scanVersion++; CancelRequests();
        cancellation = null; scanCancellation = null; folderCancellation = null;
        IsBusy = false; IsScanning = false; downloads.DownloadCompleted -= DownloadFinished;
        foreach (var row in Files) row.PropertyChanged -= RowChanged;
        return Task.CompletedTask;
    }
}
