using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Windows.Data;
using System.Windows.Input;
using ModelDesk.Core;
using ModelDesk.Desktop.Presentation;

namespace ModelDesk.Desktop.ViewModels;

public sealed class HubFileViewModel(HubFile file) : ObservableObject
{
    private bool selected;
    public HubFile File { get; } = file;
    public string Path => File.Path;
    public string Size => DisplayFormat.Bytes(File.Size);
    public string Hash => File.LfsSha256 ?? File.BlobId ?? "Không có hash";
    public bool Selected { get => selected; set => Set(ref selected, value); }
}

public sealed class HubViewModel : ObservableObject
{
    private readonly IHuggingFaceClient client;
    private readonly DownloadsViewModel downloads;
    private readonly Func<AppSettings> settings;
    private CancellationTokenSource? cancellation;
    private string query = "GLM-5.3", modelId = "dealignai/GLM-5.3-ABLITERATED-NVFP4", revision = "main", filter = "", directory = "", error = "", status = "Tìm theo tên hoặc nhập owner/model để mở trực tiếp.";
    private string? nextPage;
    private bool busy;
    private HubModelSummary? selectedModel;
    private HubModelDetail? detail;
    private bool selecting;
    private int selectedCount;
    private long selectedBytes;
    public HubViewModel(IHuggingFaceClient client, DownloadsViewModel downloads, Func<AppSettings> settings)
    {
        this.client = client; this.downloads = downloads; this.settings = settings;
        FilesView = CollectionViewSource.GetDefaultView(Files);
        FilesView.Filter = item => item is HubFileViewModel file && file.Path.Contains(Filter, StringComparison.OrdinalIgnoreCase);
        SearchCommand = new AsyncCommand(_ => SearchAsync(false), Fail, _ => !IsBusy);
        MoreCommand = new AsyncCommand(_ => SearchAsync(true), Fail, _ => !IsBusy && nextPage is not null);
        OpenCommand = new AsyncCommand(_ => OpenAsync(), Fail, _ => !IsBusy);
        OpenSelectedCommand = new AsyncCommand(async _ => { if (SelectedModel is not null) { ModelId = SelectedModel.Id; Revision = "main"; await OpenAsync(); } }, Fail, _ => !IsBusy && SelectedModel is not null);
        CancelCommand = new RelayCommand(_ => cancellation?.Cancel(), _ => IsBusy);
        SelectVisibleCommand = new RelayCommand(_ => SelectFiles(FilesView.Cast<HubFileViewModel>(), true));
        SelectAllCommand = new RelayCommand(_ => SelectFiles(Files, true));
        SelectNoneCommand = new RelayCommand(_ => SelectFiles(Files, false));
        BrowseCommand = new RelayCommand(_ => DownloadDirectory = DesktopActions.ChooseFolder("Chọn thư mục tải model", DownloadDirectory) ?? DownloadDirectory);
        DownloadSelectedCommand = new AsyncCommand(_ => EnqueueAsync(false), Fail, _ => Detail is not null && selectedCount > 0);
        DownloadAllCommand = new AsyncCommand(_ => EnqueueAsync(true), Fail, _ => Detail is not null && Files.Count > 0);
    }
    public ObservableCollection<HubModelSummary> Models { get; } = [];
    public ObservableCollection<HubFileViewModel> Files { get; } = [];
    public ICollectionView FilesView { get; }
    public string Query { get => query; set { if (Set(ref query, value)) { nextPage = null; CommandManager.InvalidateRequerySuggested(); } } }
    public string ModelId { get => modelId; set => Set(ref modelId, value); }
    public string Revision { get => revision; set => Set(ref revision, value); }
    public string Filter { get => filter; set { Set(ref filter, value); FilesView.Refresh(); } }
    public string DownloadDirectory { get => directory; set => Set(ref directory, value); }
    public string Error { get => error; private set => Set(ref error, value); }
    public string Status { get => status; private set => Set(ref status, value); }
    public bool IsBusy { get => busy; private set { Set(ref busy, value); CommandManager.InvalidateRequerySuggested(); } }
    public HubModelSummary? SelectedModel { get => selectedModel; set { Set(ref selectedModel, value); CommandManager.InvalidateRequerySuggested(); } }
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
    public ICommand DownloadSelectedCommand { get; }
    public ICommand DownloadAllCommand { get; }
    private void Fail(Exception exception) { Error = exception.Message; Status = "Không thể hoàn tất yêu cầu. Bạn có thể thử lại."; }
    private async Task SearchAsync(bool more)
    {
        using var token = Begin();
        try
        {
            var result = await client.SearchAsync(Query.Trim(), more ? nextPage : null, token.Token);
            if (!more) Models.Clear(); foreach (var item in result.Models) Models.Add(item); nextPage = result.NextPage;
            Status = Models.Count == 0 ? "Không tìm thấy model. Thử tên khác hoặc nhập owner/model." : $"{Models.Count} kết quả · chọn model để xem các tệp";
        }
        catch (OperationCanceledException) { Status = "Đã dừng tìm kiếm"; }
        finally { End(); }
    }
    private async Task OpenAsync()
    {
        using var token = Begin();
        try
        {
            Detail = await client.GetModelAsync(ModelId.Trim(), string.IsNullOrWhiteSpace(Revision) ? "main" : Revision.Trim(), token.Token);
            Files.Clear(); selectedCount = 0; selectedBytes = 0; foreach (var file in Detail.Files)
            {
                var item = new HubFileViewModel(file);
                item.PropertyChanged += (_, args) =>
                {
                    if (args.PropertyName != nameof(HubFileViewModel.Selected)) return;
                    selectedCount += item.Selected ? 1 : -1; selectedBytes += item.Selected ? item.File.Size : -item.File.Size;
                    if (!selecting) { Raise(nameof(SelectionSummary)); CommandManager.InvalidateRequerySuggested(); }
                };
                Files.Add(item);
            }
            DownloadDirectory = Path.Combine(settings().DefaultDownloadDirectory, Detail.Id.Split('/').Last());
            Status = "Đã ghim revision. Chọn tệp và thư mục trước khi tải."; Raise(nameof(DetailSummary)); Raise(nameof(SelectionSummary));
        }
        catch (OperationCanceledException) { Status = "Đã dừng mở model"; }
        finally { End(); }
    }
    private CancellationTokenSource Begin() { IsBusy = true; Error = ""; Status = "Đang đọc Hugging Face…"; cancellation = new(); return cancellation; }
    private void SelectFiles(IEnumerable<HubFileViewModel> files, bool selected)
    {
        selecting = true;
        try { foreach (var file in files) file.Selected = selected; }
        finally { selecting = false; Raise(nameof(SelectionSummary)); CommandManager.InvalidateRequerySuggested(); }
    }
    private void End() { cancellation = null; IsBusy = false; }
    private async Task EnqueueAsync(bool all)
    {
        if (Detail is null) return;
        if (string.IsNullOrWhiteSpace(DownloadDirectory)) throw new ArgumentException("Chọn thư mục tải xuống.");
        var files = Files.Where(item => all || item.Selected).Select(item => new DownloadRequest(Detail.Id, Detail.Revision, item.File, Path.GetFullPath(DownloadDirectory))).ToArray();
        await downloads.EnqueueAsync(files); Status = $"Đã thêm {files.Length} tệp vào Tải xuống.";
    }
}
