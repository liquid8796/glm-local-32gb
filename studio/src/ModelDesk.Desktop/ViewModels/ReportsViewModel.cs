using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Windows.Data;
using System.Windows.Input;
using ModelDesk.Core;
using ModelDesk.Desktop.Presentation;

namespace ModelDesk.Desktop.ViewModels;

public sealed class ReportsViewModel : ObservableObject
{
    private readonly IReportService service;
    private readonly Func<AppSettings> settings;
    private readonly Func<ModelProfile?> profile;
    private string filter = "", content = "Chọn một báo cáo để đọc nội dung JSON hoặc Markdown.", error = "";
    private ReportEntry? selected;
    private int generation;
    public ReportsViewModel(IReportService service, Func<AppSettings> settings, Func<ModelProfile?> profile)
    {
        this.service = service; this.settings = settings; this.profile = profile;
        View = CollectionViewSource.GetDefaultView(Items); View.Filter = item => item is ReportEntry report && report.Name.Contains(Filter, StringComparison.OrdinalIgnoreCase);
        RefreshCommand = new AsyncCommand(_ => RefreshAsync(), Fail);
        ReadCommand = new AsyncCommand(async _ => { if (Selected is not null) Content = await service.ReadAsync(Selected.Path); }, Fail, _ => Selected is not null);
        OpenFolderCommand = new RelayCommand(_ => { try { DesktopActions.OpenFolder(Selected?.Path ?? profile()?.ReportDirectory ?? Path.Combine(settings().ProjectRoot, "reports")); } catch (Exception exception) { Fail(exception); } });
    }
    public ObservableCollection<ReportEntry> Items { get; } = [];
    public ICollectionView View { get; }
    public string Filter { get => filter; set { Set(ref filter, value); View.Refresh(); } }
    public string Content { get => content; private set => Set(ref content, value); }
    public string Error { get => error; private set => Set(ref error, value); }
    public ReportEntry? Selected { get => selected; set { Set(ref selected, value); CommandManager.InvalidateRequerySuggested(); } }
    public string Summary => Items.Count == 0 ? "Chưa có báo cáo cho profile này." : $"{Items.Count} báo cáo trong profile đang chọn";
    public ICommand RefreshCommand { get; }
    public ICommand ReadCommand { get; }
    public ICommand OpenFolderCommand { get; }
    public void Fail(Exception exception) => Error = exception.Message;
    public async Task RefreshAsync()
    {
        Error = "";
        var currentGeneration = ++generation;
        var root = settings().ProjectRoot; var folder = profile()?.ReportDirectory;
        var result = await Task.Run(() => service.List(root, folder));
        if (currentGeneration != generation) return;
        Items.Clear(); foreach (var report in result) Items.Add(report); Selected = Items.FirstOrDefault(); Raise(nameof(Summary));
    }
}
