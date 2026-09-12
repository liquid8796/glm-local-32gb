using System.Collections.ObjectModel;
using System.Windows.Input;
using ModelDesk.Core;
using ModelDesk.Desktop.Presentation;

namespace ModelDesk.Desktop.ViewModels;

public sealed class ShellViewModel : ObservableObject
{
    private readonly ISettingsStore store;
    private readonly IPythonCoreService core;
    private readonly Func<AppSettings>? discoverDefaults;
    private AppSettings current = new();
    private ModelProfile? profile;
    private int tab;
    private bool initializing;
    private string error = "", readiness = "Chưa có báo cáo sẵn sàng", readinessDetail = "Chạy kiểm tra để cập nhật thông tin phần cứng và trọng số cục bộ.";
    public ShellViewModel(ISettingsStore store, ICredentialStore credentials, IPythonCoreService core, IReportService reports, IHuggingFaceClient hub, IModelDownloader downloader, IDownloadQueueStore queue, Func<AppSettings>? discover = null, ILocalModelFileInventory? inventory = null)
    {
        this.store = store; this.core = core; discoverDefaults = discover;
        Task = new(core, () => current); Run = new(Task); Validation = new(Task);
        Task.IsAvailable = false;
        Downloads = new(downloader, queue, () => current); Hub = new(hub, Downloads, () => current, inventory);
        Reports = new(reports, () => current, () => SelectedProfile);
        Settings = new(store, credentials, () => current, ApplyAsync, discover);
        DoctorCommand = new AsyncCommand(_ => Task.RunAsync("doctor"), ShowError, _ => Task.IsAvailable && !Task.IsBusy);
        MonitorCommand = new AsyncCommand(_ => Task.RunAsync("monitor", ["--seconds", "10"]), ShowError, _ => Task.IsAvailable && !Task.IsBusy);
        GoRunCommand = new RelayCommand(_ => SelectedTab = 1); GoHubCommand = new RelayCommand(_ => SelectedTab = 3); GoValidationCommand = new RelayCommand(_ => SelectedTab = 2);
        Task.Completed += async (_, _) => { try { await Reports.RefreshAsync(); RefreshReadiness(); } catch (Exception exception) { ShowError(exception); } };
    }
    public ObservableCollection<ModelProfile> Profiles { get; } = [];
    public CoreTaskViewModel Task { get; }
    public RunViewModel Run { get; }
    public ValidationViewModel Validation { get; }
    public DownloadsViewModel Downloads { get; }
    public HubViewModel Hub { get; }
    public ReportsViewModel Reports { get; }
    public SettingsViewModel Settings { get; }
    public ModelProfile? SelectedProfile
    {
        get => profile;
        set
        {
            if (!Set(ref profile, value) || value is null) return;
            current = current with { Profile = value.Key };
            Run.ModelDirectory = string.IsNullOrWhiteSpace(current.RuntimeModelDirectory) ? Resolve(value.ModelDirectory) : current.RuntimeModelDirectory;
            Validation.ModelDirectory = Run.ModelDirectory;
            Raise(nameof(ModelId)); Raise(nameof(Revision)); Raise(nameof(ProfileDirectory));
            _ = RefreshProfileAsync();
        }
    }
    public int SelectedTab { get => tab; set => Set(ref tab, value); }
    public bool IsInitializing { get => initializing; private set { Set(ref initializing, value); Raise(nameof(PythonStatus)); } }
    public string Error { get => error; private set => Set(ref error, value); }
    public string ModelId => SelectedProfile?.ModelId ?? "Chưa chọn profile";
    public string Revision => SelectedProfile?.Revision ?? "—";
    public string ProfileDirectory => SelectedProfile is null ? "—" : Resolve(SelectedProfile.ModelDirectory);
    public string PythonStatus => IsInitializing ? "Đang nạp workspace…" : File.Exists(current.PythonExecutable) ? "Python · đã tìm thấy" : "Python · kiểm tra đường dẫn";
    public string PythonPath => current.PythonExecutable;
    public string ResourceSummary => $"{current.RamBudgetBytes / 1_000_000_000d:0.##} GB RAM · CPU ≤ {current.CpuLimitPercent}% · GPU mục tiêu {current.GpuTargetPercent:0.#}%";
    public string Readiness { get => readiness; private set => Set(ref readiness, value); }
    public string ReadinessDetail { get => readinessDetail; private set => Set(ref readinessDetail, value); }
    public ICommand DoctorCommand { get; }
    public ICommand MonitorCommand { get; }
    public ICommand GoRunCommand { get; }
    public ICommand GoHubCommand { get; }
    public ICommand GoValidationCommand { get; }
    public event Action<string>? ThemeChanged;
    public async Task InitializeAsync()
    {
        IsInitializing = true;
        try
        {
            AppSettings? loaded = null;
            try { loaded = await store.LoadAsync(); }
            catch (Exception exception)
            {
                ShowError(exception);
                try { loaded = discoverDefaults?.Invoke(); } catch (Exception discoveryError) { ShowError(discoveryError); }
            }
            try { if (loaded is not null) await ApplyAsync(loaded); }
            catch (Exception exception) { ShowError(exception); Settings.Load(current); }
            // Hub/queue recovery remains available even if the Python workspace needs repair.
            try { await Downloads.InitializeAsync(); } catch (Exception exception) { ShowError(exception); }
            try { await Settings.InspectTokenAsync(); } catch (Exception exception) { ShowError(exception); }
        }
        finally { IsInitializing = false; }
    }
    public void ShowError(Exception exception) { Error = exception.Message; DiagnosticLog.Write(exception.ToString()); }
    public async Task CloseAsync() => await System.Threading.Tasks.Task.WhenAll(Hub.StopAsync(), Task.StopAsync(), Downloads.StopAsync());
    private string Resolve(string path) => Path.IsPathRooted(path) ? path : Path.Combine(current.ProjectRoot, path);
    private async Task ApplyAsync(AppSettings settings)
    {
        current = settings; Settings.Load(settings); Profiles.Clear(); Task.IsAvailable = false;
        Raise(nameof(PythonStatus)); Raise(nameof(PythonPath)); Raise(nameof(ResourceSummary));
        ThemeChanged?.Invoke(settings.Theme); await Downloads.ApplySettingsAsync();
        var profiles = await System.Threading.Tasks.Task.Run(() => core.GetProfiles(settings.ProjectRoot));
        foreach (var item in profiles) Profiles.Add(item);
        SelectedProfile = Profiles.FirstOrDefault(item => item.Key == current.Profile) ?? Profiles.FirstOrDefault();
        Task.IsAvailable = SelectedProfile is not null;
        if (SelectedProfile is not null)
        {
            Run.ModelDirectory = string.IsNullOrWhiteSpace(current.RuntimeModelDirectory) ? Resolve(SelectedProfile.ModelDirectory) : current.RuntimeModelDirectory;
            Validation.ModelDirectory = Run.ModelDirectory;
        }
        await Reports.RefreshAsync(); RefreshReadiness();
    }
    private async Task RefreshProfileAsync()
    {
        try { await Reports.RefreshAsync(); RefreshReadiness(); } catch (Exception exception) { ShowError(exception); }
    }
    private void RefreshReadiness()
    {
        var latest = Reports.Items.FirstOrDefault(item => Path.GetFileName(item.Path).Equals("latest.json", StringComparison.OrdinalIgnoreCase));
        Readiness = latest is null ? "Chưa kiểm tra sẵn sàng" : latest.Status ?? "Đã có báo cáo";
        ReadinessDetail = latest is null ? "Kiểm tra phần cứng và dữ liệu trước khi chạy model." : $"Báo cáo {latest.ModifiedAt.LocalDateTime:dd/MM/yyyy HH:mm} · mở Báo cáo để đọc chi tiết.";
    }
}
