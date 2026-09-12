using System.Globalization;
using System.Windows.Input;
using ModelDesk.Core;
using ModelDesk.Desktop.Presentation;

namespace ModelDesk.Desktop.ViewModels;

public sealed class SettingsViewModel : ObservableObject
{
    private readonly ISettingsStore store;
    private readonly ICredentialStore credentials;
    private readonly Func<AppSettings> current;
    private readonly Func<AppSettings, Task> apply;
    private string root = "", python = "", downloads = "", runtime = "", theme = "Dark", speed = "0", concurrent = "2", connections = "4", ram = "32", cpu = "70", gpu = "0", target = "60", window = "10", message = "", tokenStatus = "Chưa kiểm tra token";
    public SettingsViewModel(ISettingsStore store, ICredentialStore credentials, Func<AppSettings> current, Func<AppSettings, Task> apply, Func<AppSettings>? discover = null)
    {
        this.store = store; this.credentials = credentials; this.current = current; this.apply = apply;
        SaveCommand = new AsyncCommand(_ => SaveAsync(), exception => Message = exception.Message);
        DiscoverCommand = new RelayCommand(_ =>
        {
            try
            {
                if (discover is null) return;
                var found = discover(); var oldRoot = ProjectRoot;
                var automaticPython = string.IsNullOrWhiteSpace(PythonExecutable) || PythonExecutable is "python" or "py"
                    || string.Equals(PythonExecutable, Path.Combine(oldRoot, ".venv-reference", "Scripts", "python.exe"), StringComparison.OrdinalIgnoreCase);
                if (!File.Exists(Path.Combine(ProjectRoot, "glm_local", "__main__.py"))) ProjectRoot = found.ProjectRoot;
                if (automaticPython)
                {
                    var candidate = Path.Combine(ProjectRoot, ".venv-reference", "Scripts", "python.exe");
                    PythonExecutable = File.Exists(candidate) ? candidate : "python";
                }
                if (string.IsNullOrWhiteSpace(DownloadDirectory)) DownloadDirectory = found.DefaultDownloadDirectory;
                Message = automaticPython ? "Đã tìm đường dẫn. Lưu cấu hình để nạp lại profile." : "Đã tìm core và giữ nguyên Python tùy chỉnh. Lưu cấu hình để nạp lại profile.";
            }
            catch (Exception exception) { Message = exception.Message; }
        }, _ => discover is not null);
        BrowseRootCommand = new RelayCommand(_ => ProjectRoot = DesktopActions.ChooseFolder("Chọn thư mục Python core", ProjectRoot) ?? ProjectRoot);
        BrowsePythonCommand = new RelayCommand(_ => PythonExecutable = DesktopActions.ChooseFile("Chọn Python", "Python executable|python.exe;python3.exe;pythonw.exe|Executable|*.exe") ?? PythonExecutable);
        BrowseDownloadsCommand = new RelayCommand(_ => DownloadDirectory = DesktopActions.ChooseFolder("Chọn thư mục tải mặc định", DownloadDirectory) ?? DownloadDirectory);
        BrowseRuntimeCommand = new RelayCommand(_ => RuntimeDirectory = DesktopActions.ChooseFolder("Chọn thư mục trọng số chạy model", RuntimeDirectory) ?? RuntimeDirectory);
        ClearTokenCommand = new AsyncCommand(async _ => { await credentials.SetTokenAsync(null); TokenStatus = "Đã xóa token lưu trên máy"; }, exception => Message = exception.Message);
    }
    public string ProjectRoot { get => root; set => Set(ref root, value); }
    public string PythonExecutable { get => python; set => Set(ref python, value); }
    public string DownloadDirectory { get => downloads; set => Set(ref downloads, value); }
    public string RuntimeDirectory { get => runtime; set => Set(ref runtime, value); }
    public string Theme { get => theme; set => Set(ref theme, value); }
    public string Speed { get => speed; set => Set(ref speed, value); }
    public string Concurrent { get => concurrent; set => Set(ref concurrent, value); }
    public string ConnectionsPerFile { get => connections; set => Set(ref connections, value); }
    public string Ram { get => ram; set => Set(ref ram, value); }
    public string Cpu { get => cpu; set => Set(ref cpu, value); }
    public string GpuIndex { get => gpu; set => Set(ref gpu, value); }
    public string GpuTarget { get => target; set => Set(ref target, value); }
    public string GpuWindow { get => window; set => Set(ref window, value); }
    public string Message { get => message; private set => Set(ref message, value); }
    public string TokenStatus { get => tokenStatus; private set => Set(ref tokenStatus, value); }
    public string SettingsPath => store.FilePath;
    public string[] Themes { get; } = ["Dark", "Light", "System"];
    public ICommand SaveCommand { get; }
    public ICommand DiscoverCommand { get; }
    public ICommand BrowseRootCommand { get; }
    public ICommand BrowsePythonCommand { get; }
    public ICommand BrowseDownloadsCommand { get; }
    public ICommand BrowseRuntimeCommand { get; }
    public ICommand ClearTokenCommand { get; }
    public void Load(AppSettings value)
    {
        ProjectRoot = value.ProjectRoot; PythonExecutable = value.PythonExecutable; DownloadDirectory = value.DefaultDownloadDirectory; RuntimeDirectory = value.RuntimeModelDirectory; Theme = value.Theme;
        Speed = (value.DownloadBytesPerSecond / 1048576d).ToString("0.###", CultureInfo.InvariantCulture); Concurrent = value.ConcurrentDownloads.ToString(CultureInfo.InvariantCulture);
        ConnectionsPerFile = value.DownloadConnectionsPerFile.ToString(CultureInfo.InvariantCulture);
        Ram = (value.RamBudgetBytes / 1_000_000_000d).ToString("0.###", CultureInfo.InvariantCulture); Cpu = value.CpuLimitPercent.ToString(CultureInfo.InvariantCulture);
        GpuIndex = value.GpuIndex.ToString(CultureInfo.InvariantCulture); GpuTarget = value.GpuTargetPercent.ToString(CultureInfo.InvariantCulture); GpuWindow = value.GpuWindowSeconds.ToString(CultureInfo.InvariantCulture);
    }
    public async Task InspectTokenAsync() => TokenStatus = string.IsNullOrEmpty(await credentials.GetTokenAsync()) ? "Chưa lưu token · model công khai không cần token" : "Đã lưu token mã hóa cho tài khoản Windows hiện tại";
    public async Task SaveTokenAsync(string token)
    {
        try { await credentials.SetTokenAsync(token.Trim()); await InspectTokenAsync(); Message = "Đã cập nhật token."; }
        catch (Exception exception) { Message = exception.Message; }
    }
    private async Task SaveAsync()
    {
        var value = current() with
        {
            ProjectRoot = ProjectRoot.Trim(), PythonExecutable = PythonExecutable.Trim(), DefaultDownloadDirectory = DownloadDirectory.Trim(), RuntimeModelDirectory = RuntimeDirectory.Trim(), Theme = Theme,
            DownloadBytesPerSecond = checked((long)(Number(Speed, "Tốc độ tải") * 1048576)), ConcurrentDownloads = Integer(Concurrent, "Số tải đồng thời"),
            DownloadConnectionsPerFile = Integer(ConnectionsPerFile, "Kết nối trên mỗi tệp"),
            RamBudgetBytes = checked((long)(Number(Ram, "RAM") * 1_000_000_000)), CpuLimitPercent = Integer(Cpu, "CPU"), GpuIndex = Integer(GpuIndex, "GPU index"), GpuTargetPercent = Number(GpuTarget, "GPU target"), GpuWindowSeconds = Integer(GpuWindow, "GPU window")
        };
        value.Validate();
        if (string.IsNullOrWhiteSpace(value.ProjectRoot) || !Directory.Exists(value.ProjectRoot)) throw new ArgumentException("Thư mục Python core chưa tồn tại.");
        if (!File.Exists(Path.Combine(value.ProjectRoot, "glm_local", "__main__.py"))) throw new ArgumentException("Thư mục đã chọn không có glm_local/__main__.py. Hãy chọn đúng Python core trước khi lưu.");
        if (string.IsNullOrWhiteSpace(value.DefaultDownloadDirectory)) throw new ArgumentException("Nhập thư mục tải mặc định.");
        await store.SaveAsync(value); await apply(value); Message = "Đã lưu. Giới hạn tải được áp dụng ngay; tác vụ Python tiếp theo dùng cấu hình mới.";
    }
    private static double Number(string value, string label) => double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var number) && double.IsFinite(number) && number >= 0 ? number : throw new ArgumentException(label + " phải là số không âm (dùng dấu chấm thập phân).");
    private static int Integer(string value, string label) => int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var number) ? number : throw new ArgumentException(label + " phải là số nguyên.");
}
