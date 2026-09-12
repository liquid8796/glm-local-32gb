using System.Net.Http;
using System.Windows;
using ModelDesk.Desktop.Presentation;
using ModelDesk.Desktop.ViewModels;
using ModelDesk.Infrastructure.Downloads;
using ModelDesk.Infrastructure.Hub;
using ModelDesk.Infrastructure.Reports;
using ModelDesk.Infrastructure.Runtime;
using ModelDesk.Infrastructure.Settings;

namespace ModelDesk.Desktop;

public partial class App : Application
{
    private HttpClient? http;
    protected override async void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        ShutdownMode = ShutdownMode.OnExplicitShutdown;
        var state = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "ModelDesk");
        try
        {
            DiagnosticLog.Initialize(state);
            var credentials = new WindowsCredentialStore(state);
            http = HubHttpClientFactory.Create();
            var model = new ShellViewModel(new JsonSettingsStore(state), credentials, new PythonCoreService(state), new ReportService(),
                new HuggingFaceClient(http, credentials), new ModelDownloader(http, credentials), new JsonDownloadQueueStore(state), JsonSettingsStore.CreateDefaults);
            model.ThemeChanged += ThemeManager.Apply;
            ThemeManager.Apply("Dark");
            DispatcherUnhandledException += (_, args) => { DiagnosticLog.Write(args.Exception.ToString()); model.ShowError(args.Exception); args.Handled = true; };
            var window = new MainWindow(model); MainWindow = window; window.Show();
            ShutdownMode = ShutdownMode.OnMainWindowClose;
            DiagnosticLog.Write("Main window shown; initializing workspace");
            await model.InitializeAsync();
            DiagnosticLog.Write($"Workspace initialization finished: {model.Profiles.Count} profiles; {model.PythonStatus}");
        }
        catch (Exception exception)
        {
            DiagnosticLog.Write(exception.ToString());
            MessageBox.Show("Không thể mở ModelDesk.\n" + exception.Message + "\n\nChi tiết: " + Path.Combine(state, "startup.log"), "ModelDesk", MessageBoxButton.OK, MessageBoxImage.Error);
            Shutdown(1);
        }
    }
    protected override void OnExit(ExitEventArgs e) { http?.Dispose(); base.OnExit(e); }
}
