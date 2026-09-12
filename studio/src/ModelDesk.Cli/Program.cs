using ModelDesk.Cli;
using ModelDesk.Infrastructure.Downloads;
using ModelDesk.Infrastructure.Hub;
using ModelDesk.Infrastructure.Reports;
using ModelDesk.Infrastructure.Runtime;
using ModelDesk.Infrastructure.Settings;

Console.OutputEncoding = System.Text.Encoding.UTF8;
using var cancellation = new CancellationTokenSource();
Console.CancelKeyPress += (_, eventArgs) => { eventArgs.Cancel = true; cancellation.Cancel(); };
using var http = HubHttpClientFactory.Create();
var credentials = new WindowsCredentialStore();
var application = new CliApplication(new JsonSettingsStore(), new HuggingFaceClient(http, credentials),
    new ModelDownloader(http, credentials), new PythonCoreService(), new ReportService());
return await application.RunAsync(args, cancellation.Token);
