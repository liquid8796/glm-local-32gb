using System.Text;
using System.Windows.Input;
using System.Windows.Threading;
using ModelDesk.Core;
using ModelDesk.Desktop.Presentation;

namespace ModelDesk.Desktop.ViewModels;

public sealed class CoreTaskViewModel(IPythonCoreService service, Func<AppSettings> settings) : ObservableObject
{
    private CancellationTokenSource? cancellation;
    private TaskCompletionSource? finished;
    private readonly StringBuilder buffer = new();
    private bool busy;
    private bool available = true;
    private string status = "Sẵn sàng", log = "Chọn một tác vụ để xem nhật ký trực tiếp.", error = "";
    private bool expanded;
    public bool IsBusy { get => busy; private set { Set(ref busy, value); CommandManager.InvalidateRequerySuggested(); } }
    public bool IsAvailable { get => available; set { Set(ref available, value); CommandManager.InvalidateRequerySuggested(); } }
    public bool IsExpanded { get => expanded; set => Set(ref expanded, value); }
    public string Status { get => status; private set => Set(ref status, value); }
    public string Log { get => log; private set => Set(ref log, value); }
    public string Error { get => error; private set => Set(ref error, value); }
    public CoreRunResult? LastResult { get; private set; }
    public event EventHandler? Completed;
    public ICommand CancelCommand => new RelayCommand(_ => Cancel(), _ => IsBusy);
    public ICommand OpenLogCommand => new RelayCommand(_ => { if (LastResult is not null) DesktopActions.OpenFolder(LastResult.LogPath); }, _ => LastResult is not null);
    public void Cancel() => cancellation?.Cancel();
    public async Task StopAsync() { Cancel(); if (finished is not null) await finished.Task; }
    public void ShowError(Exception exception) { Error = exception.Message; IsExpanded = true; }
    public async Task RunAsync(string operation, IReadOnlyList<string>? arguments = null)
    {
        if (IsBusy) throw new InvalidOperationException("Hãy đợi hoặc dừng tác vụ Python đang chạy.");
        var operationName = CoreOperations.Get(operation).Name;
        finished = new(TaskCreationOptions.RunContinuationsAsynchronously);
        cancellation = new(); IsBusy = true; IsExpanded = true; Error = "";
        buffer.Clear(); Log = ""; Status = operationName + " · đang chạy";
        bool dirty = false;
        var timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(150) };
        timer.Tick += (_, _) => { if (dirty) { Log = buffer.ToString(); dirty = false; } };
        timer.Start();
        try
        {
            LastResult = await service.RunAsync(new(settings(), operation, arguments ?? []),
                new Progress<CoreOutput>(item =>
                {
                    buffer.AppendLine(item.Text);
                    if (buffer.Length > 350_000) buffer.Remove(0, buffer.Length - 300_000);
                    dirty = true;
                }), cancellation.Token);
            Status = LastResult.Cancelled ? "Đã dừng tác vụ"
                : LastResult.TimedOut ? "Hết thời gian chờ · xem giai đoạn cuối trong nhật ký"
                : LastResult.ExitCode == 0 ? "Hoàn tất" : $"Cần xem báo cáo · mã {LastResult.ExitCode}";
            Raise(nameof(LastResult)); Completed?.Invoke(this, EventArgs.Empty);
        }
        catch (OperationCanceledException) { Status = "Đã dừng tác vụ"; }
        catch (Exception exception) { Status = "Không thể hoàn tất"; ShowError(exception); }
        finally { timer.Stop(); Log = buffer.ToString(); IsBusy = false; cancellation.Dispose(); cancellation = null; finished.TrySetResult(); }
    }
}
