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
    private long runVersion;
    public bool IsBusy { get => busy; private set { Set(ref busy, value); CommandManager.InvalidateRequerySuggested(); } }
    public bool IsAvailable { get => available; set { Set(ref available, value); CommandManager.InvalidateRequerySuggested(); } }
    public bool IsExpanded { get => expanded; set => Set(ref expanded, value); }
    public string Status { get => status; private set => Set(ref status, value); }
    public string Log { get => log; private set => Set(ref log, value); }
    public string Error { get => error; private set => Set(ref error, value); }
    public CoreRunResult? LastResult { get; private set; }
    public event EventHandler? Completed;
    public event Action<CoreOutput>? OutputReceived;
    public string ConversationContextKey
    {
        get { var current = settings(); return current.ProjectRoot + "\n" + current.Profile + "\n" + current.RuntimeModelDirectory; }
    }
    public ICommand CancelCommand => new RelayCommand(_ => Cancel(), _ => IsBusy);
    public ICommand OpenLogCommand => new RelayCommand(_ => { if (LastResult is not null) DesktopActions.OpenFolder(LastResult.LogPath); }, _ => LastResult is not null);
    public void Cancel() => cancellation?.Cancel();
    public async Task StopAsync() { Cancel(); if (finished is not null) await finished.Task; }
    public void ShowError(Exception exception) { Error = exception.Message; IsExpanded = true; }
    public Task<CoreRunResult?> RunChatAsync(ChatRunOptions chat, IReadOnlyList<string>? arguments = null) => RunAsync("generate", arguments, chat);
    public async Task<CoreRunResult?> RunAsync(string operation, IReadOnlyList<string>? arguments = null, ChatRunOptions? chat = null)
    {
        if (IsBusy) throw new InvalidOperationException("Hãy đợi hoặc dừng tác vụ Python đang chạy.");
        if (!IsAvailable) throw new InvalidOperationException("Chọn workspace và profile Python trước khi chạy.");
        var version = ++runVersion;
        var operationName = CoreOperations.Get(operation).Name;
        finished = new(TaskCreationOptions.RunContinuationsAsynchronously);
        cancellation = new(); IsBusy = true; IsExpanded = chat is null; Error = "";
        LastResult = null; Raise(nameof(LastResult));
        buffer.Clear(); Log = ""; Status = operationName + " · đang chạy";
        bool dirty = false;
        var timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(150) };
        timer.Tick += (_, _) => { if (version == runVersion && dirty) { Log = buffer.ToString(); dirty = false; } };
        timer.Start();
        var streamed = new GenerationAccumulator();
        var accepting = 1;
        try
        {
            IProgress<CoreOutput> uiProgress = new Progress<CoreOutput>(item =>
                {
                    if (version != runVersion || !IsBusy) return;
                    OutputReceived?.Invoke(item);
                    if (item.StreamEvent is not null) return;
                    buffer.AppendLine(item.Text);
                    if (buffer.Length > 350_000) buffer.Remove(0, buffer.Length - 300_000);
                    dirty = true;
                });
            LastResult = await service.RunAsync(new(settings(), operation, (arguments ?? []).ToArray(), Chat: chat),
                new InlineOutput(item =>
                {
                    if (Volatile.Read(ref accepting) == 0 || version != Interlocked.Read(ref runVersion)) return;
                    if (item.StreamEvent is not null) streamed.Apply(item.StreamEvent);
                    uiProgress.Report(item);
                }), cancellation.Token);
            Interlocked.Exchange(ref accepting, 0);
            if (chat is not null)
            {
                if (LastResult.Generation is not null) streamed.Reconcile(LastResult.Generation);
                var generation = streamed.HasStarted ? streamed.Snapshot() : new GenerationResult("", "", "missing_response", false, 0, "INCOMPLETE_RESPONSE");
                if (LastResult.Cancelled || LastResult.TimedOut || LastResult.ExitCode != 0)
                    generation = generation with { AssistantResponseComplete = false };
                LastResult = LastResult with { Generation = generation,
                    ExitCode = LastResult.ExitCode == 0 && !generation.AssistantResponseComplete ? 2 : LastResult.ExitCode };
            }
            Status = LastResult.Cancelled ? "Đã dừng tác vụ"
                : LastResult.TimedOut ? "Hết thời gian chờ · xem giai đoạn cuối trong nhật ký"
                : LastResult.Generation is { AssistantResponseComplete: false } ? "Phản hồi chưa hoàn tất · có thể tăng giới hạn rồi gửi lại"
                : LastResult.ExitCode == 0 ? "Hoàn tất" : $"Cần xem báo cáo · mã {LastResult.ExitCode}";
            Raise(nameof(LastResult)); Completed?.Invoke(this, EventArgs.Empty);
        }
        catch (OperationCanceledException) { Status = "Đã dừng tác vụ"; }
        catch (Exception exception) { Status = "Không thể hoàn tất"; ShowError(exception); }
        finally { Interlocked.Exchange(ref accepting, 0); timer.Stop(); Log = buffer.ToString(); IsBusy = false; cancellation.Dispose(); cancellation = null; finished.TrySetResult(); }
        return LastResult;
    }
    private sealed class InlineOutput(Action<CoreOutput> action) : IProgress<CoreOutput>
    { public void Report(CoreOutput value) => action(value); }
}
