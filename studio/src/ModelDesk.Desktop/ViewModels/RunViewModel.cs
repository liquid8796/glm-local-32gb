using System.Collections.ObjectModel;
using System.Globalization;
using System.Windows;
using System.Windows.Input;
using ModelDesk.Core;
using ModelDesk.Desktop.Presentation;

namespace ModelDesk.Desktop.ViewModels;

public sealed class RunViewModel : ObservableObject
{
    private readonly CoreTaskViewModel task;
    private readonly List<ChatMessage> history = [];
    private string directory = "", prompt = "", tokens = "", backend = "cpu", context = "4096", generate = "256", timeout = "1800", advanced = "";
    private string promptFormat = "chat", reasoningEffort = "low", historyKey = "";
    private bool useTokens, keepThinking, directAnswer, sending;
    private long version;
    private ConversationTurnViewModel? activeTurn;
    private GenerationAccumulator? activeResponse;

    public RunViewModel(CoreTaskViewModel task)
    {
        this.task = task;
        PlanCommand = new AsyncCommand(_ => task.RunAsync("runtime-plan", Arguments(false)), task.ShowError, _ => task.IsAvailable && !task.IsBusy);
        GenerateCommand = new AsyncCommand(_ => SendAsync(), task.ShowError, _ => CanSend);
        BrowseCommand = new RelayCommand(_ => ModelDirectory = DesktopActions.ChooseFolder("Chọn thư mục trọng số", ModelDirectory) ?? ModelDirectory, _ => CanConfigure);
        ResetCommand = new RelayCommand(_ => ResetConversation());
        CopyResponseCommand = new RelayCommand(item =>
        {
            var text = CopyText(item as ConversationTurnViewModel);
            if (text.Length > 0) Clipboard.SetText(text);
        }, item => item is ConversationTurnViewModel { CanCopy: true });
        task.OutputReceived += Receive;
        task.PropertyChanged += (_, args) =>
        {
            if (args.PropertyName is nameof(CoreTaskViewModel.IsBusy) or nameof(CoreTaskViewModel.IsAvailable))
            { Raise(nameof(CanConfigure)); Raise(nameof(CanSend)); CommandManager.InvalidateRequerySuggested(); }
        };
    }

    public ObservableCollection<ConversationTurnViewModel> Turns { get; } = [];
    public string ModelDirectory { get => directory; set { if (Set(ref directory, value)) { history.Clear(); historyKey = ""; Raise(nameof(HistorySummary)); } } }
    public string Prompt { get => prompt; set { Set(ref prompt, value); Raise(nameof(CanSend)); CommandManager.InvalidateRequerySuggested(); } }
    public string Tokens { get => tokens; set { Set(ref tokens, value); Raise(nameof(CanSend)); CommandManager.InvalidateRequerySuggested(); } }
    public bool UseTokens { get => useTokens; set { Set(ref useTokens, value); Raise(nameof(IsTextInput)); Raise(nameof(IsChatInput)); Raise(nameof(SendLabel)); Raise(nameof(CanSend)); } }
    public string PromptFormat { get => promptFormat; set { Set(ref promptFormat, value); Raise(nameof(IsChatInput)); Raise(nameof(SendLabel)); } }
    public string ReasoningEffort { get => reasoningEffort; set => Set(ref reasoningEffort, value); }
    public bool KeepThinking { get => keepThinking; set => Set(ref keepThinking, value); }
    public bool DirectAnswer { get => directAnswer; set => Set(ref directAnswer, value); }
    public string Backend { get => backend; set => Set(ref backend, value); }
    public string Context { get => context; set => Set(ref context, value); }
    public string Generate { get => generate; set => Set(ref generate, value); }
    public string Timeout { get => timeout; set => Set(ref timeout, value); }
    public string Advanced { get => advanced; set => Set(ref advanced, value); }
    public bool IsTextInput => !UseTokens;
    public bool IsChatInput => IsTextInput && PromptFormat == "chat";
    public bool IsSending { get => sending; private set { Set(ref sending, value); Raise(nameof(CanSend)); } }
    public bool CanConfigure => !task.IsBusy;
    public bool CanSend => task.IsAvailable && !task.IsBusy && !IsSending && !string.IsNullOrWhiteSpace(UseTokens ? Tokens : Prompt);
    public string SendLabel => UseTokens ? "Chạy token ID" : PromptFormat == "raw" ? "Chạy prompt thô" : "Gửi tin nhắn";
    public string HistorySummary => $"{history.Count / 2} lượt hoàn tất trong ngữ cảnh. Các lượt dừng/chưa hoàn tất chỉ được giữ để xem.";
    public IReadOnlyList<ChatMessage> ContextMessages => HistoryForRequest(KeepThinking).ToArray();
    public string[] Backends { get; } = ["cpu", "hybrid"];
    public string[] PromptFormats { get; } = ["chat", "raw"];
    public string[] ReasoningEfforts { get; } = ["low", "high", "max"];
    public ICommand PlanCommand { get; }
    public ICommand GenerateCommand { get; }
    public ICommand BrowseCommand { get; }
    public ICommand ResetCommand { get; }
    public ICommand CopyResponseCommand { get; }

    public async Task SendAsync()
    {
        if (!CanSend) throw new InvalidOperationException("Nhập nội dung và đợi tác vụ hiện tại kết thúc.");
        var arguments = Arguments(true);
        var wasTokens = UseTokens;
        var submitted = wasTokens ? Tokens : Prompt;
        var asChat = !wasTokens && PromptFormat == "chat";
        var key = task.ConversationContextKey + "\n" + ModelDirectory;
        if (historyKey != key) { history.Clear(); historyKey = key; Raise(nameof(HistorySummary)); }
        var keep = KeepThinking;
        var effort = ReasoningEffort;
        ChatRunOptions? chat = asChat ? new ChatRunOptions([.. HistoryForRequest(keep), new("user", submitted)], effort, keep, DirectAnswer).ValidateAndSnapshot() : null;
        var current = ++version;
        var turn = new ConversationTurnViewModel(wasTokens ? "Token ID: " + submitted : submitted);
        var accumulator = new GenerationAccumulator();
        activeTurn = turn; activeResponse = accumulator;
        Turns.Add(turn); IsSending = true;
        try
        {
            var result = chat is not null ? await task.RunChatAsync(chat, arguments) : await task.RunAsync("generate", arguments);
            if (current != version) return;
            var generation = result?.Generation ?? accumulator.Snapshot();
            if (result is null || result.Cancelled || result.TimedOut || result.ExitCode != 0)
                generation = generation with { AssistantResponseComplete = false,
                    StopReason = result?.Cancelled == true ? "cancelled" : result?.TimedOut == true ? "timeout" : generation.StopReason,
                    Status = result is null ? "ERROR" : generation.Status };
            accumulator.Reconcile(generation);
            generation = accumulator.Snapshot();
            turn.Update(generation, finished: true);
            if (result is null) turn.Failed("Không thể hoàn tất · mở nhật ký để xem lỗi");
            if (asChat && historyKey == key && key == task.ConversationContextKey + "\n" + ModelDirectory &&
                result is { ExitCode: 0, Cancelled: false, TimedOut: false } && generation.AssistantResponseComplete && !generation.OutputTruncated)
            {
                // Eligibility is based on completed visible messages. Keep the already
                // displayed reasoning by reference, and only resend it when requested.
                new ChatRunOptions([.. HistoryForRequest(false), new("user", submitted), new("assistant", generation.Text)], effort)
                    .ValidateAndSnapshot(requireUserTail: false);
                history.Add(new("user", submitted));
                history.Add(new("assistant", generation.Text, generation.Reasoning.Length > 0 ? generation.Reasoning : null));
                if (Prompt == submitted) Prompt = "";
                Raise(nameof(HistorySummary));
            }
            else if (wasTokens && result is { ExitCode: 0 } && result.Generation is null)
                turn.Failed("Tác vụ token đã kết thúc · xem token ID trong báo cáo");
        }
        catch (Exception exception)
        {
            if (current == version) turn.Failed(exception.Message);
            throw;
        }
        finally
        {
            if (current == version) { activeTurn = null; activeResponse = null; IsSending = false; }
        }
    }

    private void Receive(CoreOutput output)
    {
        if (!IsSending || activeTurn is null || activeResponse is null || output.StreamEvent is null) return;
        if (activeResponse.Apply(output.StreamEvent)) activeTurn.Update(activeResponse.Snapshot());
    }

    public void ResetConversation()
    {
        if (IsSending) task.Cancel();
        version++; activeTurn = null; activeResponse = null; IsSending = false;
        history.Clear(); historyKey = ""; Turns.Clear(); Raise(nameof(HistorySummary));
    }
    public static string CopyText(ConversationTurnViewModel? turn) => turn?.Response ?? "";
    private IEnumerable<ChatMessage> HistoryForRequest(bool keep) => history.Select(message => keep || message.ReasoningContent is null
        ? message : message with { ReasoningContent = null });

    private IReadOnlyList<string> Arguments(bool generation)
    {
        Positive(Context, "Context"); Positive(Generate, "Số token");
        if (Backend is not ("cpu" or "hybrid")) throw new ArgumentException("Backend không hợp lệ.");
        if (PromptFormat is not ("chat" or "raw")) throw new ArgumentException("Định dạng prompt không hợp lệ.");
        List<string> arguments = ["--backend", Backend, "--context", Context, "--generate", Generate];
        if (generation)
        {
            if (string.IsNullOrWhiteSpace(ModelDirectory)) throw new ArgumentException("Chọn thư mục trọng số cục bộ.");
            Positive(Timeout, "Timeout");
            arguments.AddRange(["--model-directory", ModelDirectory, "--timeout", Timeout]);
            if (UseTokens) arguments.AddRange(["--tokens", Tokens]);
            else if (PromptFormat == "raw") arguments.AddRange(["--prompt", Prompt, "--prompt-format", "raw", "--stream-events"]);
        }
        arguments.AddRange(ArgumentTokenizer.Parse(Advanced));
        return arguments;
    }
    private static void Positive(string value, string name)
    {
        if (!int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out var number) || number < 1) throw new ArgumentException(name + " phải là số nguyên dương.");
    }
}
