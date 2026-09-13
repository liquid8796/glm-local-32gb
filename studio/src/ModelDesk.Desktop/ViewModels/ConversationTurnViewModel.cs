using ModelDesk.Core;
using ModelDesk.Desktop.Presentation;

namespace ModelDesk.Desktop.ViewModels;

public sealed class ConversationTurnViewModel(string userText) : ObservableObject
{
    private string response = "", reasoning = "", state = "Đang bắt đầu…", stopReason = "";
    private bool running = true, complete;
    private int tokens;
    public string UserText { get; } = userText;
    public string Response { get => response; private set { Set(ref response, value); Raise(nameof(ResponseDisplay)); Raise(nameof(CanCopy)); } }
    public string Reasoning { get => reasoning; private set { Set(ref reasoning, value); Raise(nameof(HasReasoning)); } }
    public string State { get => state; private set => Set(ref state, value); }
    public string StopReason { get => stopReason; private set => Set(ref stopReason, value); }
    public int GeneratedTokens { get => tokens; private set => Set(ref tokens, value); }
    public bool IsRunning { get => running; private set { Set(ref running, value); Raise(nameof(ResponseDisplay)); } }
    public bool IsComplete { get => complete; private set => Set(ref complete, value); }
    public bool HasReasoning => Reasoning.Length > 0;
    public bool CanCopy => !string.IsNullOrWhiteSpace(Response);
    public string ResponseDisplay => !string.IsNullOrWhiteSpace(Response) ? Response : IsRunning
        ? HasReasoning ? "Model đang suy nghĩ; phần trả lời sẽ xuất hiện tại đây." : "Đang xử lý tin nhắn…"
        : "Chưa có phần trả lời hiển thị.";

    internal void Update(GenerationResult value, bool finished = false)
    {
        Response = value.Text; Reasoning = value.Reasoning;
        GeneratedTokens = value.GeneratedTokens; StopReason = value.StopReason ?? "";
        IsComplete = finished && value.AssistantResponseComplete && !value.OutputTruncated && !string.IsNullOrWhiteSpace(value.Text);
        IsRunning = !finished;
        State = !finished ? $"Đang sinh · {GeneratedTokens} token" : IsComplete ? $"Hoàn tất · {GeneratedTokens} token"
            : value.OutputTruncated ? "Nội dung vượt giới hạn hiển thị · chưa thêm vào ngữ cảnh"
            : value.StopReason == "timeout" ? "Hết thời gian · giữ lại phần đang có"
            : value.StopReason == "cancelled" ? "Đã dừng · giữ lại phần đang có"
            : value.StopReason == "max_tokens" ? "Hết token · tăng giới hạn và gửi lại; chưa thêm vào ngữ cảnh"
            : "Chưa hoàn tất · chưa thêm vào ngữ cảnh";
        Raise(nameof(ResponseDisplay));
        System.Windows.Input.CommandManager.InvalidateRequerySuggested();
    }
    internal void Failed(string message) { IsRunning = false; IsComplete = false; State = message; }
}
