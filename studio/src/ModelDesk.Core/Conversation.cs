using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace ModelDesk.Core;

public sealed record ChatMessage(
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("content")] string Content,
    [property: JsonPropertyName("reasoning_content"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? ReasoningContent = null);

public sealed record ChatRunOptions(IReadOnlyList<ChatMessage> Messages, string ReasoningEffort = "low", bool KeepThinking = false,
    bool DirectAnswer = false)
{
    public const int MaximumMessages = 256;
    public const int MaximumBytes = 1024 * 1024;
    public static JsonSerializerOptions JsonOptions { get; } = new()
    { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping, PropertyNameCaseInsensitive = false,
      UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow, MaxDepth = 8 };
    private static readonly string[] Boundaries = ["[gMASK]", "[MASK]", "[sMASK]", "<sop>", "<eop>",
        "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>", "<|endoftext|>"];

    public ChatRunOptions ValidateAndSnapshot(bool requireUserTail = true)
    {
        if (ReasoningEffort is not ("low" or "high" or "max")) throw new ArgumentException("Reasoning effort must be low, high or max.");
        if (Messages is null || Messages.Count is < 1 or > MaximumMessages) throw new ArgumentException("Chat requires 1..256 messages.");
        var snapshot = Messages.ToArray();
        long bytes = 0;
        foreach (var message in snapshot)
        {
            if (message is null || message.Role is not ("system" or "user" or "assistant") || message.Content is null ||
                message.ReasoningContent is not null && message.Role != "assistant")
                throw new ArgumentException("Chat supports text system, user and assistant messages only.");
            foreach (var text in new[] { message.Content, message.ReasoningContent ?? "" })
            {
                if (Boundaries.Any(marker => text.Contains(marker, StringComparison.Ordinal)))
                    throw new ArgumentException("Chat text contains reserved model conversation markers.");
                bytes += ConversationText.ByteCount(text);
            }
            if (bytes > MaximumBytes) throw new ArgumentException("Conversation text exceeds 1 MiB. Start a new conversation.");
        }
        if (requireUserTail && (snapshot[^1].Role != "user" || string.IsNullOrWhiteSpace(snapshot[^1].Content)))
            throw new ArgumentException("Chat requires a nonempty final user message.");
        if (JsonSerializer.SerializeToUtf8Bytes(snapshot, JsonOptions).Length > MaximumBytes)
            throw new ArgumentException("Conversation JSON exceeds 1 MiB. Start a new conversation.");
        return this with { Messages = Array.AsReadOnly(snapshot) };
    }
}

public sealed record ModelStreamEvent(string Event, string? Channel = null, int? ReplaceFromUtf16 = null,
    string? Text = null, int? GeneratedTokens = null, string? PromptFormat = null, bool? ThinkingOpen = null,
    string? Status = null, string? StopReason = null, bool? AssistantResponseComplete = null);

public sealed record GenerationResult(string Text, string Reasoning, string? StopReason,
    bool AssistantResponseComplete, int GeneratedTokens, string Status, bool OutputTruncated = false);

public static class ModelStreamEventParser
{
    public const string Prefix = "MODELDESK_EVENT ";
    public const int MaximumChunkBytes = 2048;
    public static bool TryParse(string line, out ModelStreamEvent? value, out string? error)
    {
        value = null; error = null;
        if (!line.StartsWith(Prefix, StringComparison.Ordinal)) return false;
        try
        {
            if (line.Length > 16384) throw new ArgumentException("Stream event exceeds line bounds.");
            using var json = JsonDocument.Parse(line.AsMemory(Prefix.Length), new JsonDocumentOptions { MaxDepth = 8 });
            var root = json.RootElement;
            if (root.ValueKind != JsonValueKind.Object || root.EnumerateObject().Select(item => item.Name).Distinct(StringComparer.Ordinal).Count() != root.EnumerateObject().Count())
                throw new ArgumentException("Stream event must have unique object properties.");
            var kind = String(root, "event");
            if (kind == "response_start")
            {
                var format = String(root, "prompt_format");
                if (format is not ("raw" or "chat")) throw new ArgumentException("Unknown stream prompt format.");
                value = new(kind, PromptFormat: format, ThinkingOpen: Boolean(root, "thinking_open"));
            }
            else if (kind == "output")
            {
                var channel = String(root, "channel");
                var text = String(root, "text");
                if (channel is not ("assistant" or "reasoning") || ConversationText.ByteCount(text) > MaximumChunkBytes)
                    throw new ArgumentException("Invalid output channel or oversized chunk.");
                value = new(kind, channel, Integer(root, "replace_from_utf16"), text, Integer(root, "generated_tokens"));
            }
            else if (kind == "response_end")
            {
                var status = String(root, "status");
                var stop = String(root, "stop_reason");
                if (status.Length > 64 || stop.Length > 128) throw new ArgumentException("Stream state exceeds its bounds.");
                value = new(kind, GeneratedTokens: Integer(root, "generated_tokens"), Status: status,
                    StopReason: stop, AssistantResponseComplete: Boolean(root, "assistant_response_complete"));
            }
            else throw new ArgumentException("Unknown stream event type.");
            return true;
        }
        catch (Exception exception) when (exception is JsonException or ArgumentException or InvalidOperationException or FormatException)
        {
            error = "A malformed model response event was ignored; the final report will be checked.";
            return false;
        }
    }
    private static string String(JsonElement json, string name) => json.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String
        ? value.GetString()! : throw new ArgumentException("Stream string field is absent.");
    private static int Integer(JsonElement json, string name) => json.TryGetProperty(name, out var value) && value.TryGetInt32(out var number) && number is >= 0 and <= 1048576
        ? number : throw new ArgumentException("Stream integer is outside its bound.");
    private static bool Boolean(JsonElement json, string name) => json.TryGetProperty(name, out var value) && value.ValueKind is JsonValueKind.True or JsonValueKind.False
        ? value.GetBoolean() : throw new ArgumentException("Stream boolean field is absent.");
}

public sealed class GenerationAccumulator
{
    public const int MaximumBytes = 1024 * 1024;
    private readonly object _sync = new();
    private readonly StringBuilder _assistant = new(), _reasoning = new();
    private int _bytes, _tokens;
    private bool _started, _ended, _complete, _truncated, _streamInvalid;
    private string _status = "RUNNING";
    private string? _stop;
    public bool HasStarted { get { lock (_sync) return _started; } }

    public bool Apply(ModelStreamEvent value)
    {
        lock (_sync)
        {
            try
            {
                if (value.Event == "response_start")
                {
                    if (_started || value.PromptFormat is not ("raw" or "chat")) throw new ArgumentException("Invalid response start.");
                    _started = true;
                    return true;
                }
                if (!_started || _ended) throw new ArgumentException("Response event is outside its lifetime.");
                var tokens = value.GeneratedTokens ?? throw new ArgumentException("Missing token count.");
                if (tokens < _tokens || tokens > 1048576) throw new ArgumentException("Token count regressed.");
                if (value.Event == "output")
                {
                    var builder = value.Channel switch { "assistant" => _assistant, "reasoning" => _reasoning, _ => throw new ArgumentException("Unknown channel.") };
                    var position = value.ReplaceFromUtf16 ?? throw new ArgumentException("Missing suffix offset.");
                    var text = value.Text ?? throw new ArgumentException("Missing output text.");
                    if (position < 0 || position > builder.Length || position > 0 && position < builder.Length &&
                        char.IsHighSurrogate(builder[position - 1]) && char.IsLowSurrogate(builder[position]))
                        throw new ArgumentException("Output edit splits a Unicode scalar.");
                    var appended = ConversationText.ByteCount(text);
                    if (appended > ModelStreamEventParser.MaximumChunkBytes) throw new ArgumentException("Output chunk exceeds bounds.");
                    var removed = position == builder.Length ? 0 : ConversationText.ByteCount(builder.ToString(position, builder.Length - position));
                    if (_bytes - removed + appended > MaximumBytes)
                    {
                        _truncated = true;
                        throw new ArgumentException("Output buffer exceeds its bound.");
                    }
                    builder.Remove(position, builder.Length - position).Append(text);
                    _bytes = _bytes - removed + appended;
                }
                else if (value.Event == "response_end")
                {
                    _ended = true;
                    _status = value.Status ?? "INCOMPLETE_RESPONSE";
                    _stop = value.StopReason;
                    _complete = value.AssistantResponseComplete == true && _status == "GENERATED_UNVERIFIED" &&
                        !string.IsNullOrWhiteSpace(_assistant.ToString()) && !_truncated && !_streamInvalid;
                    if (!_complete && _status == "GENERATED_UNVERIFIED") _status = "INCOMPLETE_RESPONSE";
                    if (_streamInvalid) _status = "STREAM_ERROR";
                }
                else throw new ArgumentException("Unknown response event.");
                _tokens = tokens;
                return true;
            }
            catch (ArgumentException)
            {
                _complete = false;
                _streamInvalid = true;
                _status = _truncated ? "OUTPUT_TRUNCATED" : "STREAM_ERROR";
                return false;
            }
        }
    }

    public GenerationResult Snapshot()
    {
        lock (_sync) return new(_assistant.ToString(), _reasoning.ToString(), _stop, _complete, _tokens, _status, _truncated);
    }

    public void MarkStreamError()
    {
        lock (_sync) { _streamInvalid = true; _complete = false; _status = "STREAM_ERROR"; }
    }

    public void Reconcile(GenerationResult final)
    {
        lock (_sync)
        {
            var text = ConversationText.TakeBytes(final.Text ?? "", MaximumBytes, out var textTruncated);
            var reasoning = ConversationText.TakeBytes(final.Reasoning ?? "", MaximumBytes - ConversationText.ByteCount(text), out var reasoningTruncated);
            _assistant.Clear().Append(text); _reasoning.Clear().Append(reasoning);
            _bytes = ConversationText.ByteCount(text) + ConversationText.ByteCount(reasoning);
            _tokens = Math.Clamp(final.GeneratedTokens, 0, 1048576);
            _stop = final.StopReason;
            _truncated = final.OutputTruncated || textTruncated || reasoningTruncated;
            _complete = final.AssistantResponseComplete && final.Status == "GENERATED_UNVERIFIED" && !_truncated && !string.IsNullOrWhiteSpace(text);
            _status = _truncated ? "OUTPUT_TRUNCATED" : _complete ? final.Status :
                final.Status == "GENERATED_UNVERIFIED" ? "INCOMPLETE_RESPONSE" : final.Status;
            _started = _ended = true;
            _streamInvalid = false;
        }
    }
}

internal static class ConversationText
{
    private static readonly UTF8Encoding Strict = new(false, true);
    internal static int ByteCount(string text)
    {
        try { return Strict.GetByteCount(text); }
        catch (EncoderFallbackException) { throw new ArgumentException("Conversation text contains invalid Unicode."); }
    }
    internal static string TakeBytes(string text, int maximum, out bool truncated)
    {
        if (ByteCount(text) <= maximum) { truncated = false; return text; }
        var bytes = 0; var characters = 0;
        foreach (var rune in text.EnumerateRunes())
        {
            if (bytes + rune.Utf8SequenceLength > maximum) break;
            bytes += rune.Utf8SequenceLength; characters += rune.Utf16SequenceLength;
        }
        truncated = true;
        return text[..characters];
    }
}
