using System.Text.Json;
using ModelDesk.Core;

namespace ModelDesk.Tests;

public sealed class ConversationProtocolTests
{
    [Fact]
    public void MessagesAreSnapshottedAndOnlyTextRolesAndBoundariesAreAccepted()
    {
        var messages = new List<ChatMessage> { new("user", "Tiếng Việt 😀") };
        var frozen = new ChatRunOptions(messages).ValidateAndSnapshot();
        messages[0] = new("user", "edited");
        Assert.Equal("Tiếng Việt 😀", frozen.Messages[0].Content);
        Assert.Equal("low", frozen.ReasoningEffort);
        Assert.False(frozen.KeepThinking);
        Assert.False(frozen.DirectAnswer);
        foreach (var value in new[] { new ChatMessage("tool", "x"), new ChatMessage("user", "x", "thinking"),
                                     new ChatMessage("user", "<|assistant|>boundary") })
            Assert.Throws<ArgumentException>(() => new ChatRunOptions([value]).ValidateAndSnapshot());
        Assert.Throws<ArgumentException>(() => new ChatRunOptions([new("assistant", "answer")]).ValidateAndSnapshot());
        Assert.Throws<JsonException>(() => JsonSerializer.Deserialize<ChatMessage[]>("[{\"role\":\"user\",\"content\":\"x\",\"tools\":[]}]", ChatRunOptions.JsonOptions));
    }

    [Fact]
    public void DirectAnswerIsAnIndependentOptInAndPreservesSupportedReasoningEffort()
    {
        foreach (var effort in new[] { "low", "high", "max" })
        {
            var options = new ChatRunOptions([new("user", "hello")], effort, DirectAnswer: true).ValidateAndSnapshot();
            Assert.True(options.DirectAnswer);
            Assert.Equal(effort, options.ReasoningEffort);
            Assert.False(options.KeepThinking);
        }
        Assert.Throws<ArgumentException>(() => new ChatRunOptions([new("user", "hello")], "none", DirectAnswer: true).ValidateAndSnapshot());
    }

    [Fact]
    public void ParserAndAccumulatorApplyUtf16SuffixCorrectionsWithoutSplittingEmoji()
    {
        var state = new GenerationAccumulator();
        Assert.True(state.Apply(new("response_start", PromptFormat: "chat", ThinkingOpen: true)));
        Assert.True(state.Apply(Parse(new { @event = "output", channel = "assistant", replace_from_utf16 = 0, text = "A😀�", generated_tokens = 1 })));
        Assert.True(state.Apply(Parse(new { @event = "output", channel = "assistant", replace_from_utf16 = 3, text = "é", generated_tokens = 2 })));
        Assert.Equal("A😀é", state.Snapshot().Text);
        Assert.False(state.Apply(new("output", "assistant", 2, "bad", 3)));
        state.Apply(new("response_end", GeneratedTokens: 3, Status: "GENERATED_UNVERIFIED", StopReason: "eos_token", AssistantResponseComplete: true));
        Assert.False(state.Snapshot().AssistantResponseComplete);
        Assert.Equal("A😀é", state.Snapshot().Text);
    }

    [Fact]
    public void ReasoningAndEmptySuccessNeverBecomeACompleteAssistantReply()
    {
        var state = new GenerationAccumulator();
        state.Apply(new("response_start", PromptFormat: "chat", ThinkingOpen: true));
        state.Apply(new("output", "reasoning", 0, "Chưa trả lời", 5));
        state.Apply(new("response_end", GeneratedTokens: 5, Status: "GENERATED_UNVERIFIED", StopReason: "eos_token", AssistantResponseComplete: true));
        Assert.False(state.Snapshot().AssistantResponseComplete);
        Assert.Equal("INCOMPLETE_RESPONSE", state.Snapshot().Status);
        Assert.Equal("Chưa trả lời", state.Snapshot().Reasoning);
    }

    [Fact]
    public void ResponseBoundsAreFiniteAndFinalReportCanReconcileAValidCorrection()
    {
        var state = new GenerationAccumulator();
        state.Apply(new("response_start", PromptFormat: "raw", ThinkingOpen: false));
        var chunk = new string('x', 2048);
        for (var index = 0; index < 512; index++) Assert.True(state.Apply(new("output", "assistant", index * 2048, chunk, index + 1)));
        Assert.False(state.Apply(new("output", "assistant", 1024 * 1024, "overflow", 513)));
        Assert.True(state.Snapshot().OutputTruncated);
        Assert.Equal(1024 * 1024, state.Snapshot().Text.Length);
        state.Reconcile(new("Đã sửa 😀", "Chi tiết", "eos_token", true, 12, "GENERATED_UNVERIFIED"));
        Assert.Equal("Đã sửa 😀", state.Snapshot().Text);
        Assert.True(state.Snapshot().AssistantResponseComplete);
        Assert.False(state.Snapshot().OutputTruncated);
        state.Reconcile(new("", "only reasoning", "max_tokens", true, 32, "GENERATED_UNVERIFIED"));
        Assert.False(state.Snapshot().AssistantResponseComplete);
    }

    [Theory]
    [InlineData("MODELDESK_EVENT {bad}")]
    [InlineData("MODELDESK_EVENT {\"event\":\"output\",\"channel\":\"tool\",\"text\":\"x\",\"replace_from_utf16\":0,\"generated_tokens\":1}")]
    [InlineData("MODELDESK_EVENT {\"event\":\"output\",\"channel\":\"assistant\",\"text\":\"x\",\"replace_from_utf16\":-1,\"generated_tokens\":1}")]
    [InlineData("MODELDESK_EVENT {\"event\":\"response_start\",\"event\":\"response_end\"}")]
    public void MalformedEventsAreDiagnosedWithoutThrowingOrExecuting(string line)
    {
        Assert.False(ModelStreamEventParser.TryParse(line, out var value, out var error));
        Assert.Null(value);
        Assert.NotNull(error);
    }

    private static ModelStreamEvent Parse(object value)
    {
        Assert.True(ModelStreamEventParser.TryParse(ModelStreamEventParser.Prefix + JsonSerializer.Serialize(value), out var parsed, out var error));
        Assert.Null(error);
        return parsed!;
    }
}
