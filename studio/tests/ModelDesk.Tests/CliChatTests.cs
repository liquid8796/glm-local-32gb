using System.Text;
using System.Text.Json;
using ModelDesk.Cli;
using ModelDesk.Core;

namespace ModelDesk.Tests;

public sealed class CliChatTests
{
    [Fact]
    public async Task Prompt_system_and_per_run_options_use_chat_dto_with_literal_unicode()
    {
        var test = new Fixtures();
        const string prompt = "Xin chào 🙂 $(whoami) & echo literal";
        const string directory = @"D:\Mô hình\GLM NVFP4";
        Assert.Equal(0, await test.App.RunAsync(["chat", "--prompt", prompt, "--system", "Trả lời ngắn bằng tiếng Việt.",
            "--profile", "fp8", "--model-directory", directory, "--context", "8192", "--max-tokens", "512",
            "--timeout", "3600", "--backend", "hybrid", "--reasoning", "high", "--keep-thinking", "--json"]));
        var request = Assert.IsType<CoreRunRequest>(test.Core.Request);
        Assert.Equal("generate", request.Operation); Assert.Null(request.ConfigPath);
        Assert.Equal("fp8", request.Settings.Profile);
        Assert.Equal(["--backend", "hybrid", "--context", "8192", "--generate", "512", "--timeout", "3600", "--model-directory", directory], request.Arguments);
        var chat = Assert.IsType<ChatRunOptions>(request.Chat);
        Assert.Equal("high", chat.ReasoningEffort); Assert.True(chat.KeepThinking);
        Assert.Equal([new ChatMessage("system", "Trả lời ngắn bằng tiếng Việt."), new ChatMessage("user", prompt)], chat.Messages);
        Assert.DoesNotContain("--messages-file", request.Arguments);
        Assert.DoesNotContain(prompt, request.Arguments);
        Assert.Equal("nvfp4", test.Settings.Value.Profile); Assert.Equal(0, test.Settings.Saves);
        using var result = JsonDocument.Parse(test.Out.ToString());
        Assert.Equal(0, result.RootElement.GetProperty("ExitCode").GetInt32());
        Assert.Equal("Xin chào🙂", result.RootElement.GetProperty("Generation").GetProperty("Text").GetString());
    }

    [Fact]
    public async Task Defaults_stream_visible_assistant_text_and_keep_logs_off_stdout()
    {
        var test = new Fixtures();
        Assert.Equal(0, await test.App.RunAsync(["chat", "--prompt", "hello"]));
        Assert.Equal(["--backend", "cpu", "--context", "4096", "--generate", "256", "--timeout", "1800"], test.Core.Request!.Arguments);
        Assert.Equal("low", test.Core.Request.Chat!.ReasoningEffort); Assert.False(test.Core.Request.Chat.KeepThinking);
        Assert.Equal("Xin chào🙂" + Environment.NewLine, test.Out.ToString());
        Assert.Contains("loading fixture", test.Error.ToString()); Assert.Contains("[Completed]", test.Error.ToString());
        Assert.DoesNotContain("loading fixture", test.Out.ToString());
    }

    [Fact]
    public async Task Json_flag_is_not_confused_with_a_literal_prompt_value()
    {
        var plain = new Fixtures();
        Assert.Equal(0, await plain.App.RunAsync(["chat", "--prompt", "--json"]));
        Assert.Equal("--json", plain.Core.Request!.Chat!.Messages[0].Content);
        Assert.Equal("Xin chào🙂" + Environment.NewLine, plain.Out.ToString());
        var json = new Fixtures();
        Assert.Equal(0, await json.App.RunAsync(["--json", "chat", "--prompt", "--json"]));
        Assert.Equal("--json", json.Core.Request!.Chat!.Messages[0].Content);
        using var parsed = JsonDocument.Parse(json.Out.ToString());
        Assert.Equal("Xin chào🙂", parsed.RootElement.GetProperty("Generation").GetProperty("Text").GetString());
        var invalid = new Fixtures();
        Assert.Equal(1, await invalid.App.RunAsync(["chat", "--prompt", "hello", "--backend", "gpu", "--json"]));
        using var failure = JsonDocument.Parse(invalid.Out.ToString());
        Assert.True(failure.RootElement.TryGetProperty("error", out _));
    }

    [Fact]
    public async Task Bounded_bom_utf8_history_keeps_order_prior_reasoning_and_file_bytes()
    {
        using var files = new TestDirectory();
        var path = Path.Combine(files.Path, "lịch sử.json");
        ChatMessage[] history = [new("system", "Bạn là trợ lý."), new("user", "Xin chào"),
            new("assistant", "Chào bạn 🙂", "previous fixture reasoning"), new("user", "Tiếp tục nhé")];
        var json = JsonSerializer.SerializeToUtf8Bytes(history, ChatRunOptions.JsonOptions);
        var raw = Encoding.UTF8.GetPreamble().Concat(json).ToArray();
        await File.WriteAllBytesAsync(path, raw);
        var test = new Fixtures();
        Assert.Equal(0, await test.App.RunAsync(["chat", "--messages", path, "--reasoning", "max", "--keep-thinking"]));
        Assert.Equal(history, test.Core.Request!.Chat!.Messages);
        Assert.True(test.Core.Request.Chat.KeepThinking);
        Assert.Equal(raw, await File.ReadAllBytesAsync(path));
        Assert.DoesNotContain(path, test.Core.Request.Arguments);
        Assert.Equal(0, test.Settings.Saves);
    }

    [Fact]
    public async Task Assistant_text_is_written_before_the_final_core_result_arrives()
    {
        var test = new Fixtures();
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        test.Core.Handler = async (_, progress, _) =>
        {
            Emit(progress, new("response_start", PromptFormat: "chat", ThinkingOpen: false));
            Emit(progress, new("output", "assistant", 0, "xin", 1));
            await release.Task;
            return Result("xin chào", complete: true);
        };
        var run = test.App.RunAsync(["chat", "--prompt", "hello"]);
        try
        {
            Assert.False(run.IsCompleted); Assert.Equal("xin", test.Out.ToString());
            release.TrySetResult(); Assert.Equal(0, await run);
            Assert.Equal("xin chào" + Environment.NewLine, test.Out.ToString());
        }
        finally { release.TrySetResult(); await run; }
    }

    [Fact]
    public async Task Suffix_revisions_are_reconciled_and_clearly_marked_in_plain_output()
    {
        var test = new Fixtures();
        test.Core.Handler = (_, progress, _) =>
        {
            Emit(progress, new("response_start", PromptFormat: "chat", ThinkingOpen: false));
            Emit(progress, new("output", "assistant", 0, "🙂a", 1));
            Emit(progress, new("output", "assistant", 2, "b", 2));
            Emit(progress, new("response_end", GeneratedTokens: 2, Status: "GENERATED_UNVERIFIED", StopReason: "eos", AssistantResponseComplete: true));
            return Task.FromResult(Result("🙂b", complete: true));
        };
        Assert.Equal(0, await test.App.RunAsync(["chat", "--prompt", "hello"]));
        Assert.Contains("[assistant revised]", test.Out.ToString());
        Assert.EndsWith("🙂b" + Environment.NewLine, test.Out.ToString());
        Assert.DoesNotContain("🙂ab", test.Out.ToString());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Live_reasoning_is_optional_labeled_and_separate_from_assistant(bool show)
    {
        var test = new Fixtures();
        test.Core.Handler = (_, progress, _) =>
        {
            Emit(progress, new("response_start", PromptFormat: "chat", ThinkingOpen: true));
            Emit(progress, new("output", "reasoning", 0, "fixture thought", 1));
            Emit(progress, new("output", "assistant", 0, "visible answer", 2));
            return Task.FromResult(Result("visible answer", complete: true, reasoning: "fixture thought"));
        };
        string[] args = show ? ["chat", "--prompt", "hello", "--keep-thinking"] : ["chat", "--prompt", "hello"];
        Assert.Equal(0, await test.App.RunAsync(args));
        Assert.Equal("visible answer" + Environment.NewLine, test.Out.ToString());
        Assert.DoesNotContain("fixture thought", test.Out.ToString());
        if (show) { Assert.Contains("[reasoning]", test.Error.ToString()); Assert.Contains("fixture thought", test.Error.ToString()); }
        else Assert.DoesNotContain("fixture thought", test.Error.ToString());
    }

    [Fact]
    public async Task Structured_mode_keeps_a_single_result_on_stdout_and_events_on_stderr()
    {
        var test = new Fixtures();
        Assert.Equal(0, await test.App.RunAsync(["--json", "chat", "--prompt", "hello"]));
        using var document = JsonDocument.Parse(test.Out.ToString());
        Assert.True(document.RootElement.GetProperty("Generation").GetProperty("AssistantResponseComplete").GetBoolean());
        Assert.DoesNotContain("response_start", test.Out.ToString());
        Assert.Contains("response_start", test.Error.ToString());
        Assert.Contains("loading fixture", test.Error.ToString());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Partial_and_reasoning_only_results_keep_exit_two(bool thinkingOnly)
    {
        var test = new Fixtures();
        test.Core.Handler = (_, _, _) => Task.FromResult(Result(thinkingOnly ? "" : "partial", complete: false,
            reasoning: "unfinished fixture reasoning", code: 2));
        Assert.Equal(2, await test.App.RunAsync(["chat", "--prompt", "hello"]));
        Assert.Equal(thinkingOnly ? "" : "partial" + Environment.NewLine, test.Out.ToString());
        Assert.Contains("[Incomplete response]", test.Error.ToString());
        Assert.DoesNotContain("[Completed]", test.Error.ToString());
        var structured = new Fixtures(); structured.Core.Handler = test.Core.Handler;
        Assert.Equal(2, await structured.App.RunAsync(["--json", "chat", "--prompt", "hello"]));
        using var json = JsonDocument.Parse(structured.Out.ToString());
        Assert.Equal(2, json.RootElement.GetProperty("ExitCode").GetInt32());
        Assert.False(json.RootElement.GetProperty("Generation").GetProperty("AssistantResponseComplete").GetBoolean());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Alternate_core_zero_exit_does_not_promote_missing_or_incomplete_chat(bool missing)
    {
        var test = new Fixtures();
        test.Core.Handler = (_, _, _) => Task.FromResult(missing ? Result("") with { Generation = null } : Result("partial", false, code: 0));
        Assert.Equal(2, await test.App.RunAsync(["--json", "chat", "--prompt", "hello"]));
        using var json = JsonDocument.Parse(test.Out.ToString());
        Assert.Equal(2, json.RootElement.GetProperty("ExitCode").GetInt32());
        Assert.False(json.RootElement.GetProperty("Generation").GetProperty("AssistantResponseComplete").GetBoolean());
    }

    [Fact]
    public async Task Model_terminal_controls_are_displayed_as_text_and_preserved_in_json_data()
    {
        const string modelText = "answer\u001b[2J\u009b31m\b";
        var plain = new Fixtures(); plain.Core.Handler = (_, _, _) => Task.FromResult(Result(modelText));
        Assert.Equal(0, await plain.App.RunAsync(["chat", "--prompt", "hello"]));
        Assert.DoesNotContain('\u001b', plain.Out.ToString());
        Assert.DoesNotContain('\u009b', plain.Out.ToString());
        Assert.Contains("\\u001b", plain.Out.ToString());
        var structured = new Fixtures(); structured.Core.Handler = plain.Core.Handler;
        Assert.Equal(0, await structured.App.RunAsync(["--json", "chat", "--prompt", "hello"]));
        using var json = JsonDocument.Parse(structured.Out.ToString());
        Assert.Equal(modelText, json.RootElement.GetProperty("Generation").GetProperty("Text").GetString());
    }

    [Fact]
    public async Task Cancellation_preserves_130_and_never_labels_partial_output_complete()
    {
        var test = new Fixtures();
        test.Core.Handler = (_, _, _) => Task.FromResult(Result("partial", false, code: 130) with { Cancelled = true });
        Assert.Equal(130, await test.App.RunAsync(["chat", "--prompt", "hello"]));
        Assert.Contains("[Cancelled]", test.Error.ToString()); Assert.DoesNotContain("[Completed]", test.Error.ToString());
        var cancelled = new Fixtures(); using var token = new CancellationTokenSource(); token.Cancel();
        Assert.Equal(130, await cancelled.App.RunAsync(["chat", "--prompt", "hello"], token.Token));
        Assert.Equal(0, cancelled.Core.Calls); Assert.Equal("", cancelled.Out.ToString());
    }

    [Theory]
    [InlineData()]
    [InlineData("--prompt", "a", "--messages", "absent.json")]
    [InlineData("--messages", "absent.json", "--system", "system")]
    [InlineData("--prompt", " ")]
    [InlineData("--prompt", "a", "--prompt", "b")]
    [InlineData("--prompt", "a", "--context", "0")]
    [InlineData("--prompt", "a", "--context", "2147483648")]
    [InlineData("--prompt", "a", "--max-tokens", "0")]
    [InlineData("--prompt", "a", "--context", "8", "--max-tokens", "9")]
    [InlineData("--prompt", "a", "--timeout", "0")]
    [InlineData("--prompt", "a", "--timeout", "86401")]
    [InlineData("--prompt", "a", "--backend", "gpu")]
    [InlineData("--prompt", "a", "--reasoning", "medium")]
    [InlineData("--prompt", "a", "--model-directory", "")]
    [InlineData("--prompt", "a", "--profile", "")]
    [InlineData("--prompt")]
    public async Task Invalid_or_conflicting_chat_options_fail_before_core(params string[] args)
    {
        var test = new Fixtures();
        Assert.Equal(1, await test.App.RunAsync(["chat", .. args]));
        Assert.Equal(0, test.Core.Calls); Assert.Equal(0, test.Settings.Saves);
    }

    [Theory]
    [InlineData("{}")]
    [InlineData("[]")]
    [InlineData("[null]")]
    [InlineData("[{\"role\":\"tool\",\"content\":\"x\"}]")]
    [InlineData("[{\"role\":\"assistant\",\"content\":\"x\"}]")]
    [InlineData("[{\"role\":\"user\",\"content\":null}]")]
    [InlineData("[{\"role\":\"user\",\"content\":[\"x\"]}]")]
    [InlineData("[{\"role\":\"user\",\"content\":\"x\",\"reasoning_content\":\"x\"}]")]
    [InlineData("[{\"role\":\"user\",\"content\":\"<|assistant|>\"}]")]
    [InlineData("[{\"role\":\"user\",\"content\":\"x\",\"content\":\"y\"}]")]
    [InlineData("[{\"role\":\"user\",\"content\":\"x\",\"tool_calls\":[]}]")]
    public async Task Invalid_history_roles_shapes_and_duplicate_fields_fail_before_core(string content)
    {
        using var files = new TestDirectory(); var path = Path.Combine(files.Path, "messages.json");
        await File.WriteAllTextAsync(path, content, new UTF8Encoding(false));
        var test = new Fixtures();
        Assert.Equal(1, await test.App.RunAsync(["chat", "--messages", path])); Assert.Equal(0, test.Core.Calls);
    }

    [Fact]
    public async Task History_file_and_inline_prompt_byte_limits_are_enforced()
    {
        using var files = new TestDirectory(); var path = Path.Combine(files.Path, "oversized.json");
        await using (var file = File.Create(path)) file.SetLength(ChatRunOptions.MaximumBytes + 1);
        var test = new Fixtures();
        Assert.Equal(1, await test.App.RunAsync(["chat", "--messages", path]));
        Assert.Equal(1, await test.App.RunAsync(["chat", "--prompt", new string('x', ChatRunOptions.MaximumBytes + 1)]));
        Assert.Equal(0, test.Core.Calls);
        var history = Enumerable.Range(0, ChatRunOptions.MaximumMessages + 1).Select(_ => new ChatMessage("user", "x")).ToArray();
        await File.WriteAllTextAsync(path, JsonSerializer.Serialize(history, ChatRunOptions.JsonOptions));
        Assert.Equal(1, await test.App.RunAsync(["chat", "--messages", path])); Assert.Equal(0, test.Core.Calls);
    }

    [Fact]
    public async Task Unsupported_secret_options_do_not_echo_the_value_or_start_core()
    {
        var test = new Fixtures();
        Assert.Equal(1, await test.App.RunAsync(["chat", "--prompt", "hello", "--api-key", "synthetic-secret-value"]));
        Assert.DoesNotContain("synthetic-secret-value", test.Out.ToString() + test.Error.ToString());
        Assert.Equal(0, test.Core.Calls);
    }

    private static CoreRunResult Result(string text = "Xin chào🙂", bool complete = true, string reasoning = "", int code = 0) =>
        new("chat-fixture", code, false, "fixture.log", "fixture.json", DateTimeOffset.UtcNow, DateTimeOffset.UtcNow,
            Generation: new(text, reasoning, complete ? "eos" : "length", complete, 2, complete ? "GENERATED_UNVERIFIED" : "INCOMPLETE_RESPONSE"));
    private static void Emit(IProgress<CoreOutput>? progress, ModelStreamEvent value) => progress?.Report(new(DateTimeOffset.UtcNow, "", StreamEvent: value));
    private sealed class Fixtures
    {
        public FakeSettings Settings { get; } = new();
        public FakeCore Core { get; } = new();
        public StringWriter Out { get; } = new();
        public StringWriter Error { get; } = new();
        public CliApplication App => new(Settings, new NoHub(), new NoDownloader(), Core, new NoReports(), Out, Error);
    }
    private sealed class FakeSettings : ISettingsStore
    {
        public AppSettings Value { get; } = new() { ProjectRoot = Path.GetTempPath(), DefaultDownloadDirectory = Path.GetTempPath() };
        public int Saves { get; private set; }
        public string FilePath => "unused-chat-settings.json";
        public Task<AppSettings> LoadAsync(CancellationToken cancellationToken = default) => Task.FromResult(Value);
        public Task SaveAsync(AppSettings settings, CancellationToken cancellationToken = default) { Saves++; return Task.CompletedTask; }
    }
    private sealed class FakeCore : IPythonCoreService
    {
        public CoreRunRequest? Request { get; private set; }
        public int Calls { get; private set; }
        public Func<CoreRunRequest, IProgress<CoreOutput>?, CancellationToken, Task<CoreRunResult>>? Handler { get; set; }
        public IReadOnlyList<ModelProfile> GetProfiles(string projectRoot) => [];
        public Task<CoreRunResult> RunAsync(CoreRunRequest request, IProgress<CoreOutput>? progress = null, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested(); Calls++; Request = request;
            if (Handler is not null) return Handler(request, progress, cancellationToken);
            progress?.Report(new(DateTimeOffset.UtcNow, "loading fixture"));
            Emit(progress, new("response_start", PromptFormat: "chat", ThinkingOpen: false));
            Emit(progress, new("output", "assistant", 0, "Xin", 1));
            Emit(progress, new("output", "assistant", 3, " chào🙂", 2));
            Emit(progress, new("response_end", GeneratedTokens: 2, Status: "GENERATED_UNVERIFIED", StopReason: "eos", AssistantResponseComplete: true));
            return Task.FromResult(Result());
        }
    }
    private sealed class NoHub : IHuggingFaceClient
    {
        public Task<HubSearchResult> SearchAsync(string query, string? nextPage = null, CancellationToken cancellationToken = default) => throw new InvalidOperationException("Chat must not call the Hub.");
        public Task<HubModelDetail> GetModelAsync(string modelId, string revision = "main", CancellationToken cancellationToken = default) => throw new InvalidOperationException("Chat must not call the Hub.");
    }
    private sealed class NoDownloader : IModelDownloader
    {
        public long BytesPerSecondLimit { get; set; }
        public Task<DownloadResult> DownloadAsync(DownloadRequest request, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default) => throw new InvalidOperationException("Chat must not download models.");
    }
    private sealed class NoReports : IReportService
    {
        public IReadOnlyList<ReportEntry> List(string projectRoot, string? profileReportDirectory = null) => [];
        public Task<string> ReadAsync(string path, CancellationToken cancellationToken = default) => throw new InvalidOperationException("The service owns final report handling.");
    }
    private sealed class TestDirectory : IDisposable
    {
        private static readonly string Boundary = System.IO.Path.GetFullPath(System.IO.Path.Combine(System.IO.Path.GetTempPath(), "ModelDesk.CliChatTests"));
        public string Path { get; } = System.IO.Path.Combine(Boundary, Guid.NewGuid().ToString("N"));
        public TestDirectory() => Directory.CreateDirectory(Path);
        public void Dispose()
        {
            var resolved = System.IO.Path.GetFullPath(Path);
            if (!resolved.StartsWith(Boundary + System.IO.Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase)) throw new IOException("Chat test cleanup escaped its fixture folder.");
            Directory.Delete(resolved, true);
        }
    }
}
