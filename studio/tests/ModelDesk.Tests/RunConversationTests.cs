using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Windows.Threading;
using ModelDesk.Core;
using ModelDesk.Desktop.ViewModels;
using ModelDesk.Infrastructure.Runtime;

namespace ModelDesk.Tests;

public sealed class RunConversationTests
{
    [Fact]
    public Task ChatDefaultsAndCompletedHistoryAreStructuredAndFinalReportWins() => OnDispatcher(async () =>
    {
        var (core, _, view) = Fixture();
        Assert.Equal("chat", view.PromptFormat); Assert.Equal("low", view.ReasoningEffort); Assert.Equal("256", view.Generate);
        Assert.False(view.DirectAnswer);
        view.Prompt = "first 😀";
        var run = view.SendAsync();
        var call = Assert.Single(core.Calls);
        Assert.Equal("generate", call.Request.Operation);
        Assert.Equal("first 😀", Assert.Single(call.Request.Chat!.Messages).Content);
        Assert.False(call.Request.Chat.DirectAnswer);
        Assert.DoesNotContain("--prompt", call.Request.Arguments);
        call.Emit(new("response_start", PromptFormat: "chat", ThinkingOpen: true));
        call.Emit(new("output", "reasoning", 0, "private model reasoning", 1));
        call.Emit(new("output", "assistant", 0, "A😀�", 2));
        call.Emit(new("output", "assistant", 3, "é", 3));
        await Drain();
        Assert.Equal("A😀é", view.Turns[0].Response);
        Assert.True(view.Turns[0].HasReasoning);
        call.Finish(Result("A😀é final", reasoning: "model reasoning"));
        await run;
        Assert.True(view.Turns[0].IsComplete);
        Assert.Equal("A😀é final", RunViewModel.CopyText(view.Turns[0]));
        Assert.Equal(2, view.ContextMessages.Count);
        Assert.Null(view.ContextMessages[1].ReasoningContent);
        view.Prompt = "second";
        var next = view.SendAsync();
        Assert.Equal(["user", "assistant", "user"], core.Calls[1].Request.Chat!.Messages.Select(item => item.Role).ToArray());
        Assert.Equal("A😀é final", core.Calls[1].Request.Chat!.Messages[1].Content);
        core.Calls[1].Finish(Result("second answer"));
        await next;
    });

    [Fact]
    public Task DirectAnswerIsSnapshottedPerRunAndKeepsCompletedHistoryRules() => OnDispatcher(async () =>
    {
        var (core, _, view) = Fixture(); view.Prompt = "answer directly"; view.DirectAnswer = true;
        var run = view.SendAsync();
        var call = Assert.Single(core.Calls);
        view.DirectAnswer = false;
        Assert.True(call.Request.Chat!.DirectAnswer);
        Assert.Equal("low", call.Request.Chat.ReasoningEffort);
        Assert.DoesNotContain("--direct-answer", call.Request.Arguments);
        call.Emit(new("response_start", PromptFormat: "chat", ThinkingOpen: false));
        call.Emit(new("output", "assistant", 0, "Direct answer", 1));
        await Drain();
        Assert.Equal("Direct answer", view.Turns[0].Response);
        call.Finish(Result("Direct answer")); await run;
        Assert.True(view.Turns[0].IsComplete);
        Assert.Equal(2, view.ContextMessages.Count);
        view.Prompt = "next";
        var next = view.SendAsync();
        Assert.False(core.Calls[1].Request.Chat!.DirectAnswer);
        Assert.Equal(3, core.Calls[1].Request.Chat!.Messages.Count);
        core.Calls[1].Finish(Result("partial", complete: false, exit: 2, stop: "max_tokens")); await next;
        Assert.Equal("next", view.Prompt);
        Assert.Equal(2, view.ContextMessages.Count);
    });

    [Fact]
    public Task OpenAiChatDoesNotRequireLocalWeightsAndDisablesNativeMemoryPlan() => OnDispatcher(async () =>
    {
        var (core, _, view) = Fixture();
        view.UseTokens = true; view.PromptFormat = "raw"; view.DirectAnswer = true;
        view.Backend = "openai"; view.ModelDirectory = ""; view.Prompt = "served model";
        Assert.False(view.UseTokens); Assert.Equal("chat", view.PromptFormat);
        Assert.False(view.DirectAnswer);
        Assert.False(view.PlanCommand.CanExecute(null));
        var run = view.SendAsync();
        var call = Assert.Single(core.Calls);
        Assert.Contains("openai", call.Request.Arguments);
        Assert.DoesNotContain("--model-directory", call.Request.Arguments);
        call.Finish(Result("server answer")); await run;
        Assert.Equal("server answer", view.Turns[0].Response);
    });

    [Fact]
    public Task JarvisMcpChoiceIsSnapshottedPerRunAndNativeSwitchClearsIt() => OnDispatcher(async () =>
    {
        var (core, _, view) = Fixture();
        view.Backend = "openai"; view.UseMcp = true; view.Prompt = "use tools";
        var first = view.SendAsync();
        var call = Assert.Single(core.Calls);
        view.UseMcp = false;
        Assert.True(call.Request.Chat!.UseMcp);
        call.Finish(Result("tool answer")); await first;
        view.UseMcp = true; view.Backend = "cpu";
        Assert.False(view.UseMcp); Assert.False(view.CanUseMcp);
        view.Prompt = "native";
        var second = view.SendAsync();
        Assert.False(core.Calls[1].Request.Chat!.UseMcp);
        Assert.Single(core.Calls[1].Request.Chat!.Messages);
        core.Calls[1].Finish(Result("native answer")); await second;
    });

    [Theory]
    [InlineData("endpoint")]
    [InlineData("model")]
    [InlineData("mcp")]
    public Task ChangingRuntimeIdentityStartsANewConversationContext(string change) => OnDispatcher(async () =>
    {
        var core = new FakeCore();
        var settings = new AppSettings { ProjectRoot = "D:\\synthetic", Profile = "nvfp4" };
        var task = new CoreTaskViewModel(core, () => settings);
        var view = new RunViewModel(task) { ModelDirectory = "E:\\shared-model", Prompt = "first" };
        var first = view.SendAsync(); core.Calls[0].Finish(Result("first answer")); await first;
        settings = change switch
        {
            "endpoint" => settings with { OpenAiBaseUrl = "http://127.0.0.1:9000/v1" },
            "model" => settings with { OpenAiModel = "different-served-model" },
            _ => settings with { JarvisMcpEndpoint = "http://127.0.0.1:9001/mcp" }
        };
        view.Prompt = "new runtime";
        var next = view.SendAsync();
        Assert.Single(core.Calls[1].Request.Chat!.Messages);
        core.Calls[1].Finish(Result("new answer")); await next;
    });

    [Fact]
    public Task IncompleteTurnKeepsDraftAndRetryDoesNotSendPartialAssistantAsHistory() => OnDispatcher(async () =>
    {
        var (core, _, view) = Fixture(); view.Prompt = "retry me";
        var first = view.SendAsync();
        core.Calls[0].Finish(Result("partial answer", complete: false, exit: 2, stop: "max_tokens"));
        await first;
        Assert.Equal("retry me", view.Prompt);
        Assert.False(view.Turns[0].IsComplete); Assert.Empty(view.ContextMessages);
        Assert.Contains("Hết token", view.Turns[0].State);
        view.Generate = "512";
        var retry = view.SendAsync();
        Assert.Single(core.Calls[1].Request.Chat!.Messages);
        Assert.Contains("512", core.Calls[1].Request.Arguments);
        core.Calls[1].Finish(Result("complete answer")); await retry;
        Assert.Equal(2, view.Turns.Count);
        Assert.Equal(2, view.ContextMessages.Count);
        Assert.Equal("complete answer", view.ContextMessages[1].Content);
        Assert.Equal("partial answer", RunViewModel.CopyText(view.Turns[0]));
    });

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    public Task ErrorCancelTimeoutNeverAddAClaimedCompleteResponseToHistory(bool cancelled, bool timedOut) => OnDispatcher(async () =>
    {
        var (core, _, view) = Fixture(); view.Prompt = "keep this";
        var run = view.SendAsync();
        core.Calls[0].Finish(Result("partial", complete: true, exit: cancelled ? 130 : timedOut ? 1 : 2) with { Cancelled = cancelled, TimedOut = timedOut });
        await run;
        Assert.Empty(view.ContextMessages); Assert.False(view.Turns[0].IsComplete);
        Assert.Equal("partial", view.Turns[0].Response); Assert.Equal("keep this", view.Prompt);
    });

    [Fact]
    public Task ResetAndNextRunIgnoreDelayedEventsFromAnOlderProcess() => OnDispatcher(async () =>
    {
        var (core, _, view) = Fixture(); view.Prompt = "old";
        var oldRun = view.SendAsync(); var old = core.Calls[0];
        view.ResetConversation();
        Assert.True(old.Token.IsCancellationRequested); Assert.Empty(view.Turns);
        old.Emit(new("response_start", PromptFormat: "chat", ThinkingOpen: true));
        old.Emit(new("output", "assistant", 0, "late old", 1));
        old.Finish(Result("old answer")); await oldRun; await Drain();
        Assert.Empty(view.ContextMessages); Assert.Empty(view.Turns);
        view.Prompt = "new";
        var newer = view.SendAsync();
        old.Emit(new("output", "assistant", 0, "late after new run", 2));
        await Drain();
        Assert.Empty(view.Turns[0].Response);
        core.Calls[1].Finish(Result("new answer")); await newer;
        Assert.Equal("new answer", view.Turns[0].Response);
        Assert.Equal("new", view.ContextMessages[0].Content);
    });

    [Fact]
    public Task InputEditedDuringGenerationIsNotClearedOrUsedRetroactively() => OnDispatcher(async () =>
    {
        var (core, _, view) = Fixture(); view.Prompt = "submitted";
        var run = view.SendAsync(); view.Prompt = "next draft";
        core.Calls[0].Finish(Result("answer")); await run;
        Assert.Equal("submitted", core.Calls[0].Request.Chat!.Messages[0].Content);
        Assert.Equal("submitted", view.ContextMessages[0].Content);
        Assert.Equal("next draft", view.Prompt);
    });

    [Fact]
    public Task ChangedModelDirectoryCannotAcquireTheOldRunsConversationContext() => OnDispatcher(async () =>
    {
        var (core, _, view) = Fixture(); view.Prompt = "first";
        var run = view.SendAsync(); view.ModelDirectory = "E:\\different-model";
        core.Calls[0].Finish(Result("old model answer")); await run;
        Assert.Empty(view.ContextMessages);
        view.Prompt = "new model";
        var next = view.SendAsync(); Assert.Single(core.Calls[1].Request.Chat!.Messages);
        core.Calls[1].Finish(Result("new model answer")); await next;
    });

    [Fact]
    public Task ProfileChangedDuringRunCannotAdoptItsOldHistoryEvenWithTheSameDirectory() => OnDispatcher(async () =>
    {
        var core = new FakeCore();
        var settings = new AppSettings { ProjectRoot = "D:\\synthetic", Profile = "nvfp4" };
        var task = new CoreTaskViewModel(core, () => settings);
        var view = new RunViewModel(task) { ModelDirectory = "E:\\shared-model", Prompt = "old profile" };
        var run = view.SendAsync(); settings = settings with { Profile = "fp8" };
        core.Calls[0].Finish(Result("old answer")); await run;
        Assert.Empty(view.ContextMessages);
    });

    [Fact]
    public Task RawTextAndTokenModesStaySeparateFromChatHistory() => OnDispatcher(async () =>
    {
        var (core, _, view) = Fixture(); view.PromptFormat = "raw"; view.Prompt = "raw input"; view.DirectAnswer = true;
        var raw = view.SendAsync(); Assert.Null(core.Calls[0].Request.Chat);
        Assert.Contains("--prompt", core.Calls[0].Request.Arguments);
        Assert.Contains("--stream-events", core.Calls[0].Request.Arguments);
        Assert.DoesNotContain("--direct-answer", core.Calls[0].Request.Arguments);
        core.Calls[0].Finish(Result("raw output")); await raw; Assert.Empty(view.ContextMessages);
        view.UseTokens = true; view.Tokens = "1,2,3";
        var token = view.SendAsync(); Assert.Null(core.Calls[1].Request.Chat);
        Assert.Contains("--tokens", core.Calls[1].Request.Arguments);
        Assert.DoesNotContain("--stream-events", core.Calls[1].Request.Arguments);
        Assert.DoesNotContain("--direct-answer", core.Calls[1].Request.Arguments);
        core.Calls[1].Finish(Result("unused") with { Generation = null }); await token;
        Assert.Empty(view.ContextMessages);
    });

    [Fact]
    public Task KeepThinkingIsExplicitAndCanIncludePreviouslyCompletedReasoning() => OnDispatcher(async () =>
    {
        var (core, _, view) = Fixture(); view.Prompt = "first";
        var first = view.SendAsync(); core.Calls[0].Finish(Result("answer", reasoning: "observed model reasoning")); await first;
        Assert.Null(view.ContextMessages[1].ReasoningContent);
        view.KeepThinking = true; view.Prompt = "next";
        var next = view.SendAsync();
        Assert.True(core.Calls[1].Request.Chat!.KeepThinking);
        Assert.Equal("observed model reasoning", core.Calls[1].Request.Chat!.Messages[1].ReasoningContent);
        core.Calls[1].Finish(Result("answer2")); await next;
    });

    [Fact]
    public Task EmptyOrTruncatedFinalResultCannotLookComplete() => OnDispatcher(async () =>
    {
        var (core, _, view) = Fixture(); view.Prompt = "first";
        var empty = view.SendAsync(); core.Calls[0].Finish(Result(" ")); await empty;
        Assert.False(view.Turns[0].IsComplete); Assert.Empty(view.ContextMessages); Assert.False(view.Turns[0].CanCopy);
        var next = view.SendAsync();
        core.Calls[1].Finish(Result("visible partial") with { Generation = Result("visible partial").Generation! with { OutputTruncated = true } });
        await next; Assert.False(view.Turns[1].IsComplete); Assert.Empty(view.ContextMessages);
        Assert.Contains("vượt giới hạn", view.Turns[1].State);
    });

    [Fact]
    public Task AlternateCoreCannotReportAnEmptyChatAsSuccessful() => OnDispatcher(async () =>
    {
        var (core, task, view) = Fixture(); view.Prompt = "hello";
        var empty = view.SendAsync(); core.Calls[0].Finish(Result("") with { Generation = null }); await empty;
        Assert.Equal(2, task.LastResult!.ExitCode);
        Assert.Contains("chưa hoàn tất", task.Status);
        Assert.Empty(view.ContextMessages);
        var valid = view.SendAsync();
        core.Calls[1].Emit(new("response_start", PromptFormat: "chat", ThinkingOpen: false));
        core.Calls[1].Emit(new("output", "assistant", 0, "stream only", 1));
        core.Calls[1].Emit(new("response_end", GeneratedTokens: 1, Status: "GENERATED_UNVERIFIED", StopReason: "eos_token", AssistantResponseComplete: true));
        core.Calls[1].Finish(Result("") with { Generation = null }); await valid;
        Assert.Equal(0, task.LastResult!.ExitCode);
        Assert.Equal("stream only", view.ContextMessages[1].Content);
    });

    [Fact]
    public Task RealExternalRuntimeCompletesTwoGuiTurnsAndSendsStructuredHistory() => OnDispatcher(async () =>
    {
        var fixture = Directory.CreateTempSubdirectory("ModelDesk-external-conversation-");
        try
        {
            Directory.CreateDirectory(Path.Combine(fixture.FullName, "glm_local"));
            await File.WriteAllTextAsync(Path.Combine(fixture.FullName, "glm_local", "__main__.py"), "# unexecuted profile-discovery marker");
            var profiles = Path.Combine(fixture.FullName, "config", "models"); Directory.CreateDirectory(profiles);
            await File.WriteAllTextAsync(Path.Combine(profiles, "flash-cybersecurity-w4a16.json"), JsonSerializer.Serialize(new
            {
                model_id = "example/flash", revision = new string('a', 40), model_directory = "weights",
                report_namespace = "flash-w4a16"
            }));
            var sent = new List<JsonElement>();
            using var http = new HttpClient(new ConversationHttpHandler(async request =>
            {
                using var body = JsonDocument.Parse(await request.Content!.ReadAsStringAsync());
                sent.Add(body.RootElement.Clone());
                var answer = sent.Count == 1 ? "first external answer" : "second external answer";
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent(JsonSerializer.Serialize(new
                    {
                        choices = new[] { new { message = new { role = "assistant", content = answer }, finish_reason = "stop" } },
                        usage = new { completion_tokens = 4 }
                    }), Encoding.UTF8, "application/json")
                };
            }));
            var settings = new AppSettings { ProjectRoot = fixture.FullName, Profile = "flash-w4a16" };
            var runtime = new ModelRuntimeService(new PythonCoreService(), http, environmentReader: _ => null);
            var task = new CoreTaskViewModel(runtime, () => settings);
            var view = new RunViewModel(task) { Backend = "openai", ModelDirectory = "", Prompt = "first question" };
            await view.SendAsync();
            Assert.True(view.Turns[0].IsComplete); Assert.Equal(2, view.ContextMessages.Count);
            Assert.Equal("first external answer", view.ContextMessages[1].Content);
            view.Prompt = "second question";
            await view.SendAsync();
            Assert.Equal(2, sent.Count); Assert.True(view.Turns[1].IsComplete);
            var history = sent[1].GetProperty("messages").EnumerateArray().ToArray();
            Assert.Equal(["user", "assistant", "user"], history.Select(message => message.GetProperty("role").GetString()!).ToArray());
            Assert.Equal(["first question", "first external answer", "second question"], history.Select(message => message.GetProperty("content").GetString()!).ToArray());
            Assert.Equal(4, view.ContextMessages.Count);
        }
        finally
        {
            var expected = Path.TrimEndingDirectorySeparator(Path.GetFullPath(Path.GetTempPath())) + Path.DirectorySeparatorChar;
            if (!Path.GetFullPath(fixture.FullName).StartsWith(expected, StringComparison.OrdinalIgnoreCase)) throw new IOException("Conversation test cleanup escaped its temporary root.");
            fixture.Delete(true);
        }
    });

    private static (FakeCore Core, CoreTaskViewModel Task, RunViewModel View) Fixture()
    {
        var core = new FakeCore();
        var task = new CoreTaskViewModel(core, () => new AppSettings { ProjectRoot = "D:\\synthetic-project", Profile = "nvfp4" });
        return (core, task, new RunViewModel(task) { ModelDirectory = "E:\\synthetic-model" });
    }
    private static CoreRunResult Result(string text, string reasoning = "", bool complete = true, int exit = 0, string stop = "eos_token") =>
        new(Guid.NewGuid().ToString("N"), exit, false, "output.log", null, DateTimeOffset.UtcNow, DateTimeOffset.UtcNow,
            Generation: new(text, reasoning, stop, complete, 8, complete ? "GENERATED_UNVERIFIED" : "INCOMPLETE_RESPONSE"));
    private static Task Drain() => Dispatcher.CurrentDispatcher.InvokeAsync(() => { }, DispatcherPriority.ApplicationIdle).Task;
    private static async Task OnDispatcher(Func<Task> test)
    {
        var done = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var thread = new Thread(() =>
        {
            var dispatcher = Dispatcher.CurrentDispatcher;
            SynchronizationContext.SetSynchronizationContext(new DispatcherSynchronizationContext(dispatcher));
            dispatcher.BeginInvoke(async () =>
            {
                try { await test(); done.TrySetResult(); }
                catch (Exception error) { done.TrySetException(error); }
                finally { dispatcher.BeginInvokeShutdown(DispatcherPriority.Background); }
            });
            Dispatcher.Run();
        }) { IsBackground = true };
        thread.SetApartmentState(ApartmentState.STA); thread.Start();
        try { await done.Task.WaitAsync(TimeSpan.FromSeconds(15)); }
        finally { Assert.True(thread.Join(TimeSpan.FromSeconds(3))); }
    }
    private sealed class FakeCore : IPythonCoreService
    {
        public List<Call> Calls { get; } = [];
        public IReadOnlyList<ModelProfile> GetProfiles(string projectRoot) => [];
        public Task<CoreRunResult> RunAsync(CoreRunRequest request, IProgress<CoreOutput>? output = null, CancellationToken cancellationToken = default)
        { var call = new Call(request, output!, cancellationToken); Calls.Add(call); return call.Completion.Task; }
    }
    private sealed record Call(CoreRunRequest Request, IProgress<CoreOutput> Progress, CancellationToken Token)
    {
        internal TaskCompletionSource<CoreRunResult> Completion { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        internal void Emit(ModelStreamEvent value) => Progress.Report(new(DateTimeOffset.UtcNow, "structured event", StreamEvent: value));
        internal void Finish(CoreRunResult result) => Completion.SetResult(result);
    }
    private sealed class ConversationHttpHandler(Func<HttpRequestMessage, Task<HttpResponseMessage>> send) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) => send(request);
    }
}
