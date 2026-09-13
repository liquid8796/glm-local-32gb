using System.Text.Json;
using ModelDesk.Core;
using ModelDesk.Infrastructure.Runtime;

namespace ModelDesk.Tests;

public sealed class PythonChatServiceTests : IDisposable
{
    private readonly string root = Path.Combine(Path.GetTempPath(), "modeldesk-chat-" + Guid.NewGuid().ToString("N"));
    private readonly PythonCoreService service = new();
    private readonly AppSettings settings;
    public PythonChatServiceTests()
    {
        Directory.CreateDirectory(Path.Combine(root, "glm_local"));
        Directory.CreateDirectory(Path.Combine(root, "config", "models"));
        File.WriteAllText(Path.Combine(root, "glm_local", "__init__.py"), "");
        File.WriteAllText(Path.Combine(root, "glm_local", "__main__.py"), Fake);
        File.WriteAllText(Path.Combine(root, "config", "models", "abliterated-nvfp4.json"),
            "{\"model_id\":\"example/chat\",\"revision\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"model_directory\":\"models/fake\",\"report_namespace\":\"nvfp4\"}");
        settings = new AppSettings { ProjectRoot = root, PythonExecutable = "python" };
    }
    public void Dispose() => Directory.Delete(root, recursive: true);
    private CoreRunRequest Request(params string[] args) => new(settings, "generate", args,
        Chat: new ChatRunOptions([new("user", "chào bạn 😀 ' \"quoted\" & | $()")], "low"));

    [Fact]
    public async Task ServiceOwnsFreshMessageFilesAndFreezesMutableCallerHistory()
    {
        var history = new List<ChatMessage> { new("user", "before") };
        var request = Request() with { Chat = new ChatRunOptions(history) };
        var running = service.RunAsync(request);
        history[0] = new("user", "after");
        var first = await running;
        var second = await service.RunAsync(Request());
        Assert.Equal(0, first.ExitCode);
        Assert.NotEqual(first.RunId, second.RunId);
        var firstMessages = Path.Combine(Path.GetDirectoryName(first.LogPath)!, "messages.json");
        var secondMessages = Path.Combine(Path.GetDirectoryName(second.LogPath)!, "messages.json");
        using var body = JsonDocument.Parse(await File.ReadAllTextAsync(firstMessages));
        Assert.Equal("before", body.RootElement[0].GetProperty("content").GetString());
        Assert.True(File.Exists(secondMessages));
        using var report = JsonDocument.Parse(await File.ReadAllTextAsync(first.ReportPath!));
        var arguments = report.RootElement.GetProperty("arguments").EnumerateArray().Select(item => item.GetString()).ToArray();
        Assert.Contains(firstMessages, arguments);
        Assert.Contains("chat", arguments);
        Assert.Contains("low", arguments);
        Assert.Contains("--stream-events", arguments);
        Assert.DoesNotContain("before", arguments);
    }

    [Fact]
    public async Task SplitUtf8StreamsProduceTypedEditsAndFreshReportReconcilesFinalText()
    {
        var events = new List<ModelStreamEvent>();
        var result = await service.RunAsync(Request(), new InlineProgress<CoreOutput>(item => { if (item.StreamEvent is not null) events.Add(item.StreamEvent); }));
        Assert.Equal(0, result.ExitCode);
        Assert.Contains(events, item => item.Event == "output" && item.ReplaceFromUtf16 == 3 && item.Text == "é");
        Assert.Equal("A😀é final", result.Generation?.Text);
        Assert.Equal("Suy nghĩ", result.Generation?.Reasoning);
        Assert.True(result.Generation?.AssistantResponseComplete);
        Assert.Equal(4, result.Generation?.GeneratedTokens);
    }

    [Theory]
    [InlineData(false, "low")]
    [InlineData(true, "low")]
    [InlineData(true, "high")]
    [InlineData(true, "max")]
    public async Task DirectAnswerIsForwardedOnlyByExplicitChatOption(bool directAnswer, string effort)
    {
        var events = new List<ModelStreamEvent>();
        var request = Request();
        request = request with { Chat = request.Chat! with { DirectAnswer = directAnswer, ReasoningEffort = effort } };
        var result = await service.RunAsync(request, new InlineProgress<CoreOutput>(item =>
        { if (item.StreamEvent is not null) events.Add(item.StreamEvent); }));
        Assert.Equal(0, result.ExitCode);
        using var report = JsonDocument.Parse(await File.ReadAllTextAsync(result.ReportPath!));
        var arguments = report.RootElement.GetProperty("arguments").EnumerateArray().Select(item => item.GetString()).ToArray();
        Assert.Equal(directAnswer ? 1 : 0, arguments.Count(item => item == "--direct-answer"));
        Assert.Equal(effort, arguments[Array.IndexOf(arguments, "--reasoning-effort") + 1]);
        var start = Assert.Single(events, item => item.Event == "response_start");
        Assert.Equal("chat", start.PromptFormat);
        Assert.Equal(!directAnswer, start.ThinkingOpen);
        Assert.True(result.Generation?.AssistantResponseComplete);
        // A model may reopen reasoning even after the direct-answer prefix.
        Assert.Equal("Suy nghĩ", result.Generation?.Reasoning);
    }

    [Theory]
    [InlineData("--partial")]
    [InlineData("--empty-success")]
    [InlineData("--malformed-event")]
    public async Task PartialOrEmptyResponsesNeverReportSuccessfulChat(string mode)
    {
        var result = await service.RunAsync(Request(mode));
        Assert.Equal(2, result.ExitCode);
        Assert.NotNull(result.Generation);
        Assert.False(result.Generation.AssistantResponseComplete);
        if (mode == "--partial")
        {
            Assert.Equal("Chưa xong", result.Generation.Text);
            Assert.Equal("max_tokens", result.Generation.StopReason);
        }
    }

    [Fact]
    public async Task TimeoutRecoversPartialResponseWithoutPromotingCompletion()
    {
        var result = await service.RunAsync(Request("--timeout"));
        Assert.Equal(1, result.ExitCode);
        Assert.True(result.TimedOut);
        Assert.Equal("Phần đang viết", result.Generation?.Text);
        Assert.Equal("timeout", result.Generation?.StopReason);
        Assert.False(result.Generation?.AssistantResponseComplete);
    }

    [Fact]
    public async Task SameModelFreshReportFromAnotherMessagesFileIsRejected()
    {
        var result = await service.RunAsync(Request("--foreign-request"));
        Assert.Null(result.ReportPath);
        Assert.Equal("own partial", result.Generation?.Text);
        Assert.DoesNotContain("foreign reply", result.Generation?.Text ?? "");
        Assert.Equal(2, result.ExitCode);
    }

    [Theory]
    [InlineData("--stale-timeout")]
    [InlineData("--same-count-timeout")]
    public async Task OlderRecoveredCheckpointCannotEraseNewerStreamedPartialText(string mode)
    {
        var result = await service.RunAsync(Request(mode));
        Assert.True(result.TimedOut);
        Assert.Equal(1, result.ExitCode);
        Assert.Equal("A😀é", result.Generation?.Text);
        Assert.Equal("Suy nghĩ", result.Generation?.Reasoning);
        Assert.Equal(3, result.Generation?.GeneratedTokens);
        Assert.Equal("ERROR", result.Generation?.Status);
        Assert.False(result.Generation?.AssistantResponseComplete);
    }

    [Fact]
    public async Task ErrorBeforeFirstTokenHasTerminalStatusInsteadOfRunning()
    {
        var result = await service.RunAsync(Request("--error-before-token"));
        Assert.Equal(1, result.ExitCode);
        Assert.Equal("ERROR", result.Generation?.Status);
        Assert.False(result.Generation?.AssistantResponseComplete);
        Assert.Equal("", result.Generation?.Text);
    }

    [Theory]
    [InlineData("--prompt")]
    [InlineData("--prompt-format=raw")]
    [InlineData("--messages-file")]
    [InlineData("--reasoning-effort")]
    [InlineData("--direct-answer")]
    [InlineData("--direct-answer=true")]
    public async Task CompetingInputFlagsAreRejectedBeforeCreatingRunArtifacts(string flag)
    {
        await Assert.ThrowsAsync<ArgumentException>(() => service.RunAsync(Request(flag, "value")));
        Assert.False(Directory.Exists(Path.Combine(root, "reports")));
    }

    private const string Fake = """
        import json,os,pathlib,sys
        config_path=pathlib.Path(sys.argv[sys.argv.index('--config')+1])
        config=json.loads(config_path.read_text(encoding='utf-8-sig'))
        args=sys.argv[sys.argv.index('--config')+2:]
        messages=pathlib.Path(args[args.index('--messages-file')+1])
        body=json.loads(messages.read_text(encoding='utf-8-sig'))
        directory=pathlib.Path.cwd()/'reports'/'nvfp4'/'generate'/config_path.parent.name
        directory.mkdir(parents=True)
        (directory/'request.json').write_text(json.dumps({'parameters':{'messages_file':str(messages)+('.other' if '--foreign-request' in args else '')}}),encoding='utf-8')
        def emit(value):
            raw=('MODELDESK_EVENT '+json.dumps(value,ensure_ascii=False)+'\n').encode('utf-8')
            for offset in range(0,len(raw),7):os.write(1,raw[offset:offset+7])
        emit({'event':'response_start','prompt_format':'chat','thinking_open':'--direct-answer' not in args})
        text='A😀é final';reasoning='Suy nghĩ';complete=True;status='GENERATED_UNVERIFIED';stop='eos_token';code=0
        if '--empty-success' not in args and '--error-before-token' not in args:
            emit({'event':'output','channel':'reasoning','replace_from_utf16':0,'text':reasoning,'generated_tokens':1})
            if '--partial' in args:text='Chưa xong'
            if '--timeout' in args:text='Phần đang viết'
            if '--foreign-request' in args:text='own partial'
            shown='A😀�' if text=='A😀é final' else text
            emit({'event':'output','channel':'assistant','replace_from_utf16':0,'text':shown,'generated_tokens':2})
            if text=='A😀é final':emit({'event':'output','channel':'assistant','replace_from_utf16':3,'text':'é','generated_tokens':3})
        else:text='';reasoning=''
        if '--partial' in args:complete=False;status='INCOMPLETE_RESPONSE';stop='max_tokens';code=2
        if '--timeout' in args:complete=False;status='ERROR';stop='timeout';code=1
        if '--stale-timeout' in args or '--same-count-timeout' in args:complete=False;status='ERROR';stop='timeout';code=1
        if '--error-before-token' in args:complete=False;status='ERROR';stop='error';code=1
        if '--malformed-event' in args:os.write(1,b'MODELDESK_EVENT {invalid}\n')
        if not any(flag in args for flag in ('--timeout','--foreign-request','--stale-timeout','--same-count-timeout','--error-before-token')):
            emit({'event':'response_end','generated_tokens':4,'status':status,'stop_reason':stop,'assistant_response_complete':complete})
        if '--malformed-event' not in args:
            report={'model_id':config['model_id'],'revision':config['revision'],'action':'generate','run_directory':str(directory),
                    'text':'foreign reply' if '--foreign-request' in args else text,'reasoning':reasoning,'assistant_response_complete':complete,
                    'status':status,'stop_reason':stop,'generated_token_ids':[1,2,3,4],'arguments':args,'messages':body,
                    'timed_out':'--timeout' in args,'error_type':'TIMEOUT' if '--timeout' in args else None}
            if '--stale-timeout' in args or '--same-count-timeout' in args:
                report.update(text='A',reasoning='old',generated_tokens=1 if '--stale-timeout' in args else 3,
                              partial_response_recovered=True,timed_out=True,error_type='TIMEOUT')
            if '--error-before-token' in args:
                report.pop('text');report.pop('reasoning')
            path=directory/'result.json';path.write_text(json.dumps(report,ensure_ascii=False),encoding='utf-8')
            print('Report: '+str(path),flush=True)
        sys.exit(code)
        """;
}
