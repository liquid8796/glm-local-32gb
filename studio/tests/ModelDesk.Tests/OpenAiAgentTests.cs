using System.Net;
using System.Text;
using System.Text.Json;
using ModelDesk.Core;
using ModelDesk.Infrastructure.Runtime;

namespace ModelDesk.Tests;

public sealed class OpenAiAgentTests : IDisposable
{
    private readonly string root = Path.Combine(Path.GetTempPath(), "modeldesk-agent-" + Guid.NewGuid().ToString("N"));
    public void Dispose() { if (Directory.Exists(root)) Directory.Delete(root, true); }
    private CoreRunRequest Request => new(new AppSettings { ProjectRoot = root, Profile = "flash-w4a16", JarvisMcpEndpoint = "https://jarvis.example/mcp" },
        "generate", ["--generate", "2000"], Chat: new ChatRunOptions([new("user", "Inspect my PC")], UseMcp: true));
    private static HttpResponseMessage Reply(object payload) => new(HttpStatusCode.OK)
    { Content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json") };
    private static object Calls(params object[] calls) => new { choices = new[] { new { message = new { role = "assistant", content = (string?)null, tool_calls = calls }, finish_reason = "tool_calls" } } };
    private static object Call(string id = "call-1", string name = "jarvis_tool_0", string args = "{\"action\":\"get_window_state\"}") =>
        new { id, type = "function", function = new { name, arguments = args } };
    private static object Final(string text = "PC observed") => new { choices = new[] { new { message = new { role = "assistant", content = text }, finish_reason = "stop" } } };

    [Fact]
    public async Task DiscoveredSchemaAndToolResultsReachModelWithImageAndObservationIds()
    {
        var turn = 0; var mcp = new Mcp();
        using var http = new HttpClient(new Handler(async request =>
        {
            using var body = JsonDocument.Parse(await request.Content!.ReadAsStringAsync());
            var json = body.RootElement;
            Assert.False(json.GetProperty("parallel_tool_calls").GetBoolean());
            var function = json.GetProperty("tools")[0].GetProperty("function");
            Assert.Equal("jarvis_tool_0", function.GetProperty("name").GetString());
            Assert.Equal("action", function.GetProperty("parameters").GetProperty("required")[0].GetString());
            if (turn++ == 0) return Reply(Calls(Call()));
            var messages = json.GetProperty("messages");
            var tool = messages.EnumerateArray().Single(item => item.GetProperty("role").GetString() == "tool");
            Assert.Equal("call-1", tool.GetProperty("tool_call_id").GetString());
            using var result = JsonDocument.Parse(tool.GetProperty("content").GetString()!);
            Assert.Equal("observation-7", result.RootElement.GetProperty("structuredContent").GetProperty("observation_id").GetString());
            Assert.DoesNotContain("aGVsbG8=", tool.GetProperty("content").GetString());
            var images = messages[messages.GetArrayLength() - 1].GetProperty("content");
            Assert.Equal("data:image/png;base64,aGVsbG8=", images[1].GetProperty("image_url").GetProperty("url").GetString());
            return Reply(Final());
        }));
        var result = await new ModelRuntimeService(new Native(root), http, _ => mcp).RunAsync(Request);
        Assert.Equal(0, result.ExitCode); Assert.True(result.Generation!.AssistantResponseComplete);
        Assert.Equal("computer_use", mcp.Calls.Single().Name);
        Assert.Equal("get_window_state", mcp.Calls[0].Args.GetProperty("action").GetString());
        Assert.True(mcp.Disposed);
    }

    [Fact]
    public async Task InvalidCallInBatchPreventsAllActionsAndNoCallIsRetried()
    {
        var mcp = new Mcp(); var requests = 0;
        using var http = new HttpClient(new Handler(_ => { requests++; return Task.FromResult(Reply(Calls(Call("call-2", "unadvertised")))); }));
        var result = await new ModelRuntimeService(new Native(root), http, _ => mcp).RunAsync(Request);
        Assert.Equal(1, result.ExitCode); Assert.Empty(mcp.Calls); Assert.Equal(1, requests);
    }

    [Fact]
    public async Task RepeatedCallIdCannotExecuteActionTwice()
    {
        var mcp = new Mcp(); var requests = 0;
        using var http = new HttpClient(new Handler(_ => { requests++; return Task.FromResult(Reply(Calls(Call()))); }));
        var result = await new ModelRuntimeService(new Native(root), http, _ => mcp).RunAsync(Request);
        Assert.Equal(1, result.ExitCode); Assert.Single(mcp.Calls); Assert.Equal(2, requests);
    }

    [Fact]
    public async Task MultiplePcActionsInOneCompletionRequireANewObservationRound()
    {
        var mcp = new Mcp();
        using var http = new HttpClient(new Handler(_ => Task.FromResult(Reply(Calls(Call("first"), Call("second"))))));
        var result = await new ModelRuntimeService(new Native(root), http, _ => mcp).RunAsync(Request);
        Assert.Equal(2, result.ExitCode); Assert.Empty(mcp.Calls);
        Assert.False(result.Generation!.AssistantResponseComplete);
    }

    [Fact]
    public async Task ToolErrorIsReturnedAsDataSoModelCanExplainAgentState()
    {
        var mcp = new Mcp { IsError = true }; var requests = 0;
        using var http = new HttpClient(new Handler(async request =>
        {
            if (requests++ == 0) return Reply(Calls(Call()));
            using var body = JsonDocument.Parse(await request.Content!.ReadAsStringAsync());
            var tool = body.RootElement.GetProperty("messages").EnumerateArray().Single(item => item.GetProperty("role").GetString() == "tool");
            using var result = JsonDocument.Parse(tool.GetProperty("content").GetString()!);
            Assert.True(result.RootElement.GetProperty("isError").GetBoolean());
            return Reply(Final("Arm the connected Agent to continue."));
        }));
        var result = await new ModelRuntimeService(new Native(root), http, _ => mcp).RunAsync(Request);
        Assert.Equal(0, result.ExitCode); Assert.Single(mcp.Calls);
        Assert.Equal("Arm the connected Agent to continue.", result.Generation!.Text);
    }

    [Fact]
    public async Task ToolLoopIsBoundedAndFinalNoToolsRequestCannotExecuteAnotherAction()
    {
        var mcp = new Mcp(); var requests = 0;
        using var http = new HttpClient(new Handler(async request =>
        {
            using var body = JsonDocument.Parse(await request.Content!.ReadAsStringAsync());
            if (requests == 12) Assert.Equal("none", body.RootElement.GetProperty("tool_choice").GetString());
            requests++; return Reply(Calls(Call("call-" + requests)));
        }));
        var result = await new ModelRuntimeService(new Native(root), http, _ => mcp).RunAsync(Request);
        Assert.Equal(2, result.ExitCode); Assert.Equal(12, mcp.Calls.Count); Assert.Equal(13, requests);
        Assert.False(result.Generation!.AssistantResponseComplete);
    }

    [Fact]
    public async Task ToolsAreNeverConnectedForOrdinaryChat()
    {
        using var http = new HttpClient(new Handler(_ => Task.FromResult(Reply(Final()))));
        var result = await new ModelRuntimeService(new Native(root), http, _ => throw new Xunit.Sdk.XunitException("MCP unexpectedly connected"))
            .RunAsync(Request with { Chat = Request.Chat! with { UseMcp = false } });
        Assert.Equal(0, result.ExitCode);
    }

    private sealed class Handler(Func<HttpRequestMessage, Task<HttpResponseMessage>> send) : HttpMessageHandler
    { protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) => send(request); }
    private sealed class Native(string root) : IPythonCoreService
    {
        public IReadOnlyList<ModelProfile> GetProfiles(string projectRoot) => [new("flash-w4a16", "Flash", "dealignai/GLM-5.3-Flash-CYBERSECURITY-W4A16", new string('a',40), "", "", Path.Combine(root,"reports"))];
        public Task<CoreRunResult> RunAsync(CoreRunRequest request, IProgress<CoreOutput>? output = null, CancellationToken cancellationToken = default) => throw new Xunit.Sdk.XunitException("Unexpected native inference");
    }
    private sealed class Mcp : IMcpToolClient
    {
        public List<(string Name, JsonElement Args)> Calls { get; } = [];
        public bool Disposed; public bool IsError;
        public Task ConnectAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
        public Task<IReadOnlyList<McpToolDescriptor>> ListToolsAsync(CancellationToken cancellationToken = default)
            => Task.FromResult<IReadOnlyList<McpToolDescriptor>>([new("computer_use", "Observe/input", JsonSerializer.SerializeToElement(new
            { type = "object", properties = new { action = new { type = "string" } }, required = new[] { "action" } }))]);
        public Task<McpToolResult> CallToolAsync(string name, JsonElement arguments, CancellationToken cancellationToken = default)
        {
            Calls.Add((name, arguments.Clone()));
            var raw = JsonSerializer.SerializeToElement(new { isError = IsError, content = new object[]
            { new { type = "text", text = "observation-7 screenshot-8" }, new { type = "image", mimeType = "image/png", data = "aGVsbG8=" } },
                structuredContent = new { observation_id = "observation-7", screenshot_id = "screenshot-8" } });
            return Task.FromResult(new McpToolResult(IsError, raw.GetProperty("content"), raw.GetProperty("structuredContent"), raw));
        }
        public ValueTask DisposeAsync() { Disposed = true; return ValueTask.CompletedTask; }
    }
}
