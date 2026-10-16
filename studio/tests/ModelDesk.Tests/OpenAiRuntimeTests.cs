using System.Net;
using System.Text;
using System.Text.Json;
using ModelDesk.Core;
using ModelDesk.Infrastructure.Runtime;
using ModelDesk.Infrastructure.Reports;

namespace ModelDesk.Tests;

public sealed class OpenAiRuntimeTests : IDisposable
{
    private readonly string root = Path.Combine(Path.GetTempPath(), "modeldesk-openai-" + Guid.NewGuid().ToString("N"));
    public void Dispose() { if (Directory.Exists(root)) Directory.Delete(root, true); }
    private AppSettings Settings => new() { ProjectRoot = root, Profile = "flash-w4a16", OpenAiBaseUrl = "http://localhost:8000/v1" };
    private CoreRunRequest Request(params string[] args) => new(Settings, "generate", args,
        Chat: new ChatRunOptions([new("user", "Xin chào")], "low"));

    [Fact]
    public async Task FlashUsesConfiguredEndpointModelAndTemplateOptionsWithoutStartingNative()
    {
        JsonElement sent = default;
        using var http = new HttpClient(new Handler(async request =>
        {
            Assert.Equal("http://localhost:8000/v1/chat/completions", request.RequestUri!.AbsoluteUri);
            Assert.Equal("test-secret", request.Headers.Authorization?.Parameter);
            sent = JsonDocument.Parse(await request.Content!.ReadAsStringAsync()).RootElement.Clone();
            return Reply("{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"Xin chào!\",\"reasoning\":\"Một suy nghĩ\"},\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":12}}");
        }));
        var native = new Native();
        var service = new ModelRuntimeService(native, http, environmentReader: _ => "test-secret");
        var events = new List<ModelStreamEvent>();
        var result = await service.RunAsync(Request("--generate", "2000", "--context", "8192", "--timeout", "30") with
        { Settings = Settings with { OpenAiModel = "served-flash" } }, new ProgressSink(item => { if (item.StreamEvent is { } value) events.Add(value); }));
        Assert.Equal(0, native.Runs);
        Assert.Equal("served-flash", sent.GetProperty("model").GetString());
        Assert.Equal("Xin chào", sent.GetProperty("messages")[0].GetProperty("content").GetString());
        Assert.Equal(2000, sent.GetProperty("max_tokens").GetInt32());
        var template = sent.GetProperty("chat_template_kwargs");
        Assert.Equal("low", template.GetProperty("reasoning_effort").GetString());
        Assert.True(template.GetProperty("clear_thinking").GetBoolean());
        Assert.False(template.TryGetProperty("enable_thinking", out _));
        Assert.Equal("Xin chào!", result.Generation?.Text);
        Assert.Equal("Một suy nghĩ", result.Generation?.Reasoning);
        Assert.True(result.Generation?.AssistantResponseComplete);
        var accumulated = new GenerationAccumulator();
        foreach (var value in events) Assert.True(accumulated.Apply(value));
        accumulated.Reconcile(result.Generation!);
        Assert.True(accumulated.Snapshot().AssistantResponseComplete);
        Assert.Contains(events, item => item.Event == "response_end");
        Assert.DoesNotContain("test-secret", await File.ReadAllTextAsync(result.ReportPath!));
        Assert.DoesNotContain("test-secret", await File.ReadAllTextAsync(result.LogPath));
        Assert.Contains(new ReportService().List(root, Path.Combine(root, "reports", "flash-w4a16")), item => item.Path == result.ReportPath);
    }

    [Theory]
    [InlineData("length", "partial", 2)]
    [InlineData("stop", "", 2)]
    [InlineData("stop", "complete", 0)]
    public async Task EmptyAndTokenLimitedRepliesNeverBecomeCompletedHistory(string finish, string content, int exit)
    {
        using var http = new HttpClient(new Handler(_ => Task.FromResult(Reply(JsonSerializer.Serialize(new
        { choices = new[] { new { message = new { role = "assistant", content }, finish_reason = finish } } })))));
        var result = await new ModelRuntimeService(new Native(), http).RunAsync(Request());
        Assert.Equal(exit, result.ExitCode);
        Assert.Equal(exit == 0, result.Generation!.AssistantResponseComplete);
    }

    [Fact]
    public async Task ExistingNativeProfilesStillDelegateAndMcpCannotSilentlyBeIgnored()
    {
        using var http = new HttpClient(new Handler(_ => throw new Xunit.Sdk.XunitException("Unexpected external HTTP")));
        var native = new Native(); var service = new ModelRuntimeService(native, http);
        await service.RunAsync(Request() with { Settings = Settings with { Profile = "nvfp4" } });
        Assert.Equal(1, native.Runs);
        await Assert.ThrowsAsync<ArgumentException>(() => service.RunAsync(Request() with
        { Settings = Settings with { Profile = "nvfp4" }, Chat = Request().Chat! with { UseMcp = true } }));
        Assert.Equal(1, native.Runs);
    }

    [Fact]
    public async Task ExplicitNativeConfigOverridesSavedFlashAndMetadataDiagnosticsRemainAvailable()
    {
        using var http = new HttpClient(new Handler(_ => throw new Xunit.Sdk.XunitException("Unexpected external HTTP")));
        var native = new Native(); var service = new ModelRuntimeService(native, http);
        await service.RunAsync(new(Settings, "doctor", [], "config/models/abliterated-nvfp4.json"));
        await service.RunAsync(new(Settings, "metadata-check", []));
        Assert.Equal(2, native.Runs);
    }

    [Fact]
    public void InvalidMcpUrlFailsWhenSavingSettings()
    {
        Assert.Throws<ArgumentException>(() => new AppSettings { JarvisMcpEndpoint = "http://remote.example/mcp" }.Validate());
        new AppSettings { JarvisMcpEndpoint = "http://localhost:5000/mcp" }.Validate();
    }

    [Fact]
    public async Task CancelledRequestProducesIncompleteReceiptAndDoesNotRetry()
    {
        var requests = 0;
        using var cancellation = new CancellationTokenSource();
        using var http = new HttpClient(new Handler(async _ => { requests++; cancellation.Cancel(); await Task.Yield(); throw new OperationCanceledException(cancellation.Token); }));
        var result = await new ModelRuntimeService(new Native(), http).RunAsync(Request(), cancellationToken: cancellation.Token);
        Assert.True(result.Cancelled);
        Assert.False(result.Generation!.AssistantResponseComplete);
        Assert.Equal(1, requests);
        Assert.True(File.Exists(result.ReportPath));
    }

    [Fact]
    public async Task HttpFailureBodyIsNotLoggedOrRetried()
    {
        var requests = 0;
        using var http = new HttpClient(new Handler(_ => { requests++; return Task.FromResult(new HttpResponseMessage(HttpStatusCode.Unauthorized)
        { Content = new StringContent("server-private-token") }); }));
        var result = await new ModelRuntimeService(new Native(), http).RunAsync(Request());
        Assert.Equal(1, result.ExitCode); Assert.Equal(1, requests);
        Assert.DoesNotContain("server-private-token", await File.ReadAllTextAsync(result.LogPath));
    }

    internal static HttpResponseMessage Reply(string json) => new(HttpStatusCode.OK) { Content = new StringContent(json, Encoding.UTF8, "application/json") };
    private sealed class Handler(Func<HttpRequestMessage, Task<HttpResponseMessage>> send) : HttpMessageHandler
    { protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) => send(request); }
    private sealed class ProgressSink(Action<CoreOutput> report) : IProgress<CoreOutput> { public void Report(CoreOutput value) => report(value); }
    private sealed class Native : IPythonCoreService
    {
        public int Runs;
        public IReadOnlyList<ModelProfile> GetProfiles(string projectRoot) => [new("flash-w4a16", "Flash", "dealignai/GLM-5.3-Flash-CYBERSECURITY-W4A16", new string('a',40), "", "", Path.Combine(projectRoot, "reports", "flash-w4a16"))];
        public Task<CoreRunResult> RunAsync(CoreRunRequest request, IProgress<CoreOutput>? output = null, CancellationToken cancellationToken = default)
        { Runs++; return Task.FromResult(new CoreRunResult("native", 0, false, "", null, DateTimeOffset.UtcNow, DateTimeOffset.UtcNow)); }
    }
}
