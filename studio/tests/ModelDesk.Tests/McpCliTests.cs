using System.Net;
using System.Text;
using System.Text.Json;
using ModelDesk.Cli;
using ModelDesk.Core;
using ModelDesk.Infrastructure.Mcp;

namespace ModelDesk.Tests;

public sealed class McpCliTests
{
    [Fact]
    public async Task DiscoveryPrintsLiveSchemasAndDisposesTheConnection()
    {
        var test = new Fixtures();
        Assert.Equal(0, await test.App.RunAsync(["--json", "mcp", "tools"]));
        using var json = JsonDocument.Parse(test.Out.ToString());
        var tool = Assert.Single(json.RootElement.EnumerateArray());
        Assert.Equal("computer_use", tool.GetProperty("Name").GetString());
        Assert.True(tool.GetProperty("InputSchema").GetProperty("properties").TryGetProperty("observation_id", out _));
        Assert.Equal(1, test.Mcp.Connects);
        Assert.Equal(1, test.Mcp.Lists);
        Assert.True(test.Mcp.Disposed);
        Assert.Equal(0, test.Mcp.Calls);
        Assert.Equal("https://example.test/mcp", test.ConnectionSettings!.JarvisMcpEndpoint);
        Assert.Equal("MCP_TEST_ACCESS_ENV", test.ConnectionSettings.JarvisMcpTokenEnvironmentVariable);
        Assert.Equal("", test.Error.ToString());
    }

    [Theory]
    [InlineData(false, 0)]
    [InlineData(true, 2)]
    public async Task DirectCallReadsObjectFileAndPreservesRawToolResult(bool isError, int exitCode)
    {
        using var file = new ArgumentFile("{\"action\":\"get_window_state\",\"include_text\":true}");
        var test = new Fixtures();
        test.Mcp.IsError = isError;
        Assert.Equal(exitCode, await test.App.RunAsync(["mcp", "call", "computer_use", "--arguments", file.Path, "--json"]));
        Assert.Equal("computer_use", test.Mcp.ToolName);
        Assert.Equal("get_window_state", test.Mcp.Arguments.GetProperty("action").GetString());
        Assert.True(test.Mcp.Arguments.GetProperty("include_text").GetBoolean());
        using var json = JsonDocument.Parse(test.Out.ToString());
        Assert.Equal(isError, json.RootElement.GetProperty("isError").GetBoolean());
        Assert.Equal("aGVsbG8=", json.RootElement.GetProperty("content")[1].GetProperty("data").GetString());
        Assert.Equal("obs_test", json.RootElement.GetProperty("structuredContent").GetProperty("observation_id").GetString());
        Assert.Equal(1, test.Mcp.Calls);
        Assert.Equal(0, test.Mcp.Lists);
        Assert.True(test.Mcp.Disposed);
    }

    [Theory]
    [InlineData("[]")]
    [InlineData("null")]
    [InlineData("{broken")]
    public async Task InvalidArgumentsFailBeforeCreatingAMcpConnection(string text)
    {
        using var file = new ArgumentFile(text);
        var test = new Fixtures();
        Assert.Equal(1, await test.App.RunAsync(["--json", "mcp", "call", "computer_use", "--arguments", file.Path]));
        Assert.Equal(0, test.Factories);
        Assert.Equal(0, test.Mcp.Connects);
        using var error = JsonDocument.Parse(test.Out.ToString());
        Assert.True(error.RootElement.TryGetProperty("error", out _));
    }

    [Fact]
    public async Task EmptyEndpointFailsBeforeCreatingAMcpConnection()
    {
        var test = new Fixtures();
        test.Settings.Value = test.Settings.Value with { JarvisMcpEndpoint = "" };
        Assert.Equal(1, await test.App.RunAsync(["mcp", "tools"]));
        Assert.Equal(0, test.Factories);
        Assert.Contains("mcp-url", test.Error.ToString());
    }

    [Fact]
    public async Task TokenArgumentsAreRejectedWithoutEchoingTheirValue()
    {
        var test = new Fixtures();
        Assert.Equal(1, await test.App.RunAsync(["mcp", "tools", "--token", "synthetic-private-value"]));
        Assert.Equal(0, test.Factories);
        Assert.DoesNotContain("synthetic-private-value", test.Out.ToString() + test.Error.ToString());
    }

    [Fact]
    public async Task HttpAuthenticationFailureDoesNotPrintTheTokenOrResponseBody()
    {
        const string token = "synthetic-grant-must-stay-private";
        using var handler = new DeniedHandler(token);
        using var http = new HttpClient(handler);
        var test = new Fixtures();
        var app = new CliApplication(test.Settings, new NoServices(), new NoServices(), new NoServices(), new NoServices(), test.Out, test.Error,
            settings => new JarvisMcpClient(new(new Uri(settings.JarvisMcpEndpoint), settings.JarvisMcpTokenEnvironmentVariable), http, _ => token));
        Assert.Equal(1, await app.RunAsync(["--json", "mcp", "tools"]));
        Assert.DoesNotContain(token, test.Out.ToString() + test.Error.ToString());
        using var error = JsonDocument.Parse(test.Out.ToString());
        Assert.Contains("401", error.RootElement.GetProperty("error").GetString());
        Assert.Equal(1, handler.Calls);
    }

    [Fact]
    public async Task CancellationReturns130()
    {
        var test = new Fixtures();
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        Assert.Equal(130, await test.App.RunAsync(["mcp", "tools"], cancellation.Token));
        Assert.Equal(0, test.Factories);
    }

    private sealed class Fixtures
    {
        public MemorySettings Settings { get; } = new();
        public FakeMcp Mcp { get; } = new();
        public StringWriter Out { get; } = new();
        public StringWriter Error { get; } = new();
        public int Factories { get; private set; }
        public AppSettings? ConnectionSettings { get; private set; }
        public CliApplication App => new(Settings, new NoServices(), new NoServices(), new NoServices(), new NoServices(), Out, Error,
            settings => { Factories++; ConnectionSettings = settings; return Mcp; });
    }

    private sealed class MemorySettings : ISettingsStore
    {
        public AppSettings Value { get; set; } = new() { JarvisMcpEndpoint = "https://example.test/mcp", JarvisMcpTokenEnvironmentVariable = "MCP_TEST_ACCESS_ENV" };
        public string FilePath => "unused-mcp-test-settings.json";
        public Task<AppSettings> LoadAsync(CancellationToken cancellationToken = default) { cancellationToken.ThrowIfCancellationRequested(); return Task.FromResult(Value); }
        public Task SaveAsync(AppSettings settings, CancellationToken cancellationToken = default) => throw new InvalidOperationException("MCP commands do not save settings.");
    }

    private sealed class FakeMcp : IMcpToolClient
    {
        public int Connects, Lists, Calls;
        public bool IsError, Disposed;
        public string? ToolName;
        public JsonElement Arguments;
        public Task ConnectAsync(CancellationToken cancellationToken = default) { cancellationToken.ThrowIfCancellationRequested(); Connects++; return Task.CompletedTask; }
        public Task<IReadOnlyList<McpToolDescriptor>> ListToolsAsync(CancellationToken cancellationToken = default)
        {
            Lists++;
            return Task.FromResult<IReadOnlyList<McpToolDescriptor>>([new("computer_use", "Observe then act", JsonSerializer.SerializeToElement(new
            { type = "object", properties = new { observation_id = new { type = "string" } } }))]);
        }
        public Task<McpToolResult> CallToolAsync(string name, JsonElement arguments, CancellationToken cancellationToken = default)
        {
            Calls++; ToolName = name; Arguments = arguments.Clone();
            var raw = JsonSerializer.SerializeToElement(new { isError = IsError, content = new object[]
            { new { type = "text", text = "result" }, new { type = "image", mimeType = "image/png", data = "aGVsbG8=" } }, structuredContent = new { observation_id = "obs_test" } });
            return Task.FromResult(new McpToolResult(IsError, raw.GetProperty("content"), raw.GetProperty("structuredContent"), raw));
        }
        public ValueTask DisposeAsync() { Disposed = true; return ValueTask.CompletedTask; }
    }

    private sealed class ArgumentFile : IDisposable
    {
        public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "modeldesk-mcp-" + Guid.NewGuid().ToString("N") + ".json");
        public ArgumentFile(string text) => File.WriteAllText(Path, text, new UTF8Encoding(false));
        public void Dispose() => File.Delete(Path);
    }

    private sealed class DeniedHandler(string token) : HttpMessageHandler
    {
        public int Calls;
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Calls++;
            Assert.Equal("Bearer " + token, request.Headers.Authorization?.ToString());
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.Unauthorized) { Content = new StringContent(token) });
        }
    }

    private sealed class NoServices : IHuggingFaceClient, IModelDownloader, IPythonCoreService, IReportService
    {
        public long BytesPerSecondLimit { get; set; }
        public Task<HubSearchResult> SearchAsync(string query, string? nextPage = null, CancellationToken cancellationToken = default) => throw new InvalidOperationException("MCP commands do not call the Hub.");
        public Task<HubModelDetail> GetModelAsync(string modelId, string revision = "main", CancellationToken cancellationToken = default) => throw new InvalidOperationException("MCP commands do not call the Hub.");
        public Task<DownloadResult> DownloadAsync(DownloadRequest request, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default) => throw new InvalidOperationException("MCP commands do not download models.");
        public IReadOnlyList<ModelProfile> GetProfiles(string projectRoot) => throw new InvalidOperationException("MCP commands do not load model profiles.");
        public Task<CoreRunResult> RunAsync(CoreRunRequest request, IProgress<CoreOutput>? output = null, CancellationToken cancellationToken = default) => throw new InvalidOperationException("MCP commands do not run inference.");
        public IReadOnlyList<ReportEntry> List(string projectRoot, string? profileReportDirectory = null) => throw new InvalidOperationException("MCP commands do not list reports.");
        public Task<string> ReadAsync(string path, CancellationToken cancellationToken = default) => throw new InvalidOperationException("MCP commands do not read reports.");
    }
}
