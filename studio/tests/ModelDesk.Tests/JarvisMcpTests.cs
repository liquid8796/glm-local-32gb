using System.Net;
using System.Text;
using System.Text.Json;
using ModelDesk.Core;
using ModelDesk.Infrastructure.Mcp;

namespace ModelDesk.Tests;

public sealed class JarvisMcpTests
{
    [Fact]
    public async Task InitializesOnceAndPaginatesWithNegotiatedSessionAndVersion()
    {
        using var handler = new McpHandler((request, json, index, cancellationToken) =>
        {
            Assert.Equal("Bearer example-test-token", request.Headers.Authorization?.ToString());
            Assert.Contains(request.Headers.Accept, value => value.MediaType == "application/json");
            Assert.Contains(request.Headers.Accept, value => value.MediaType == "text/event-stream");
            var method = json.GetProperty("method").GetString();
            if (index == 1)
            {
                Assert.Equal("initialize", method);
                Assert.Equal("2025-11-25", json.GetProperty("params").GetProperty("protocolVersion").GetString());
                var response = Reply(json, new { protocolVersion = "2025-11-25", capabilities = new { tools = new { listChanged = true } }, serverInfo = new { name = "jarvis", version = "test" } });
                response.Headers.TryAddWithoutValidation("Mcp-Session-Id", "session-test");
                return Task.FromResult(response);
            }
            Assert.Equal("session-test", request.Headers.GetValues("Mcp-Session-Id").Single());
            Assert.Equal("2025-11-25", request.Headers.GetValues("MCP-Protocol-Version").Single());
            if (method == "notifications/initialized") return Task.FromResult(new HttpResponseMessage(HttpStatusCode.Accepted));
            Assert.Equal("tools/list", method);
            if (index == 3)
            {
                Assert.False(json.GetProperty("params").TryGetProperty("cursor", out _));
                return Task.FromResult(Reply(json, new { tools = new[] { Tool("computer_use") }, nextCursor = "next" }));
            }
            Assert.Equal("next", json.GetProperty("params").GetProperty("cursor").GetString());
            return Task.FromResult(Reply(json, new { tools = new[] { Tool("jarvis__tool_call") } }));
        });
        using var http = new HttpClient(handler);
        await using var mcp = Client(http);
        await mcp.ConnectAsync();
        var tools = await mcp.ListToolsAsync();
        Assert.Equal(new[] { "computer_use", "jarvis__tool_call" }, tools.Select(tool => tool.Name));
        Assert.Equal("object", tools[0].InputSchema.GetProperty("type").GetString());
        Assert.Equal(4, handler.Calls);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task PreservesTextImagesStructuredContentAndToolError(bool sse)
    {
        using var handler = new McpHandler((_, json, _, _) =>
        {
            if (Handshake(json, out var response)) return Task.FromResult(response);
            Assert.Equal("tools/call", json.GetProperty("method").GetString());
            var parameters = json.GetProperty("params");
            Assert.Equal("computer_use", parameters.GetProperty("name").GetString());
            Assert.Equal("list_windows", parameters.GetProperty("arguments").GetProperty("action").GetString());
            var result = new { isError = true, content = new object[] { new { type = "text", text = "denied" }, new { type = "image", mimeType = "image/png", data = "aGVsbG8=" } }, structuredContent = new { text = "denied", isError = true } };
            return Task.FromResult(sse ? SseReply(json, result) : Reply(json, result));
        });
        using var http = new HttpClient(handler);
        await using var mcp = Client(http);
        var result = await mcp.CallToolAsync("computer_use", Element(new { action = "list_windows" }));
        Assert.True(result.IsError);
        Assert.Equal("denied", result.Content[0].GetProperty("text").GetString());
        Assert.Equal("aGVsbG8=", result.Content[1].GetProperty("data").GetString());
        Assert.True(result.StructuredContent!.Value.GetProperty("isError").GetBoolean());
        Assert.Equal("image/png", result.RawResult.GetProperty("content")[1].GetProperty("mimeType").GetString());
    }

    [Fact]
    public async Task MissingEnvironmentTokenFailsBeforeNetwork()
    {
        using var handler = new McpHandler((_, _, _, _) => throw new InvalidOperationException("No request expected"));
        using var http = new HttpClient(handler);
        await using var mcp = new JarvisMcpClient(new(new Uri("https://example.test/mcp")), http, _ => null);
        var error = await Assert.ThrowsAsync<InvalidOperationException>(() => mcp.ListToolsAsync());
        Assert.Contains("MODEL_DESK_JARVIS_ACCESS_TOKEN", error.Message);
        Assert.Equal(0, handler.Calls);
    }

    [Theory]
    [InlineData("http://example.test/mcp")]
    [InlineData("https://user:secret@example.test/mcp")]
    [InlineData("https://example.test/mcp?access_token=example")]
    [InlineData("https://example.test/mcp#secret")]
    public void RejectsUnsafeEndpoints(string endpoint) =>
        Assert.Throws<ArgumentException>(() => new McpConnectionOptions(new Uri(endpoint)).Validate());

    [Fact]
    public async Task RejectsMalformedRpcAndDoesNotReflectServerErrorDetails()
    {
        using var handler = new McpHandler((_, json, _, _) =>
        {
            if (Handshake(json, out var response)) return Task.FromResult(response);
            return Task.FromResult(Json(new { jsonrpc = "2.0", id = json.GetProperty("id").GetInt64(), error = new { code = -32602, message = "secret-test-must-not-appear" } }));
        });
        using var http = new HttpClient(handler);
        await using var mcp = Client(http);
        var error = await Assert.ThrowsAsync<InvalidDataException>(() => mcp.ListToolsAsync());
        Assert.Contains("-32602", error.Message);
        Assert.DoesNotContain("secret-test", error.ToString());
    }

    [Fact]
    public async Task RejectsWrongResponseId()
    {
        using var handler = new McpHandler((_, json, _, _) =>
        {
            if (Handshake(json, out var response)) return Task.FromResult(response);
            return Task.FromResult(Json(new { jsonrpc = "2.0", id = 999999, result = new { tools = Array.Empty<object>() } }));
        });
        using var http = new HttpClient(handler);
        await using var mcp = Client(http);
        await Assert.ThrowsAsync<InvalidDataException>(() => mcp.ListToolsAsync());
    }

    [Theory]
    [InlineData("{\"jsonrpc\":3,\"id\":2,\"result\":{}}")]
    [InlineData("{\"jsonrpc\":\"2.0\",\"id\":2,\"error\":{\"code\":\"invalid\"}}")]
    [InlineData("{\"jsonrpc\":\"2.0\",\"id\":2,\"error\":{\"code\":-1},\"result\":{}}")]
    [InlineData("{\"jsonrpc\":\"2.0\",\"id\":2,\"id\":2,\"result\":{\"tools\":[]}}")]
    public async Task RejectsMalformedRpcEnvelopes(string envelope)
    {
        using var handler = new McpHandler((_, json, _, _) => Handshake(json, out var response) ? Task.FromResult(response)
            : Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(envelope, Encoding.UTF8, "application/json") }));
        using var http = new HttpClient(handler);
        await using var mcp = Client(http);
        await Assert.ThrowsAsync<InvalidDataException>(() => mcp.ListToolsAsync());
    }

    [Fact]
    public async Task RequestTimeoutBoundsPendingHttpCalls()
    {
        using var handler = new McpHandler(async (_, json, _, token) =>
        {
            if (Handshake(json, out var response)) return response;
            await Task.Delay(Timeout.InfiniteTimeSpan, token);
            return Reply(json, new { tools = Array.Empty<object>() });
        });
        using var http = new HttpClient(handler) { Timeout = Timeout.InfiniteTimeSpan };
        await using var mcp = new JarvisMcpClient(new(new Uri("https://example.test/mcp"), RequestTimeoutSeconds: 1), http, _ => "example-test-token");
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => mcp.ListToolsAsync());
    }

    [Fact]
    public async Task TokenIsReadAgainAfterOAuthGrantChanges()
    {
        var accessToken = "old-test-grant";
        using var handler = new McpHandler((request, json, _, _) =>
        {
            Assert.Equal("Bearer " + accessToken, request.Headers.Authorization?.ToString());
            if (Handshake(json, out var response)) return Task.FromResult(response);
            return Task.FromResult(Reply(json, new { tools = Array.Empty<object>() }));
        });
        using var http = new HttpClient(handler);
        await using var mcp = new JarvisMcpClient(new(new Uri("https://example.test/mcp")), http, _ => accessToken);
        await mcp.ConnectAsync();
        accessToken = "new-test-grant";
        await mcp.ListToolsAsync();
    }

    [Fact]
    public async Task DoesNotRetryAnUncertainToolCall()
    {
        var toolCalls = 0;
        using var handler = new McpHandler((_, json, _, _) =>
        {
            if (Handshake(json, out var response)) return Task.FromResult(response);
            toolCalls++;
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.ServiceUnavailable) { Content = new StringContent("secret-test") });
        });
        using var http = new HttpClient(handler);
        await using var mcp = Client(http);
        var error = await Assert.ThrowsAsync<HttpRequestException>(() => mcp.CallToolAsync("computer_use", Element(new { action = "click" })));
        Assert.Equal(HttpStatusCode.ServiceUnavailable, error.StatusCode);
        Assert.DoesNotContain("secret-test", error.ToString());
        Assert.Equal(1, toolCalls);
    }

    [Fact]
    public async Task ExpiredSessionDoesNotReplayToolAndExplicitReconnectStartsANewSession()
    {
        var initializations = 0;
        var toolCalls = 0;
        using var handler = new McpHandler((request, json, _, _) =>
        {
            var method = json.GetProperty("method").GetString();
            if (method == "initialize")
            {
                Assert.False(request.Headers.Contains("Mcp-Session-Id"));
                Assert.False(request.Headers.Contains("MCP-Protocol-Version"));
                initializations++;
                var response = Reply(json, new { protocolVersion = "2025-11-25", capabilities = new { tools = new { } }, serverInfo = new { name = "jarvis", version = "test" } });
                response.Headers.TryAddWithoutValidation("Mcp-Session-Id", "session-" + initializations);
                return Task.FromResult(response);
            }
            Assert.Equal("session-" + initializations, request.Headers.GetValues("Mcp-Session-Id").Single());
            if (method == "notifications/initialized") return Task.FromResult(new HttpResponseMessage(HttpStatusCode.Accepted));
            Assert.Equal("tools/call", method);
            toolCalls++;
            return Task.FromResult(toolCalls == 1 ? new HttpResponseMessage(HttpStatusCode.NotFound)
                : Reply(json, new { content = new[] { new { type = "text", text = "explicit second action" } }, isError = false }));
        });
        using var http = new HttpClient(handler);
        await using var mcp = Client(http);
        var error = await Assert.ThrowsAsync<HttpRequestException>(() => mcp.CallToolAsync("computer_use", Element(new { action = "list_windows" })));
        Assert.Equal(HttpStatusCode.NotFound, error.StatusCode);
        Assert.Contains("reconnect explicitly", error.Message);
        Assert.Equal(1, initializations);
        Assert.Equal(1, toolCalls);
        Assert.Equal(3, handler.Calls);

        await mcp.ConnectAsync();
        Assert.Equal(2, initializations);
        Assert.Equal(1, toolCalls);
        Assert.Equal(5, handler.Calls);

        var result = await mcp.CallToolAsync("computer_use", Element(new { action = "list_windows" }));
        Assert.False(result.IsError);
        Assert.Equal("explicit second action", result.Content[0].GetProperty("text").GetString());
        Assert.Equal(2, toolCalls);
        Assert.Equal(6, handler.Calls);
    }

    [Fact]
    public async Task CancellationInterruptsPendingRequest()
    {
        using var handler = new McpHandler(async (_, json, _, token) =>
        {
            if (Handshake(json, out var response)) return response;
            await Task.Delay(Timeout.InfiniteTimeSpan, token);
            return Reply(json, new { tools = Array.Empty<object>() });
        });
        using var http = new HttpClient(handler);
        await using var mcp = Client(http);
        using var cancellation = new CancellationTokenSource(TimeSpan.FromMilliseconds(80));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => mcp.ListToolsAsync(cancellation.Token));
    }

    [Fact]
    public async Task RejectsCyclicPagination()
    {
        using var handler = new McpHandler((_, json, _, _) => Handshake(json, out var response)
            ? Task.FromResult(response) : Task.FromResult(Reply(json, new { tools = Array.Empty<object>(), nextCursor = "same" })));
        using var http = new HttpClient(handler);
        await using var mcp = Client(http);
        await Assert.ThrowsAsync<InvalidDataException>(() => mcp.ListToolsAsync());
        Assert.Equal(4, handler.Calls);
    }

    [Fact]
    public async Task RejectsOversizedBodiesBeforeParsing()
    {
        using var handler = new McpHandler((_, json, _, _) => Handshake(json, out var response) ? Task.FromResult(response)
            : Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(new string('x', JarvisMcpClient.MaximumResponseBytes + 1), Encoding.UTF8, "application/json") }));
        using var http = new HttpClient(handler);
        await using var mcp = Client(http);
        await Assert.ThrowsAsync<InvalidDataException>(() => mcp.ListToolsAsync());
    }

    private static JarvisMcpClient Client(HttpClient http) => new(new(new Uri("https://example.test/mcp")), http, _ => "example-test-token");
    private static JsonElement Element(object value) => JsonSerializer.SerializeToElement(value);
    private static object Tool(string name) => new { name, description = "Tool", inputSchema = new { type = "object", additionalProperties = true } };
    private static bool Handshake(JsonElement json, out HttpResponseMessage response)
    {
        var method = json.GetProperty("method").GetString();
        response = method == "initialize" ? Reply(json, new { protocolVersion = "2025-11-25", capabilities = new { tools = new { } }, serverInfo = new { name = "jarvis", version = "test" } })
            : new HttpResponseMessage(HttpStatusCode.Accepted);
        return method is "initialize" or "notifications/initialized";
    }
    private static HttpResponseMessage Reply(JsonElement request, object result) => Json(new { jsonrpc = "2.0", id = request.GetProperty("id").GetInt64(), result });
    private static HttpResponseMessage Json(object value) => new(HttpStatusCode.OK) { Content = new StringContent(JsonSerializer.Serialize(value), Encoding.UTF8, "application/json") };
    private static HttpResponseMessage SseReply(JsonElement request, object result) => new(HttpStatusCode.OK)
    {
        Content = new StringContent(": heartbeat\n\nevent: message\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}\n\n"
            + "event: message\ndata: " + JsonSerializer.Serialize(new { jsonrpc = "2.0", id = request.GetProperty("id").GetInt64(), result }) + "\n\n", Encoding.UTF8, "text/event-stream")
    };

    private sealed class McpHandler(Func<HttpRequestMessage, JsonElement, int, CancellationToken, Task<HttpResponseMessage>> handler) : HttpMessageHandler
    {
        internal int Calls;
        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Assert.Equal(HttpMethod.Post, request.Method);
            Assert.Equal("https://example.test/mcp", request.RequestUri!.AbsoluteUri);
            using var json = JsonDocument.Parse(await request.Content!.ReadAsStringAsync(cancellationToken));
            return await handler(request, json.RootElement, Interlocked.Increment(ref Calls), cancellationToken);
        }
    }
}
