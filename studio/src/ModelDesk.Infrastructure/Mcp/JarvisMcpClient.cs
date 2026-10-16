using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using ModelDesk.Core;

namespace ModelDesk.Infrastructure.Mcp;

/// <summary>
/// Streamable HTTP client for the OAuth-bound Jarvis device. It never obtains enrollment credentials,
/// bypasses Agent approvals, or retries a tool call whose effect may already have occurred.
/// </summary>
public sealed class JarvisMcpClient : IMcpToolClient
{
    public const int MaximumResponseBytes = 8 * 1024 * 1024;
    private const int MaximumRequestBytes = 256 * 1024;
    private const string RequestedProtocolVersion = "2025-11-25";
    private static readonly HashSet<string> SupportedVersions = [RequestedProtocolVersion, "2025-06-18", "2025-03-26"];
    private readonly McpConnectionOptions options;
    private readonly HttpClient http;
    private readonly bool ownsHttp;
    private readonly Func<string, string?> environmentReader;
    private readonly SemaphoreSlim gate = new(1, 1);
    private long nextId;
    private string? sessionId;
    private string? protocolVersion;
    private bool initialized;
    private bool disposed;

    public JarvisMcpClient(McpConnectionOptions options, HttpClient? http = null,
        Func<string, string?>? environmentReader = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        this.options = options;
        ownsHttp = http is null;
        // Never forward an OAuth bearer grant to a redirect target.
        this.http = http ?? new HttpClient(new HttpClientHandler { AllowAutoRedirect = false }) { Timeout = Timeout.InfiniteTimeSpan };
        this.environmentReader = environmentReader ?? Environment.GetEnvironmentVariable;
    }

    public async Task ConnectAsync(CancellationToken cancellationToken = default)
    {
        await gate.WaitAsync(cancellationToken);
        try { ThrowIfDisposed(); await InitializeAsync(cancellationToken); }
        finally { gate.Release(); }
    }

    public async Task<IReadOnlyList<McpToolDescriptor>> ListToolsAsync(CancellationToken cancellationToken = default)
    {
        await gate.WaitAsync(cancellationToken);
        try
        {
            ThrowIfDisposed();
            await InitializeAsync(cancellationToken);
            var tools = new List<McpToolDescriptor>();
            var names = new HashSet<string>(StringComparer.Ordinal);
            var cursors = new HashSet<string>(StringComparer.Ordinal);
            string? cursor = null;
            long schemaBytes = 0;
            for (var page = 0; page < 100; page++)
            {
                var result = await RequestAsync("tools/list", cursor is null ? new { } : (object)new { cursor }, cancellationToken);
                if (!result.TryGetProperty("tools", out var entries) || entries.ValueKind != JsonValueKind.Array)
                    throw Invalid("tools/list must return a tools array.");
                foreach (var entry in entries.EnumerateArray())
                {
                    var name = RequiredString(entry, "name", 256);
                    if (!names.Add(name)) throw Invalid("MCP returned duplicate tool names.");
                    if (!entry.TryGetProperty("inputSchema", out var input) || input.ValueKind != JsonValueKind.Object)
                        throw Invalid("MCP tool inputSchema must be an object.");
                    JsonElement? output = entry.TryGetProperty("outputSchema", out var outputNode) ? outputNode.Clone() : null;
                    if (output is { ValueKind: not JsonValueKind.Object }) throw Invalid("MCP tool outputSchema must be an object.");
                    var description = entry.TryGetProperty("description", out _)
                        ? RequiredString(entry, "description", 65536, allowEmpty: true) : "";
                    schemaBytes += Encoding.UTF8.GetByteCount(entry.GetRawText());
                    if (tools.Count >= 2048 || schemaBytes > MaximumResponseBytes)
                        throw Invalid("MCP tool catalog exceeds the supported limit.");
                    tools.Add(new(name, description, input.Clone(), output));
                }
                if (!result.TryGetProperty("nextCursor", out var next) || next.ValueKind == JsonValueKind.Null) return tools.AsReadOnly();
                cursor = RequiredString(result, "nextCursor", 8192);
                if (!cursors.Add(cursor)) throw Invalid("MCP pagination repeated a cursor.");
            }
            throw Invalid("MCP tool catalog exceeds 100 pages.");
        }
        finally { gate.Release(); }
    }

    public async Task<McpToolResult> CallToolAsync(string name, JsonElement arguments, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(name) || name.Length > 256 || name.Any(char.IsControl)) throw new ArgumentException("Choose an MCP tool name.", nameof(name));
        if (arguments.ValueKind != JsonValueKind.Object) throw new ArgumentException("MCP tool arguments must be an object.", nameof(arguments));
        // Freeze caller-owned JSON before waiting or dispatching.
        var frozen = arguments.Clone();
        await gate.WaitAsync(cancellationToken);
        try
        {
            ThrowIfDisposed();
            await InitializeAsync(cancellationToken);
            var result = await RequestAsync("tools/call", new { name, arguments = frozen }, cancellationToken);
            if (!result.TryGetProperty("content", out var content) || content.ValueKind != JsonValueKind.Array ||
                content.EnumerateArray().Any(block => block.ValueKind != JsonValueKind.Object))
                throw Invalid("MCP tool result must contain content blocks.");
            var error = false;
            if (result.TryGetProperty("isError", out var errorNode))
            {
                if (errorNode.ValueKind is not (JsonValueKind.True or JsonValueKind.False)) throw Invalid("Invalid MCP tool error flag.");
                error = errorNode.GetBoolean();
            }
            JsonElement? structured = result.TryGetProperty("structuredContent", out var structuredNode) ? structuredNode.Clone() : null;
            return new(error, content.Clone(), structured, result);
        }
        finally { gate.Release(); }
    }

    private async Task InitializeAsync(CancellationToken cancellationToken)
    {
        if (initialized) return;
        // Any failed handshake is incomplete. A subsequent explicit attempt starts a new session.
        sessionId = protocolVersion = null;
        var result = await RequestAsync("initialize", new
        {
            protocolVersion = RequestedProtocolVersion,
            capabilities = new { },
            clientInfo = new { name = "ModelDesk", version = "1.0" }
        }, cancellationToken);
        var negotiated = RequiredString(result, "protocolVersion", 32);
        if (!SupportedVersions.Contains(negotiated)) throw Invalid("MCP server selected an unsupported protocol version.");
        if (!result.TryGetProperty("capabilities", out var capabilities) || capabilities.ValueKind != JsonValueKind.Object ||
            !capabilities.TryGetProperty("tools", out var tools) || tools.ValueKind != JsonValueKind.Object)
            throw Invalid("MCP server does not advertise tool support.");
        protocolVersion = negotiated;
        await SendAsync("notifications/initialized", new { }, null, cancellationToken);
        initialized = true;
    }

    private async Task<JsonElement> RequestAsync(string method, object parameters, CancellationToken cancellationToken)
    {
        var id = ++nextId;
        var response = await SendAsync(method, parameters, id, cancellationToken);
        return response ?? throw Invalid("MCP request completed without a JSON-RPC response.");
    }

    private async Task<JsonElement?> SendAsync(string method, object parameters, long? id, CancellationToken cancellationToken)
    {
        var token = environmentReader(options.BearerTokenEnvironmentVariable)?.Trim();
        if (string.IsNullOrEmpty(token))
            throw new InvalidOperationException($"Set {options.BearerTokenEnvironmentVariable} to an OAuth access token issued for this MCP resource and device.");
        if (token.Length > 16384 || token.Any(character => char.IsWhiteSpace(character) || char.IsControl(character)))
            throw new InvalidOperationException("The MCP access token has an invalid format.");
        var payload = id is null ? JsonSerializer.SerializeToUtf8Bytes(new { jsonrpc = "2.0", method, @params = parameters })
            : JsonSerializer.SerializeToUtf8Bytes(new { jsonrpc = "2.0", id, method, @params = parameters });
        if (payload.Length > MaximumRequestBytes) throw new ArgumentException("MCP request exceeds 256 KiB.");
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(TimeSpan.FromSeconds(options.RequestTimeoutSeconds));
        using var request = new HttpRequestMessage(HttpMethod.Post, options.Endpoint);
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("text/event-stream"));
        if (sessionId is not null) request.Headers.Add("Mcp-Session-Id", sessionId);
        if (protocolVersion is not null) request.Headers.Add("MCP-Protocol-Version", protocolVersion);
        request.Content = new ByteArrayContent(payload);
        request.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
        using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, timeout.Token);
        if (!response.IsSuccessStatusCode)
        {
            var expiredSession = response.StatusCode == HttpStatusCode.NotFound && sessionId is not null;
            if (expiredSession)
            {
                initialized = false;
                sessionId = protocolVersion = null;
            }
            var hint = response.StatusCode switch
            {
                HttpStatusCode.Unauthorized => "OAuth access token is missing, expired or rejected; update its environment variable.",
                HttpStatusCode.Forbidden => "MCP authorization was denied.",
                HttpStatusCode.NotFound when expiredSession => "MCP transport session expired; reconnect explicitly before continuing.",
                _ => "MCP HTTP request failed; the request was not retried."
            };
            throw new HttpRequestException($"{hint} HTTP {(int)response.StatusCode}.", null, response.StatusCode);
        }
        if (method == "initialize" && response.Headers.TryGetValues("Mcp-Session-Id", out var sessionValues))
        {
            var values = sessionValues.ToArray();
            if (values.Length != 1 || values[0].Length is < 1 or > 512 || values[0].Any(character => character is < '!' or > '~'))
                throw Invalid("MCP server returned an invalid transport session ID.");
            sessionId = values[0];
        }
        if (id is null) return null;
        var media = response.Content.Headers.ContentType?.MediaType;
        if (media is "text/event-stream")
            return await ReadEventStreamAsync(response.Content, id.Value, timeout.Token);
        if (media is not "application/json") throw Invalid("MCP response must be JSON or an event stream.");
        var bytes = await ReadBoundedAsync(response.Content, timeout.Token);
        using var document = Parse(bytes);
        return ExtractReply(document.RootElement, id.Value) ?? throw Invalid("MCP response has no matching JSON-RPC result.");
    }

    private static async Task<byte[]> ReadBoundedAsync(HttpContent content, CancellationToken cancellationToken)
    {
        if (content.Headers.ContentLength > MaximumResponseBytes) throw Invalid("MCP response exceeds 8 MiB.");
        await using var stream = await content.ReadAsStreamAsync(cancellationToken);
        using var buffer = new MemoryStream();
        var chunk = new byte[8192];
        while (true)
        {
            var read = await stream.ReadAsync(chunk, cancellationToken);
            if (read == 0) return buffer.ToArray();
            if (buffer.Length + read > MaximumResponseBytes) throw Invalid("MCP response exceeds 8 MiB.");
            buffer.Write(chunk, 0, read);
        }
    }

    private static async Task<JsonElement> ReadEventStreamAsync(HttpContent content, long id, CancellationToken cancellationToken)
    {
        // Stream the response so a long-lived SSE connection completes as soon as its matching reply arrives.
        await using var stream = await content.ReadAsStreamAsync(cancellationToken);
        var chunk = new byte[8192];
        using var line = new MemoryStream();
        var data = new StringBuilder();
        long total = 0;
        JsonElement? ProcessLine()
        {
            string text;
            try { text = new UTF8Encoding(false, true).GetString(line.GetBuffer(), 0, checked((int)line.Length)).TrimEnd('\r'); }
            catch (DecoderFallbackException) { throw Invalid("MCP event stream contains invalid UTF-8."); }
            line.SetLength(0);
            if (text.Length == 0)
            {
                if (data.Length == 0) return null;
                using var document = Parse(Encoding.UTF8.GetBytes(data.ToString()));
                data.Clear();
                return ExtractReply(document.RootElement, id, allowNotifications: true);
            }
            if (text.StartsWith("data:", StringComparison.Ordinal))
            {
                var value = text.AsSpan(5);
                if (value.StartsWith(" ")) value = value[1..];
                if (data.Length > 0) data.Append('\n');
                data.Append(value);
            }
            return null;
        }
        while (true)
        {
            var read = await stream.ReadAsync(chunk, cancellationToken);
            if (read == 0) break;
            total += read;
            if (total > MaximumResponseBytes) throw Invalid("MCP event stream exceeds 8 MiB.");
            for (var index = 0; index < read; index++)
            {
                if (chunk[index] == '\n')
                {
                    if (ProcessLine() is { } reply) return reply;
                }
                else line.WriteByte(chunk[index]);
            }
        }
        if (line.Length > 0 && ProcessLine() is { } tail) return tail;
        if (data.Length > 0)
        {
            using var document = Parse(Encoding.UTF8.GetBytes(data.ToString()));
            if (ExtractReply(document.RootElement, id, allowNotifications: true) is { } last) return last;
        }
        throw Invalid("MCP event stream ended before the matching response.");
    }

    private static JsonDocument Parse(byte[] bytes)
    {
        try { return JsonDocument.Parse(bytes, new JsonDocumentOptions { MaxDepth = 64 }); }
        catch (JsonException) { throw Invalid("MCP response contains invalid JSON."); }
    }

    private static JsonElement? ExtractReply(JsonElement root, long id, bool allowNotifications = false)
    {
        if (root.ValueKind != JsonValueKind.Object ||
            root.EnumerateObject().Select(property => property.Name).Distinct(StringComparer.Ordinal).Count() != root.EnumerateObject().Count() ||
            !root.TryGetProperty("jsonrpc", out var version) || version.ValueKind != JsonValueKind.String || version.GetString() != "2.0")
            throw Invalid("Invalid MCP JSON-RPC envelope.");
        if (!root.TryGetProperty("id", out var responseId))
        {
            if (allowNotifications && root.TryGetProperty("method", out var method) && method.ValueKind == JsonValueKind.String) return null;
            throw Invalid("MCP response has no JSON-RPC request ID.");
        }
        if (responseId.ValueKind != JsonValueKind.Number || !responseId.TryGetInt64(out var actual) || actual != id)
            throw Invalid("MCP response has a mismatched JSON-RPC request ID.");
        var hasResult = root.TryGetProperty("result", out var result);
        var hasError = root.TryGetProperty("error", out var error);
        if (hasResult == hasError) throw Invalid("MCP response must contain exactly one result or error.");
        if (hasError)
        {
            if (error.ValueKind != JsonValueKind.Object || !error.TryGetProperty("code", out var codeNode) ||
                codeNode.ValueKind != JsonValueKind.Number || !codeNode.TryGetInt32(out var code))
                throw Invalid("MCP response contains an invalid JSON-RPC error.");
            throw Invalid($"MCP JSON-RPC request failed (code {code}). Server error details were omitted.");
        }
        if (result.ValueKind != JsonValueKind.Object) throw Invalid("MCP response must contain an object result.");
        return result.Clone();
    }

    private static string RequiredString(JsonElement root, string property, int maximum, bool allowEmpty = false)
    {
        if (root.ValueKind != JsonValueKind.Object || !root.TryGetProperty(property, out var field) || field.ValueKind != JsonValueKind.String)
            throw Invalid($"MCP response has an invalid {property} field.");
        var value = field.GetString()!;
        if ((!allowEmpty && string.IsNullOrWhiteSpace(value)) || value.Length > maximum || (!allowEmpty && value.Any(char.IsControl)))
            throw Invalid($"MCP response has an invalid {property} field.");
        return value;
    }

    private static InvalidDataException Invalid(string message) => new(message);
    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(disposed, this);
    public async ValueTask DisposeAsync()
    {
        await gate.WaitAsync();
        try { if (disposed) return; disposed = true; if (ownsHttp) http.Dispose(); }
        finally { gate.Release(); }
    }
}
