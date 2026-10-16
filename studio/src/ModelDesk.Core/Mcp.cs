using System.Text.Json;

namespace ModelDesk.Core;

public sealed record McpConnectionOptions(Uri Endpoint,
    string BearerTokenEnvironmentVariable = "MODEL_DESK_JARVIS_ACCESS_TOKEN", int RequestTimeoutSeconds = 120)
{
    public void Validate()
    {
        if (Endpoint is null || !Endpoint.IsAbsoluteUri ||
            (Endpoint.Scheme != Uri.UriSchemeHttps && !(Endpoint.Scheme == Uri.UriSchemeHttp && Endpoint.IsLoopback)) ||
            !string.IsNullOrEmpty(Endpoint.UserInfo) || !string.IsNullOrEmpty(Endpoint.Query) || !string.IsNullOrEmpty(Endpoint.Fragment))
            throw new ArgumentException("MCP requires HTTPS, or loopback HTTP, without URL credentials, query or fragment.");
        if (string.IsNullOrWhiteSpace(BearerTokenEnvironmentVariable) || BearerTokenEnvironmentVariable.Length > 128 ||
            BearerTokenEnvironmentVariable.Any(character => !char.IsAsciiLetterOrDigit(character) && character != '_'))
            throw new ArgumentException("Choose a valid environment variable name for the MCP OAuth access token.");
        if (RequestTimeoutSeconds is < 1 or > 600) throw new ArgumentOutOfRangeException(nameof(RequestTimeoutSeconds));
    }
}

public sealed record McpToolDescriptor(string Name, string Description, JsonElement InputSchema, JsonElement? OutputSchema = null);

/// <summary>Opaque MCP blocks preserve image data and structured tool output for the consuming model adapter.</summary>
public sealed record McpToolResult(bool IsError, JsonElement Content, JsonElement? StructuredContent, JsonElement RawResult);

public interface IMcpToolClient : IAsyncDisposable
{
    Task ConnectAsync(CancellationToken cancellationToken = default);
    Task<IReadOnlyList<McpToolDescriptor>> ListToolsAsync(CancellationToken cancellationToken = default);
    Task<McpToolResult> CallToolAsync(string name, JsonElement arguments, CancellationToken cancellationToken = default);
}
