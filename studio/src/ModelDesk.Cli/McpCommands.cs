using System.Text.Json;
using ModelDesk.Core;
using ModelDesk.Infrastructure.Mcp;

namespace ModelDesk.Cli;

public sealed partial class CliApplication
{
    private async Task<int> McpAsync(string[] args, bool json, CancellationToken cancellationToken)
    {
        if (args is not ["tools"] && args is not ["call", _, "--arguments", _])
            throw new ArgumentException("Usage: mcp tools | mcp call TOOL_NAME --arguments JSON_FILE");
        var settings = await settingsStore.LoadAsync(cancellationToken);
        settings.Validate();
        if (string.IsNullOrWhiteSpace(settings.JarvisMcpEndpoint))
            throw new ArgumentException("Configure mcp-url with the Jarvis Streamable HTTP endpoint first.");
        JsonElement arguments = default;
        if (args[0] == "call")
        {
            using var stream = File.OpenRead(args[3]);
            if (stream.Length is < 2 or > 1048576) throw new ArgumentException("MCP arguments must be a JSON object file of at most 1 MiB.");
            var bytes = new byte[stream.Length];
            await stream.ReadExactlyAsync(bytes, cancellationToken);
            using var document = JsonDocument.Parse(bytes);
            if (document.RootElement.ValueKind != JsonValueKind.Object) throw new ArgumentException("MCP arguments must be an object.");
            arguments = document.RootElement.Clone();
        }
        await using var client = mcpFactory?.Invoke(settings) ?? new JarvisMcpClient(new(new Uri(settings.JarvisMcpEndpoint), settings.JarvisMcpTokenEnvironmentVariable));
        await client.ConnectAsync(cancellationToken);
        if (args[0] == "tools") { await Write(await client.ListToolsAsync(cancellationToken), json); return 0; }
        var result = await client.CallToolAsync(args[1], arguments, cancellationToken);
        await Write(result.RawResult, json);
        return result.IsError ? 2 : 0;
    }
}
