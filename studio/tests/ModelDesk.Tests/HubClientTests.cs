using System.Collections.Concurrent;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using ModelDesk.Core;
using ModelDesk.Infrastructure.Hub;

namespace ModelDesk.Tests;

public sealed class HubClientTests
{
    internal const string Revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    [Fact]
    public async Task SearchPaginatesAndRestrictsTokensToHubRequests()
    {
        var credentials = new HubTestCredentials("hf_example_only");
        using var handler = new HubTestHandler((request, _, number) =>
        {
            var response = HubTestHandler.Json(new[] { new { id = "example/model", author = "example", downloads = 42, likes = 7,
                pipeline_tag = "text-generation", library_name = "transformers", @private = false, gated = "auto", tags = new[] { "nvfp4" } } });
            if (number == 1) response.Headers.TryAddWithoutValidation("Link", "<https://huggingface.co/api/models?cursor=next>; rel=\"next\"");
            return Task.FromResult(response);
        });
        using var client = new HttpClient(handler);
        var hub = new HuggingFaceClient(client, credentials);
        var first = await hub.SearchAsync("GLM 5");
        Assert.True(first.Models[0].IsGated);
        Assert.Equal(42, first.Models[0].Downloads);
        var second = await hub.SearchAsync("GLM 5", first.NextPage);
        Assert.Null(second.NextPage);
        Assert.All(handler.Calls, call => Assert.Equal("Bearer hf_example_only", call.Authorization));
        Assert.Contains("search=GLM%205", handler.Calls.First().Uri.AbsoluteUri);
        await Assert.ThrowsAsync<InvalidDataException>(() => hub.SearchAsync("x", "https://example.org/api/models?cursor=next"));
        Assert.Equal(2, handler.Calls.Count);
    }

    [Fact]
    public async Task DetailsFreezeRevisionAndCollectEveryPaginatedNestedFile()
    {
        var readme = Encoding.UTF8.GetBytes("# A model\nPlain README\0 content");
        using var handler = new HubTestHandler((request, _, _) =>
        {
            var path = request.RequestUri!.AbsolutePath;
            if (path.EndsWith("/revision/main", StringComparison.Ordinal))
                return Task.FromResult(HubTestHandler.Json(new { id = "example/model", sha = Revision, downloads = 12, likes = 1, gated = false }));
            if (path.Contains("/tree/", StringComparison.Ordinal) && !request.RequestUri.Query.Contains("cursor", StringComparison.Ordinal))
            {
                var response = HubTestHandler.Json(new object[] { new { type = "directory", path = "nested" },
                    new { type = "file", path = "nested/model.safetensors", size = 10, oid = new string('b', 40), lfs = new { oid = new string('c', 64), size = 10 } } });
                response.Headers.TryAddWithoutValidation("Link", $"<https://huggingface.co/api/models/example/model/tree/{Revision}?recursive=true&cursor=next>; rel=\"next\"");
                return Task.FromResult(response);
            }
            if (path.Contains("/tree/", StringComparison.Ordinal))
                return Task.FromResult(HubTestHandler.Json(new[] { new { type = "file", path = "README.md", size = readme.Length, oid = GitHash(readme) } }));
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(readme) });
        });
        using var client = new HttpClient(handler);
        var detail = await new HuggingFaceClient(client, new HubTestCredentials()).GetModelAsync("example/model");
        Assert.Equal(Revision, detail.Revision);
        Assert.Equal(2, detail.Files.Count);
        Assert.Equal(10 + readme.Length, detail.TotalBytes);
        Assert.True(detail.Files.Single(file => file.Path.StartsWith("nested", StringComparison.Ordinal)).IsLfs);
        Assert.DoesNotContain('\0', detail.Readme);
        Assert.All(handler.Calls.Skip(1), call => Assert.Contains(Revision, call.Uri.AbsolutePath));
    }

    [Fact]
    public async Task UnavailableOptionalReadmeDoesNotHideRepositoryFiles()
    {
        using var handler = DetailHandler(readmeStatus: HttpStatusCode.Forbidden);
        using var client = new HttpClient(handler);
        var detail = await new HuggingFaceClient(client, new HubTestCredentials()).GetModelAsync("example/model");
        Assert.Single(detail.Files);
        Assert.Contains("unavailable", detail.Readme, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task ReadmeIsBoundedAndNotRenderedOrExecuted()
    {
        using var handler = DetailHandler(readme: Encoding.UTF8.GetBytes("<script>text only</script>" + new string('x', 140 * 1024)));
        using var client = new HttpClient(handler);
        var detail = await new HuggingFaceClient(client, new HubTestCredentials()).GetModelAsync("example/model");
        Assert.StartsWith("<script>text only</script>", detail.Readme);
        Assert.Contains("truncated", detail.Readme);
        Assert.True(detail.Readme.Length < 132 * 1024);
    }

    [Theory]
    [InlineData("https://example.org/api/models/example/model/tree/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa?cursor=x")]
    [InlineData("https://huggingface.co/api/models/other/model/tree/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa?cursor=x")]
    public async Task TreePaginationCannotChangeOriginOrRepository(string next)
    {
        using var handler = new HubTestHandler((request, _, _) =>
        {
            if (request.RequestUri!.AbsolutePath.Contains("/revision/", StringComparison.Ordinal))
                return Task.FromResult(HubTestHandler.Json(new { id = "example/model", sha = Revision }));
            var response = HubTestHandler.Json(Array.Empty<object>());
            response.Headers.TryAddWithoutValidation("Link", $"<{next}>; rel=\"next\"");
            return Task.FromResult(response);
        });
        using var client = new HttpClient(handler);
        await Assert.ThrowsAsync<InvalidDataException>(() => new HuggingFaceClient(client, new HubTestCredentials()).GetModelAsync("example/model"));
        Assert.Equal(2, handler.Calls.Count);
    }

    [Fact]
    public async Task CancellationInterruptsModelSearch()
    {
        using var handler = new HubTestHandler(async (_, token, _) =>
        {
            await Task.Delay(Timeout.InfiniteTimeSpan, token);
            return HubTestHandler.Json(Array.Empty<object>());
        });
        using var client = new HttpClient(handler);
        using var cancellation = new CancellationTokenSource(TimeSpan.FromMilliseconds(50));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => new HuggingFaceClient(client, new HubTestCredentials()).SearchAsync("model", cancellationToken: cancellation.Token));
    }

    internal static string GitHash(byte[] bytes) => Convert.ToHexString(SHA1.HashData(Encoding.UTF8.GetBytes($"blob {bytes.Length}\0").Concat(bytes).ToArray())).ToLowerInvariant();

    private static HubTestHandler DetailHandler(HttpStatusCode readmeStatus = HttpStatusCode.OK, byte[]? readme = null) =>
        new((request, _, _) =>
        {
            if (request.RequestUri!.AbsolutePath.Contains("/revision/", StringComparison.Ordinal))
                return Task.FromResult(HubTestHandler.Json(new { id = "example/model", sha = Revision }));
            if (request.RequestUri.AbsolutePath.Contains("/tree/", StringComparison.Ordinal))
                return Task.FromResult(HubTestHandler.Json(new[] { new { type = "file", path = "README.md", size = 1, oid = new string('b', 40) } }));
            return Task.FromResult(new HttpResponseMessage(readmeStatus) { Content = new ByteArrayContent(readme ?? [1]) });
        });
}

internal sealed class HubTestCredentials(string? token = null) : ICredentialStore
{
    private string? _token = token;
    public Task<string?> GetTokenAsync(CancellationToken cancellationToken = default) => Task.FromResult(_token);
    public Task SetTokenAsync(string? token, CancellationToken cancellationToken = default) { _token = token; return Task.CompletedTask; }
}

internal sealed record HubTestCall(Uri Uri, string? Authorization, string? Range, string? IfRange);

internal sealed class HubTestHandler(Func<HttpRequestMessage, CancellationToken, int, Task<HttpResponseMessage>> handler) : HttpMessageHandler
{
    private int _number;
    internal ConcurrentQueue<HubTestCall> Calls { get; } = new();
    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        Calls.Enqueue(new HubTestCall(request.RequestUri!, request.Headers.Authorization?.ToString(), request.Headers.Range?.ToString(), request.Headers.IfRange?.ToString()));
        var response = await handler(request, cancellationToken, Interlocked.Increment(ref _number));
        response.RequestMessage ??= request;
        return response;
    }
    internal static HttpResponseMessage Json(object body) => new(HttpStatusCode.OK) { Content = new ByteArrayContent(JsonSerializer.SerializeToUtf8Bytes(body)) };
}
