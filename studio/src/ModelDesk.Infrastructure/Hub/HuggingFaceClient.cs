using System.Net;
using System.Text;
using System.Text.Json;
using ModelDesk.Core;

namespace ModelDesk.Infrastructure.Hub;

/// <summary>Read-only Hub APIs. Repository text is returned as plain untrusted text.</summary>
public sealed class HuggingFaceClient(HttpClient httpClient, ICredentialStore credentials) : IHuggingFaceClient
{
    public async Task<HubSearchResult> SearchAsync(string query, string? nextPage = null, CancellationToken cancellationToken = default)
    {
        if (query is null || query.Length > 512 || query.Any(char.IsControl)) throw new ArgumentException("Search text is limited to 512 characters.");
        var uri = nextPage is null ? new Uri($"https://huggingface.co/api/models?search={Uri.EscapeDataString(query.Trim())}&sort=downloads&direction=-1&limit=40&full=true")
            : HubHttp.ApiUri(nextPage, "/api/models");
        using var response = await HubHttp.SendAsync(httpClient, credentials, uri, cancellationToken).ConfigureAwait(false);
        HubHttp.EnsureSuccess(response);
        using var json = await HubHttp.JsonAsync(response.Content, cancellationToken).ConfigureAwait(false);
        if (json.RootElement.ValueKind != JsonValueKind.Array || json.RootElement.GetArrayLength() > 1000)
            throw new InvalidDataException("The Hub returned an invalid model search page.");
        var models = json.RootElement.EnumerateArray().Select(item => new HubModelSummary(
            HubHttp.ModelId(Text(item, "id") ?? Text(item, "modelId") ?? throw new InvalidDataException("Model ID is absent.")),
            Text(item, "author"), Number(item, "downloads"), (int)Math.Clamp(Number(item, "likes"), 0, int.MaxValue),
            Text(item, "pipeline_tag"), Text(item, "library_name"), Boolean(item, "private"), Gated(item),
            DateTimeOffset.TryParse(Text(item, "lastModified"), out var modified) ? modified : null,
            item.TryGetProperty("tags", out var tags) && tags.ValueKind == JsonValueKind.Array
                ? tags.EnumerateArray().Where(tag => tag.ValueKind == JsonValueKind.String).Select(tag => tag.GetString()!).Take(256).ToArray() : [])).ToArray();
        return new HubSearchResult(models, HubHttp.NextPage(response, "/api/models"));
    }

    public async Task<HubModelDetail> GetModelAsync(string modelId, string revision = "main", CancellationToken cancellationToken = default)
    {
        var encoded = HubHttp.EncodedModel(modelId);
        using var response = await HubHttp.SendAsync(httpClient, credentials,
            new Uri($"https://huggingface.co/api/models/{encoded}/revision/{Uri.EscapeDataString(HubHttp.Revision(revision))}"), cancellationToken).ConfigureAwait(false);
        HubHttp.EnsureSuccess(response);
        using var json = await HubHttp.JsonAsync(response.Content, cancellationToken).ConfigureAwait(false);
        var model = json.RootElement;
        var resolvedId = HubHttp.ModelId(Text(model, "id") ?? modelId);
        if (!resolvedId.Equals(modelId, StringComparison.OrdinalIgnoreCase)) throw new InvalidDataException("Hub model identity differs from the requested repository.");
        var sha = HubHttp.ImmutableRevision(Text(model, "sha") ?? "");
        if (HubHttp.IsSha1(revision) && !sha.Equals(revision, StringComparison.OrdinalIgnoreCase)) throw new InvalidDataException("Hub commit differs from the requested immutable revision.");
        var treePath = $"/api/models/{encoded}/tree/{sha}";
        string? page = $"https://huggingface.co{treePath}?recursive=true&expand=false&limit=1000";
        var visited = new HashSet<string>(StringComparer.Ordinal);
        var files = new Dictionary<string, HubFile>(StringComparer.Ordinal);
        while (page is not null)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (visited.Count >= 1000 || !visited.Add(page)) throw new InvalidDataException("Hub file pagination is cyclic or exceeds its limit.");
            using var treeResponse = await HubHttp.SendAsync(httpClient, credentials, HubHttp.ApiUri(page, treePath), cancellationToken).ConfigureAwait(false);
            HubHttp.EnsureSuccess(treeResponse);
            using var tree = await HubHttp.JsonAsync(treeResponse.Content, cancellationToken).ConfigureAwait(false);
            if (tree.RootElement.ValueKind != JsonValueKind.Array) throw new InvalidDataException("The Hub returned an invalid repository tree.");
            foreach (var item in tree.RootElement.EnumerateArray())
            {
                if (Text(item, "type") == "directory") continue;
                if (Text(item, "type") != "file") throw new InvalidDataException("The Hub returned an unsupported repository entry.");
                var path = Text(item, "path") ?? throw new InvalidDataException("Hub file path is absent.");
                if (path.Length is 0 or > 4096 || path.Any(char.IsControl)) throw new InvalidDataException("Hub file path is invalid.");
                var size = RequiredSize(item);
                string? lfsHash = null;
                if (item.TryGetProperty("lfs", out var lfs) && lfs.ValueKind == JsonValueKind.Object)
                {
                    lfsHash = Text(lfs, "oid");
                    if (!HubHttp.IsSha256(lfsHash)) throw new InvalidDataException("Hub LFS SHA-256 is invalid.");
                    if (lfs.TryGetProperty("size", out var lfsSize) && (!lfsSize.TryGetInt64(out var value) || value != size))
                        throw new InvalidDataException("Hub file and LFS sizes differ.");
                }
                var blob = Text(item, "oid");
                if (!HubHttp.IsSha1(blob)) blob = null;
                if (lfsHash is null && blob is null) throw new InvalidDataException("Hub file has no verifiable content hash.");
                var file = new HubFile(path, size, lfsHash?.ToLowerInvariant(), blob?.ToLowerInvariant());
                if (!files.TryAdd(path, file) && files[path] != file) throw new InvalidDataException("Hub pagination returned conflicting file metadata.");
                if (files.Count > 100_000) throw new InvalidDataException("Repository exceeds the 100,000-file browsing limit.");
            }
            page = HubHttp.NextPage(treeResponse, treePath);
        }
        var readme = "";
        var readmeFile = files.Values.FirstOrDefault(file => file.Path.Equals("README.md", StringComparison.OrdinalIgnoreCase));
        if (readmeFile is not null)
        {
            try
            {
            using var readmeResponse = await HubHttp.SendAsync(httpClient, credentials,
                new Uri($"https://huggingface.co/{encoded}/resolve/{sha}/{HubHttp.EncodedPath(readmeFile.Path)}"), cancellationToken, allowCdn: true).ConfigureAwait(false);
            if (readmeResponse.StatusCode != HttpStatusCode.NotFound)
            {
                HubHttp.EnsureSuccess(readmeResponse);
                var raw = await HubHttp.ReadBoundedAsync(readmeResponse.Content, 128 * 1024, cancellationToken, truncate: true).ConfigureAwait(false);
                readme = Encoding.UTF8.GetString(raw.AsSpan(0, Math.Min(raw.Length, 128 * 1024)));
                readme = new string(readme.Where(character => !char.IsControl(character) || character is '\r' or '\n' or '\t').ToArray());
                if (raw.Length > 128 * 1024) readme += "\n\n[README truncated at 128 KiB.]";
            }
            }
            catch (HttpRequestException) { readme = "README is currently unavailable. The repository file list is still available."; }
            catch (IOException) { readme = "README could not be read. The repository file list is still available."; }
            catch (InvalidDataException) { readme = "README did not meet the text download checks. The repository file list is still available."; }
        }
        return new HubModelDetail(resolvedId, sha, files.Values.OrderBy(file => file.Path, StringComparer.Ordinal).ToArray(), readme,
            Text(model, "pipeline_tag"), Text(model, "library_name"), Boolean(model, "private"), Gated(model),
            Number(model, "downloads"), (int)Math.Clamp(Number(model, "likes"), 0, int.MaxValue));
    }

    private static string? Text(JsonElement value, string key) => value.TryGetProperty(key, out var item) && item.ValueKind == JsonValueKind.String ? item.GetString() : null;
    private static bool Boolean(JsonElement value, string key) => value.TryGetProperty(key, out var item) && item.ValueKind == JsonValueKind.True;
    private static long Number(JsonElement value, string key) => value.TryGetProperty(key, out var item) && item.TryGetInt64(out var number) ? Math.Max(0, number) : 0;
    private static bool Gated(JsonElement value) => value.TryGetProperty("gated", out var item) && (item.ValueKind == JsonValueKind.True || item.ValueKind == JsonValueKind.String && item.GetString() is not (null or "false"));
    private static long RequiredSize(JsonElement value) => value.TryGetProperty("size", out var item) && item.TryGetInt64(out var size) && size is >= 0 and <= 8_796_093_022_208
        ? size : throw new InvalidDataException("Hub file size is invalid.");
}
