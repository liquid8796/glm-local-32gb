using System.Net;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Text.RegularExpressions;
using ModelDesk.Core;

namespace ModelDesk.Infrastructure.Hub;

internal static partial class HubHttp
{
    internal static string ModelId(string value)
    {
        if (string.IsNullOrWhiteSpace(value) || value.Length > 193 || value.Split('/').Length is < 1 or > 2 ||
            value.Split('/').Any(part => !RepoSegment().IsMatch(part) || part.Contains("..", StringComparison.Ordinal) ||
                part.Contains("--", StringComparison.Ordinal) || part.EndsWith('.') || part.EndsWith('-')))
            throw new ArgumentException("Use a Hugging Face model ID such as organization/model.");
        return value;
    }

    internal static string ImmutableRevision(string value) => Sha40().IsMatch(value ?? "")
        ? value!.ToLowerInvariant() : throw new ArgumentException("Downloads require an immutable 40-character commit revision.");

    internal static string Revision(string value)
    {
        if (string.IsNullOrWhiteSpace(value) || value.Length > 256 || value.Any(char.IsControl))
            throw new ArgumentException("Invalid model revision.");
        return value;
    }

    internal static string EncodedModel(string value) => string.Join('/', ModelId(value).Split('/').Select(Uri.EscapeDataString));
    internal static string EncodedPath(string value) => string.Join('/', value.Split('/').Select(Uri.EscapeDataString));
    internal static bool IsSha256(string? value) => value is not null && Sha64().IsMatch(value);
    internal static bool IsSha1(string? value) => value is not null && Sha40().IsMatch(value);

    internal static Uri ApiUri(string value, string requiredPath)
    {
        if (!Uri.TryCreate(value, UriKind.Absolute, out var uri) || !IsHub(uri) ||
            !uri.AbsolutePath.Equals(requiredPath, StringComparison.Ordinal))
            throw new InvalidDataException("The Hub returned an invalid pagination link.");
        return uri;
    }

    internal static bool IsHub(Uri uri) => uri.Scheme == Uri.UriSchemeHttps && uri.Host.Equals("huggingface.co", StringComparison.OrdinalIgnoreCase)
        && uri.IsDefaultPort && string.IsNullOrEmpty(uri.UserInfo) && string.IsNullOrEmpty(uri.Fragment);

    private static bool IsDownloadOrigin(Uri uri) => uri.Scheme == Uri.UriSchemeHttps && uri.IsDefaultPort &&
        string.IsNullOrEmpty(uri.UserInfo) && string.IsNullOrEmpty(uri.Fragment) &&
        (uri.Host.Equals("huggingface.co", StringComparison.OrdinalIgnoreCase) ||
         new[] { ".huggingface.co", ".hf.co", ".hfusercontent.com", ".amazonaws.com", ".cloudfront.net" }
             .Any(suffix => uri.Host.EndsWith(suffix, StringComparison.OrdinalIgnoreCase)));

    // Caller configures AllowAutoRedirect=false. Never set Authorization on HttpClient defaults.
    internal static async Task<HttpResponseMessage> SendAsync(HttpClient client, ICredentialStore credentials,
        Uri uri, CancellationToken cancellationToken, long? offset = null, string? etag = null, bool allowCdn = false,
        long? rangeEnd = null)
    {
        if (client.DefaultRequestHeaders.Authorization is not null)
            throw new InvalidOperationException("Use per-request credentials, not a default Authorization header.");
        for (var redirect = 0; redirect <= 8; redirect++)
        {
            if (!(allowCdn ? IsDownloadOrigin(uri) : IsHub(uri)))
                throw new InvalidDataException("The Hub returned an unsupported redirect origin.");
            using var request = new HttpRequestMessage(HttpMethod.Get, uri);
            request.Version = HttpVersion.Version20;
            request.VersionPolicy = HttpVersionPolicy.RequestVersionOrLower;
            request.Headers.AcceptEncoding.Add(new StringWithQualityHeaderValue("identity"));
            request.Headers.UserAgent.ParseAdd("ModelDesk/1.0");
            if (IsHub(uri))
            {
                var token = await credentials.GetTokenAsync(cancellationToken).ConfigureAwait(false);
                if (!string.IsNullOrWhiteSpace(token))
                {
                    if (token.Any(char.IsWhiteSpace) || token.Any(char.IsControl)) throw new InvalidOperationException("Stored Hub token has an invalid format.");
                    request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                }
            }
            if (offset is > 0 || rangeEnd is not null)
            {
                request.Headers.Range = new RangeHeaderValue(offset ?? 0, rangeEnd);
                if (etag is not null) request.Headers.IfRange = new RangeConditionHeaderValue(EntityTagHeaderValue.Parse(etag));
            }
            HttpResponseMessage response;
            using var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            deadline.CancelAfter(TimeSpan.FromSeconds(60));
            try { response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, deadline.Token).ConfigureAwait(false); }
            catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested) { throw new HttpRequestException("Hugging Face request timed out."); }
            catch (HttpRequestException error) { throw new HttpRequestException("Hugging Face network request failed.", null, error.StatusCode); }
            if (response.StatusCode is HttpStatusCode.Moved or HttpStatusCode.Redirect or HttpStatusCode.RedirectMethod or
                HttpStatusCode.TemporaryRedirect or HttpStatusCode.PermanentRedirect)
            {
                var location = response.Headers.Location;
                response.Dispose();
                if (location is null) throw new InvalidDataException("The Hub redirect did not include a location.");
                uri = location.IsAbsoluteUri ? location : new Uri(uri, location);
                continue;
            }
            if (response.RequestMessage?.RequestUri is { } final && !(allowCdn ? IsDownloadOrigin(final) : IsHub(final)))
            {
                response.Dispose();
                throw new InvalidDataException("The HTTP client followed an unsupported redirect.");
            }
            return response;
        }
        throw new InvalidDataException("The Hub exceeded the redirect limit.");
    }

    internal static void EnsureSuccess(HttpResponseMessage response)
    {
        if (response.IsSuccessStatusCode) return;
        var message = response.StatusCode switch
        {
            HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden => "Hub access denied. Check your token and the model's access requirements.",
            HttpStatusCode.NotFound => "The model, revision or file was not found on Hugging Face.",
            (HttpStatusCode)429 => "Hugging Face rate limit reached. Try again shortly.",
            _ => $"Hugging Face returned HTTP {(int)response.StatusCode}."
        };
        throw new HttpRequestException(message, null, response.StatusCode);
    }

    internal static async Task<byte[]> ReadBoundedAsync(HttpContent content, int maximumBytes,
        CancellationToken cancellationToken, bool truncate = false)
    {
        if (!truncate && content.Headers.ContentLength > maximumBytes) throw new InvalidDataException("Hub metadata exceeds the response size limit.");
        await using var stream = await content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        using var output = new MemoryStream();
        var buffer = new byte[64 * 1024];
        while (output.Length <= maximumBytes)
        {
            var count = await ReadWithTimeoutAsync(stream, buffer.AsMemory(0, (int)Math.Min(buffer.Length, maximumBytes + 1L - output.Length)), cancellationToken).ConfigureAwait(false);
            if (count == 0) break;
            output.Write(buffer, 0, count);
        }
        if (output.Length > maximumBytes && !truncate) throw new InvalidDataException("Hub metadata exceeds the response size limit.");
        return output.ToArray();
    }

    internal static async ValueTask<int> ReadWithTimeoutAsync(Stream stream, Memory<byte> buffer, CancellationToken cancellationToken)
    {
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(TimeSpan.FromSeconds(60));
        try { return await stream.ReadAsync(buffer, timeout.Token).ConfigureAwait(false); }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested) { throw new IOException("The transfer stalled for 60 seconds."); }
    }

    internal static async Task<JsonDocument> JsonAsync(HttpContent content, CancellationToken cancellationToken) =>
        JsonDocument.Parse(await ReadBoundedAsync(content, 8 * 1024 * 1024, cancellationToken).ConfigureAwait(false), new JsonDocumentOptions { MaxDepth = 64 });

    internal static string? NextPage(HttpResponseMessage response, string requiredPath)
    {
        if (!response.Headers.TryGetValues("Link", out var links)) return null;
        foreach (var link in links)
        foreach (Match match in LinkPattern().Matches(link))
            if (match.Groups[2].Value.Split(' ', StringSplitOptions.RemoveEmptyEntries).Contains("next", StringComparer.OrdinalIgnoreCase))
                return ApiUri(match.Groups[1].Value, requiredPath).AbsoluteUri;
        return null;
    }

    [GeneratedRegex("^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$", RegexOptions.CultureInvariant)] private static partial Regex RepoSegment();
    [GeneratedRegex("^[a-fA-F0-9]{40}$", RegexOptions.CultureInvariant)] private static partial Regex Sha40();
    [GeneratedRegex("^[a-fA-F0-9]{64}$", RegexOptions.CultureInvariant)] private static partial Regex Sha64();
    [GeneratedRegex("<([^>]+)>\\s*;\\s*rel=\"([^\"]+)\"", RegexOptions.CultureInvariant)] private static partial Regex LinkPattern();
}
