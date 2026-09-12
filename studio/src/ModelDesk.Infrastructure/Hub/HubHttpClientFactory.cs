using System.Net;

namespace ModelDesk.Infrastructure.Hub;

/// <summary>Shared pooled transport; credentials and redirect checks stay in HubHttp.</summary>
public static class HubHttpClientFactory
{
    public static HttpClient Create() => new(new SocketsHttpHandler
    {
        AllowAutoRedirect = false,
        AutomaticDecompression = DecompressionMethods.None,
        MaxConnectionsPerServer = 8,
        PooledConnectionLifetime = TimeSpan.FromMinutes(10),
        PooledConnectionIdleTimeout = TimeSpan.FromMinutes(2),
        ConnectTimeout = TimeSpan.FromSeconds(20),
        MaxResponseHeadersLength = 64,
        MaxResponseDrainSize = 0,
        KeepAlivePingDelay = TimeSpan.FromSeconds(30),
        KeepAlivePingTimeout = TimeSpan.FromSeconds(10),
        KeepAlivePingPolicy = HttpKeepAlivePingPolicy.WithActiveRequests
    })
    {
        Timeout = Timeout.InfiniteTimeSpan,
        DefaultRequestVersion = HttpVersion.Version20,
        DefaultVersionPolicy = HttpVersionPolicy.RequestVersionOrLower
    };
}
