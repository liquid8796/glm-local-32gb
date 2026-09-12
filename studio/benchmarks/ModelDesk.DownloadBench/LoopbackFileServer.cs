using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;

namespace ModelDesk.DownloadBench;

internal sealed record RequestMeasurement(string Method, long Start, long End, long BodyBytesWritten, bool Finished);
internal sealed record ServerMeasurement(long BodyBytesWritten, int PeakDataConnections, IReadOnlyList<RequestMeasurement> Requests);

internal sealed class LoopbackFileServer : IAsyncDisposable
{
    private readonly TcpListener _listener = new(IPAddress.Loopback, 0);
    private readonly CancellationTokenSource _stop = new();
    private readonly ConcurrentBag<Task> _handlers = [];
    private readonly ConcurrentQueue<RequestMeasurement> _requests = new();
    private readonly SemaphoreSlim _slots = new(16);
    private readonly long _size, _rate;
    private readonly int _latency;
    private readonly string _etag;
    private Task? _accept;
    private long _bytes;
    private int _activeData, _peakData;

    public LoopbackFileServer(long size, long bytesPerConnectionSecond, int headerLatencyMilliseconds, string sha256)
    {
        _size = size; _rate = bytesPerConnectionSecond; _latency = headerLatencyMilliseconds;
        _etag = "\"" + sha256 + "\"";
        _listener.Start(16);
        Endpoint = new Uri($"http://127.0.0.1:{((IPEndPoint)_listener.LocalEndpoint).Port}/");
        _accept = AcceptAsync();
    }

    public Uri Endpoint { get; }
    public ServerMeasurement Snapshot() => new(Interlocked.Read(ref _bytes), Volatile.Read(ref _peakData), _requests.ToArray());

    public async Task WaitIdleAsync(CancellationToken cancellationToken = default)
    {
        while (_handlers.Any(task => !task.IsCompleted)) await Task.Delay(10, cancellationToken);
    }

    private async Task AcceptAsync()
    {
        try
        {
            while (!_stop.IsCancellationRequested)
            {
                var client = await _listener.AcceptTcpClientAsync(_stop.Token);
                await _slots.WaitAsync(_stop.Token);
                _handlers.Add(HandleAsync(client));
            }
        }
        catch (OperationCanceledException) when (_stop.IsCancellationRequested) { }
        catch (SocketException) when (_stop.IsCancellationRequested) { }
    }

    private async Task HandleAsync(TcpClient client)
    {
        string method = "UNKNOWN";
        long start = 0, end = _size - 1, written = 0;
        var finished = false;
        var dataConnection = false;
        try
        {
            using (client)
            {
                client.NoDelay = true;
                await using var stream = client.GetStream();
                var header = await ReadHeaderAsync(stream, _stop.Token);
                var lines = header.Split("\r\n", StringSplitOptions.None);
                var first = lines[0].Split(' ');
                if (first.Length != 3 || first[0] is not ("GET" or "HEAD")) throw new InvalidDataException("Benchmark supports GET/HEAD only.");
                method = first[0];
                var headers = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                foreach (var line in lines.Skip(1))
                {
                    var colon = line.IndexOf(':');
                    if (colon > 0) headers[line[..colon]] = line[(colon + 1)..].Trim();
                }
                var partial = headers.TryGetValue("Range", out var range);
                if (partial)
                {
                    if (range is null || !range.StartsWith("bytes=", StringComparison.Ordinal)) throw new InvalidDataException("Invalid Range unit.");
                    var parts = range[6..].Split('-');
                    if (parts.Length != 2 || !long.TryParse(parts[0], NumberStyles.None, CultureInfo.InvariantCulture, out start) ||
                        parts[1].Length > 0 && !long.TryParse(parts[1], NumberStyles.None, CultureInfo.InvariantCulture, out end))
                        throw new InvalidDataException("Invalid Range bounds.");
                    if (start < 0 || start >= _size || end < start || end >= _size) throw new InvalidDataException("Range exceeds deterministic file.");
                }
                if (headers.TryGetValue("If-Match", out var match) && match != _etag) throw new InvalidDataException("If-Match does not match benchmark ETag.");
                if (headers.TryGetValue("If-Range", out var condition) && condition != _etag)
                { partial = false; start = 0; end = _size - 1; }
                var length = end - start + 1;
                await Task.Delay(_latency, _stop.Token);
                var response = $"HTTP/1.1 {(partial ? "206 Partial Content" : "200 OK")}\r\nContent-Length: {length}\r\n" +
                    (partial ? $"Content-Range: bytes {start}-{end}/{_size}\r\n" : "") +
                    $"ETag: {_etag}\r\nAccept-Ranges: bytes\r\nContent-Type: application/octet-stream\r\nConnection: close\r\n\r\n";
                await stream.WriteAsync(Encoding.ASCII.GetBytes(response), _stop.Token);
                if (method == "HEAD") { finished = true; return; }
                dataConnection = length > 1;
                if (dataConnection)
                {
                    var active = Interlocked.Increment(ref _activeData);
                    int peak;
                    do { peak = Volatile.Read(ref _peakData); }
                    while (active > peak && Interlocked.CompareExchange(ref _peakData, active, peak) != peak);
                }
                var buffer = new byte[65536];
                var timer = Stopwatch.StartNew();
                while (written < length)
                {
                    var count = (int)Math.Min(buffer.Length, length - written);
                    Fill(buffer.AsSpan(0, count), start + written);
                    var wait = (written + count) / (double)_rate - timer.Elapsed.TotalSeconds;
                    if (wait > 0) await Task.Delay(TimeSpan.FromSeconds(wait), _stop.Token);
                    await stream.WriteAsync(buffer.AsMemory(0, count), _stop.Token);
                    written += count;
                    var total = Interlocked.Add(ref _bytes, count);
                    if (total > _size * 8 + 1024 * 1024) throw new IOException("Benchmark server byte guard exceeded.");
                }
                finished = true;
            }
        }
        catch (Exception exception) when (exception is IOException or SocketException or OperationCanceledException) { }
        finally
        {
            if (dataConnection) Interlocked.Decrement(ref _activeData);
            _requests.Enqueue(new(method, start, end, written, finished));
            _slots.Release();
        }
    }

    private static async Task<string> ReadHeaderAsync(NetworkStream stream, CancellationToken cancellationToken)
    {
        var buffer = new byte[16384];
        var count = 0;
        while (count < buffer.Length)
        {
            var read = await stream.ReadAsync(buffer.AsMemory(count), cancellationToken);
            if (read == 0) throw new IOException("Client closed before request headers.");
            count += read;
            if (buffer.AsSpan(0, count).IndexOf("\r\n\r\n"u8) >= 0) return Encoding.ASCII.GetString(buffer, 0, count);
        }
        throw new InvalidDataException("Benchmark request headers exceed 16 KiB.");
    }

    public static void Fill(Span<byte> buffer, long offset)
    {
        for (var index = 0; index < buffer.Length; index++)
        {
            var position = offset + index;
            buffer[index] = (byte)((position * 31 + (position >> 8) ^ (position >> 16) * 17) & 255);
        }
    }

    public static string ExpectedSha256(long bytes)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var buffer = new byte[65536];
        for (long offset = 0; offset < bytes; offset += buffer.Length)
        {
            var count = (int)Math.Min(buffer.Length, bytes - offset);
            Fill(buffer.AsSpan(0, count), offset);
            hash.AppendData(buffer, 0, count);
        }
        return Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
    }

    public async ValueTask DisposeAsync()
    {
        await _stop.CancelAsync();
        _listener.Stop();
        if (_accept is not null) await _accept;
        await Task.WhenAll(_handlers.ToArray());
        _slots.Dispose();
        _stop.Dispose();
    }
}
