using System.Collections.Concurrent;
using System.Diagnostics;
using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text.Json;
using ModelDesk.Core;
using ModelDesk.Infrastructure.Downloads;

namespace ModelDesk.Tests;

public sealed class SegmentedDownloadTests : IDisposable
{
    private const int MiB = 1024 * 1024;
    private readonly string _root = Path.Combine(Path.GetTempPath(), "modeldesk-segments-" + Guid.NewGuid().ToString("N"));
    public SegmentedDownloadTests() => Directory.CreateDirectory(_root);
    public void Dispose() => Directory.Delete(_root, recursive: true);

    private DownloadRequest Request(byte[] data, string path = "weights.bin") => new("example/model", HubClientTests.Revision,
        new HubFile(path, data.Length, Convert.ToHexString(SHA256.HashData(data)).ToLowerInvariant()), _root);

    private static ModelDownloader Downloader(HttpClient client, int connections = 4) => new(client, new HubTestCredentials())
    { ConnectionsPerFile = connections, ParallelThresholdBytes = MiB, SegmentSizeBytes = MiB };

    private static byte[] Data(int mebibytes) => Enumerable.Range(0, mebibytes * MiB).Select(index => (byte)((index * 29 + index / 997) % 251)).ToArray();

    [Fact]
    public async Task ExactParallelRangesWriteOnePreallocatedFileAndVerifyItsCompleteHash()
    {
        var data = Data(4);
        var server = new RangeServer(data) { ReadDelay = TimeSpan.FromMilliseconds(2) };
        using var handler = new HubTestHandler(server.HandleAsync);
        using var client = new HttpClient(handler);
        var result = await Downloader(client).DownloadAsync(Request(data));
        Assert.True(result.HashVerified);
        Assert.Equal(data, await File.ReadAllBytesAsync(result.FilePath));
        Assert.Equal(0, result.ResumedBytes);
        var ranges = server.Requests.Where(item => item.End != 0).OrderBy(item => item.Start).ToArray();
        Assert.Equal(4, ranges.Length);
        for (var index = 0; index < 4; index++)
        {
            Assert.Equal(index * (long)MiB, ranges[index].Start);
            Assert.Equal((index + 1L) * MiB - 1, ranges[index].End);
            Assert.Equal("\"stable\"", ranges[index].IfRange);
        }
        Assert.InRange(server.PeakActive, 2, 4);
        Assert.InRange(server.MaximumRead, 1, 128 * 1024);
        Assert.Equal(0, server.Active);
        Assert.False(File.Exists(result.FilePath + ".part.json"));
        Assert.All(handler.Calls, call => Assert.Contains(HubClientTests.Revision, call.Uri.AbsolutePath));
    }

    [Fact]
    public async Task PausePersistsOnlyFlushedOffsetsAndResumeFetchesExactlyMissingSegmentTails()
    {
        var data = Data(8);
        var server = new RangeServer(data) { ReadDelay = TimeSpan.FromMilliseconds(16) };
        using var handler = new HubTestHandler(server.HandleAsync);
        using var client = new HttpClient(handler);
        using var cancel = new CancellationTokenSource();
        var progress = new ConcurrentQueue<DownloadProgress>();
        var downloader = Downloader(client);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => downloader.DownloadAsync(Request(data),
            new InlineProgress<DownloadProgress>(value =>
            {
                progress.Enqueue(value);
                if (value.Stage == "Downloading segments" && value.DownloadedBytes >= MiB) cancel.Cancel();
            }), cancel.Token));
        var part = Path.Combine(_root, "weights.bin.part");
        Assert.Equal(data.Length, new FileInfo(part).Length); // Allocation is not completion.
        using var manifest = JsonDocument.Parse(await File.ReadAllBytesAsync(part + ".json"));
        Assert.Equal(2, manifest.RootElement.GetProperty("Version").GetInt32());
        var completed = manifest.RootElement.GetProperty("Segments").EnumerateArray()
            .Sum(segment => segment.GetProperty("CompletedBytes").GetInt64());
        Assert.InRange(completed, 1, data.Length - 1);
        Assert.Equal(completed, progress.Last().DownloadedBytes);
        Assert.DoesNotContain(progress, value => value.Stage == "Completed");
        var estimate = await downloader.ValidateBatchAsync([Request(data)]);
        Assert.Equal(16 * MiB, estimate.Volumes.Single().RequiredBytes); // Physical allocation already holds its disk space.
        var before = server.Requests.Count;
        server.ReadDelay = TimeSpan.Zero;
        downloader.ConnectionsPerFile = 2;
        downloader.SegmentSizeBytes = 2 * MiB; // Resume preserves the saved geometry.
        var result = await downloader.DownloadAsync(Request(data));
        Assert.Equal(completed, result.ResumedBytes);
        var remaining = server.Requests.Skip(before).Where(item => item.End != 0).Sum(item => item.End!.Value - item.Start!.Value + 1);
        Assert.Equal(data.Length, completed + remaining);
        Assert.True(result.HashVerified);
        Assert.Equal(data, await File.ReadAllBytesAsync(result.FilePath));
    }

    [Fact]
    public async Task IgnoredWorkerRangeSettlesEverySiblingBeforeSequentialRestart()
    {
        var data = Data(4);
        var server = new RangeServer(data) { ReadDelay = TimeSpan.FromMilliseconds(3) };
        var sequentialOverlap = false;
        using var handler = new HubTestHandler((request, cancellation, number) =>
        {
            var range = request.Headers.Range?.Ranges.SingleOrDefault();
            if (range is { From: 0 } && range.To > 0) return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                { Content = new ByteArrayContent(data) });
            if (range is null) sequentialOverlap = server.Active != 0;
            return server.HandleAsync(request, cancellation, number);
        });
        using var client = new HttpClient(handler);
        var result = await Downloader(client).DownloadAsync(Request(data));
        Assert.False(sequentialOverlap);
        Assert.Single(handler.Calls, call => call.Range is null);
        Assert.True(result.HashVerified);
        Assert.Equal(data.Length, new FileInfo(result.FilePath).Length);
        Assert.Equal(data, await File.ReadAllBytesAsync(result.FilePath));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task InvalidRangeOrEtagCancelsActiveSiblingsAndNeverPublishes(bool wrongRange)
    {
        var data = Data(4);
        var server = new RangeServer(data) { ReadDelay = TimeSpan.FromMilliseconds(30) };
        using var handler = new HubTestHandler(async (request, cancellation, number) =>
        {
            if (request.Headers.Range?.Ranges.SingleOrDefault()?.From == MiB)
            {
                await Task.Delay(50, cancellation);
                var response = await server.HandleAsync(request, cancellation, number);
                if (wrongRange) response.Content.Headers.ContentRange = new ContentRangeHeaderValue(MiB + 1, 2 * MiB - 1, data.Length);
                else response.Headers.ETag = new EntityTagHeaderValue("\"changed\"");
                return response;
            }
            return await server.HandleAsync(request, cancellation, number);
        });
        using var client = new HttpClient(handler);
        await Assert.ThrowsAsync<InvalidDataException>(() => Downloader(client).DownloadAsync(Request(data)));
        Assert.False(File.Exists(Path.Combine(_root, "weights.bin")));
        Assert.Equal(0, server.Active);
        Assert.True(server.CancelledReads > 0);
        Assert.DoesNotContain(handler.Calls, call => call.Range is null);
    }

    [Fact]
    public async Task TransientSegmentRetryHonorsRetryAfterWithoutRestartingSuccessfulSegments()
    {
        var data = Data(4);
        var server = new RangeServer(data);
        var retryTimes = new ConcurrentQueue<long>();
        var timer = Stopwatch.StartNew();
        using var handler = new HubTestHandler((request, cancellation, number) =>
        {
            if (request.Headers.Range?.Ranges.SingleOrDefault()?.From == MiB)
            {
                retryTimes.Enqueue(timer.ElapsedMilliseconds);
                if (retryTimes.Count == 1)
                {
                    var unavailable = new HttpResponseMessage(HttpStatusCode.ServiceUnavailable);
                    unavailable.Headers.RetryAfter = new RetryConditionHeaderValue(TimeSpan.FromSeconds(1));
                    return Task.FromResult(unavailable);
                }
            }
            return server.HandleAsync(request, cancellation, number);
        });
        using var client = new HttpClient(handler);
        var result = await Downloader(client).DownloadAsync(Request(data));
        Assert.True(result.HashVerified);
        Assert.Equal(2, retryTimes.Count);
        Assert.True(retryTimes.Last() - retryTimes.First() >= 950);
        Assert.Equal(4, server.Requests.Count(item => item.End != 0));
    }

    [Fact]
    public async Task GlobalEightConnectionLimitIncludesSeparateDownloaderInstances()
    {
        var data = Data(4);
        var server = new RangeServer(data) { ReadDelay = TimeSpan.FromMilliseconds(10) };
        using var handler = new HubTestHandler(server.HandleAsync);
        using var client = new HttpClient(handler);
        await Task.WhenAll(Enumerable.Range(0, 3).Select(index => Downloader(client).DownloadAsync(Request(data, $"file-{index}.bin"))));
        Assert.InRange(server.PeakActive, 4, 8);
        Assert.Equal(0, server.Active);
    }

    [Fact]
    public async Task CachedCdnExpirationRefreshesOriginAndKeepsBearerOffEveryCdnRequest()
    {
        var data = Data(4);
        var server = new RangeServer(data);
        var origins = 0;
        using var handler = new HubTestHandler((request, cancellation, number) =>
        {
            if (request.RequestUri!.Host == "huggingface.co")
            {
                var origin = Interlocked.Increment(ref origins);
                var redirect = new HttpResponseMessage(HttpStatusCode.Redirect);
                redirect.Headers.Location = new Uri($"https://cdn-lfs.hf.co/temporary?signature={(origin == 1 ? "expired" : "fresh")}");
                return Task.FromResult(redirect);
            }
            if (request.RequestUri.Query.Contains("expired", StringComparison.Ordinal) && request.Headers.Range!.Ranges.Single().To != 0)
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.Forbidden));
            return server.HandleAsync(request, cancellation, number);
        });
        using var client = new HttpClient(handler);
        var downloader = new ModelDownloader(client, new HubTestCredentials("hf_test_only"))
            { ParallelThresholdBytes = MiB, SegmentSizeBytes = MiB };
        var result = await downloader.DownloadAsync(Request(data));
        Assert.True(result.HashVerified);
        Assert.InRange(origins, 2, 5);
        Assert.All(handler.Calls.Where(call => call.Uri.Host != "huggingface.co"), call => Assert.Null(call.Authorization));
        Assert.All(handler.Calls.Where(call => call.Uri.Host == "huggingface.co"), call => Assert.Equal("Bearer hf_test_only", call.Authorization));
    }

    [Fact]
    public async Task RepeatedSameSpeedSettingDoesNotStarveTheSharedLimiter()
    {
        var data = Data(1);
        var server = new RangeServer(data);
        using var handler = new HubTestHandler(server.HandleAsync);
        using var client = new HttpClient(handler);
        var downloader = Downloader(client);
        downloader.BytesPerSecondLimit = 2 * MiB;
        var timer = Stopwatch.StartNew();
        var transfer = downloader.DownloadAsync(Request(data));
        while (!transfer.IsCompleted && timer.Elapsed < TimeSpan.FromSeconds(5))
        {
            downloader.BytesPerSecondLimit = 2 * MiB;
            await Task.Delay(2);
        }
        var result = await transfer.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.True(result.HashVerified);
        Assert.True(timer.Elapsed < TimeSpan.FromSeconds(5));
        Assert.True(timer.Elapsed >= TimeSpan.FromMilliseconds(400));
    }

    private sealed record RangeCall(long? Start, long? End, string? IfRange);

    private sealed class RangeServer(byte[] data)
    {
        private readonly byte[] _data = data;
        private int _active, _peak, _maximumRead, _cancelledReads;
        internal ConcurrentQueue<RangeCall> Requests { get; } = new();
        internal TimeSpan ReadDelay { get; set; }
        internal int Active => Volatile.Read(ref _active);
        internal int PeakActive => Volatile.Read(ref _peak);
        internal int MaximumRead => Volatile.Read(ref _maximumRead);
        internal int CancelledReads => Volatile.Read(ref _cancelledReads);

        internal Task<HttpResponseMessage> HandleAsync(HttpRequestMessage request, CancellationToken _, int number)
        {
            var range = request.Headers.Range?.Ranges.SingleOrDefault();
            var start = range?.From ?? 0;
            var end = range?.To ?? _data.LongLength - 1;
            Requests.Enqueue(new RangeCall(range?.From, range?.To, request.Headers.IfRange?.ToString()));
            var active = Interlocked.Increment(ref _active);
            Maximum(ref _peak, active);
            var response = new HttpResponseMessage(range is null ? HttpStatusCode.OK : HttpStatusCode.PartialContent)
            { Content = new StreamContent(new RangeBody(this, start, end + 1)) };
            response.Headers.ETag = new EntityTagHeaderValue("\"stable\"");
            response.Content.Headers.ContentLength = end - start + 1;
            if (range is not null) response.Content.Headers.ContentRange = new ContentRangeHeaderValue(start, end, _data.Length);
            return Task.FromResult(response);
        }

        private static void Maximum(ref int location, int value)
        {
            var before = Volatile.Read(ref location);
            while (before < value)
            {
                var observed = Interlocked.CompareExchange(ref location, value, before);
                if (observed == before) return;
                before = observed;
            }
        }

        private sealed class RangeBody(RangeServer owner, long start, long end) : Stream
        {
            private long _position = start;
            private readonly long _start = start;
            private int _disposed;
            public override bool CanRead => true;
            public override bool CanSeek => false;
            public override bool CanWrite => false;
            public override long Length => end - _start;
            public override long Position { get => _position - _start; set => throw new NotSupportedException(); }
            public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
            {
                Maximum(ref owner._maximumRead, buffer.Length);
                try
                {
                    if (_position >= end) return 0;
                    if (owner.ReadDelay > TimeSpan.Zero && end - _start > 1) await Task.Delay(owner.ReadDelay, cancellationToken);
                    cancellationToken.ThrowIfCancellationRequested();
                    var count = (int)Math.Min(buffer.Length, end - _position);
                    owner._data.AsMemory((int)_position, count).CopyTo(buffer);
                    _position += count;
                    return count;
                }
                catch (OperationCanceledException) { Interlocked.Increment(ref owner._cancelledReads); throw; }
            }
            protected override void Dispose(bool disposing)
            {
                if (Interlocked.Exchange(ref _disposed, 1) == 0) Interlocked.Decrement(ref owner._active);
                base.Dispose(disposing);
            }
            public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
            public override void Flush() => throw new NotSupportedException();
            public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
            public override void SetLength(long value) => throw new NotSupportedException();
            public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        }
    }
}
