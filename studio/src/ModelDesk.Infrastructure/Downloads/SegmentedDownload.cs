using System.Buffers;
using System.Diagnostics;
using System.Net;
using System.Runtime.ExceptionServices;
using ModelDesk.Core;
using ModelDesk.Infrastructure.Hub;

namespace ModelDesk.Infrastructure.Downloads;

public sealed partial class ModelDownloader
{
    private const int SegmentBufferSize = 128 * 1024;
    private const long CheckpointInterval = 8 * 1024 * 1024;
    private readonly object _optionsSync = new();
    private int _connections = 4;
    private long _parallelThreshold = 64 * 1024 * 1024;
    private long _segmentSize = 16 * 1024 * 1024;

    public int ConnectionsPerFile
    {
        get { lock (_optionsSync) return _connections; }
        set { if (value is < 1 or > 4) throw new ArgumentOutOfRangeException(nameof(value)); lock (_optionsSync) _connections = value; }
    }
    public long ParallelThresholdBytes
    {
        get { lock (_optionsSync) return _parallelThreshold; }
        set { if (value is < 1_048_576 or > 8_796_093_022_208) throw new ArgumentOutOfRangeException(nameof(value)); lock (_optionsSync) _parallelThreshold = value; }
    }
    public long SegmentSizeBytes
    {
        get { lock (_optionsSync) return _segmentSize; }
        set { if (value is < 1_048_576 or > 268_435_456) throw new ArgumentOutOfRangeException(nameof(value)); lock (_optionsSync) _segmentSize = value; }
    }

    private DownloadOptions SnapshotOptions() { lock (_optionsSync) return new(_connections, _parallelThreshold, _segmentSize); }
    private sealed record DownloadOptions(int Connections, long Threshold, long SegmentBytes);
    private sealed record SegmentCheckpoint(long Start, long Length, long CompletedBytes);
    private sealed class RangeIgnoredException : Exception;

    private sealed class DownloadRetryException(HttpStatusCode status, TimeSpan? retryAfter)
        : HttpRequestException($"Download server returned HTTP {(int)status}.", null, status)
    {
        internal TimeSpan? RetryAfter { get; } = retryAfter;
    }

    private static void EnsureDownloadSuccess(HttpResponseMessage response)
    {
        if (response.StatusCode is (HttpStatusCode)429 or HttpStatusCode.ServiceUnavailable)
        {
            var retry = response.Headers.RetryAfter;
            TimeSpan? delay = retry?.Delta ?? (retry?.Date is { } date ? date - DateTimeOffset.UtcNow : null);
            if (delay < TimeSpan.Zero) delay = TimeSpan.Zero;
            throw new DownloadRetryException(response.StatusCode, delay);
        }
        HubHttp.EnsureSuccess(response);
    }

    private static async Task DelayRetryAsync(Exception error, int failure, CancellationToken cancellationToken)
    {
        var delay = TimeSpan.FromMilliseconds(500 * (1 << (failure - 1)) + Random.Shared.Next(0, 251));
        if (error is DownloadRetryException { RetryAfter: { } requested })
        {
            // Never retry earlier than Retry-After. A long server cooldown becomes a
            // resumable failure instead of tying up a worker for an unbounded interval.
            if (requested > TimeSpan.FromMinutes(2)) throw new IOException("The server requested a cooldown longer than two minutes. Resume this download later.");
            if (requested > delay) delay = requested;
        }
        await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
    }

    private sealed class RedirectCache
    {
        // Signed CDN URLs are ephemeral in-memory transport state, never persisted or logged.
        internal Uri? Value;
    }

    private async Task<HttpResponseMessage> SendRangeAsync(Uri origin, RedirectCache cache, long start, long end,
        string? etag, CancellationToken cancellationToken)
    {
        var cached = Volatile.Read(ref cache.Value);
        var response = await HubHttp.SendAsync(httpClient, credentials, cached ?? origin, cancellationToken,
            start, etag, allowCdn: true, rangeEnd: end).ConfigureAwait(false);
        if (cached is not null && response.StatusCode is HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden)
        {
            response.Dispose();
            Interlocked.CompareExchange(ref cache.Value, null, cached);
            response = await HubHttp.SendAsync(httpClient, credentials, origin, cancellationToken,
                start, etag, allowCdn: true, rangeEnd: end).ConfigureAwait(false);
        }
        if (response.IsSuccessStatusCode && response.RequestMessage?.RequestUri is { } resolved && !HubHttp.IsHub(resolved))
            Volatile.Write(ref cache.Value, resolved);
        return response;
    }

    private static string ValidateRange(HttpResponseMessage response, long start, long end, long total, string? expectedEtag)
    {
        EnsureDownloadSuccess(response);
        if (response.StatusCode == HttpStatusCode.OK) throw new RangeIgnoredException();
        var range = response.Content.Headers.ContentRange;
        if (response.StatusCode != HttpStatusCode.PartialContent || range is null || !range.HasRange || !range.HasLength ||
            range.Unit != "bytes" || range.From != start || range.To != end || range.Length != total ||
            response.Content.Headers.ContentLength is { } length && length != end - start + 1 ||
            response.Content.Headers.ContentEncoding.Any(value => !value.Equals("identity", StringComparison.OrdinalIgnoreCase)))
            throw new InvalidDataException("Segment response does not match its exact frozen byte range.");
        var tag = response.Headers.ETag is { IsWeak: false } strong ? strong.ToString() : null;
        if (tag is null && expectedEtag is null) throw new RangeIgnoredException();
        if (tag is null || expectedEtag is not null && tag != expectedEtag)
            throw new InvalidDataException("Segment response ETag differs from the frozen transfer identity.");
        return tag;
    }

    private async Task<string> ProbeRangeAsync(Uri origin, RedirectCache cache, long size, string? etag, CancellationToken cancellationToken)
    {
        for (var failure = 1; ; failure++)
        {
            try
            {
                using var connection = await DownloadConnections.AcquireAsync(cancellationToken).ConfigureAwait(false);
                using var response = await SendRangeAsync(origin, cache, 0, 0, etag, cancellationToken).ConfigureAwait(false);
                var confirmed = ValidateRange(response, 0, 0, size, etag);
                await using var body = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
                var single = new byte[1];
                await _rate.AcquireAsync(1, cancellationToken).ConfigureAwait(false);
                if (await HubHttp.ReadWithTimeoutAsync(body, single, cancellationToken).ConfigureAwait(false) != 1 ||
                    await HubHttp.ReadWithTimeoutAsync(body, single, cancellationToken).ConfigureAwait(false) != 0)
                    throw new InvalidDataException("The one-byte range probe returned an invalid body.");
                return confirmed;
            }
            catch (Exception error) when (IsTransient(error) && !cancellationToken.IsCancellationRequested)
            {
                if (failure >= MaximumAttempts) throw new IOException("Range negotiation failed after four attempts. Resume this download later.");
                await DelayRetryAsync(error, failure, cancellationToken).ConfigureAwait(false);
            }
        }
    }

    private async Task<DownloadResult?> TrySegmentedAsync(DownloadRequest request, string finalPath, string partPath,
        string statePath, TransferIdentity identity, DownloadOptions options, IProgress<DownloadProgress>? progress,
        Stopwatch timer, CancellationToken cancellationToken)
    {
        var origin = new Uri($"https://huggingface.co/{HubHttp.EncodedModel(request.ModelId)}/resolve/{HubHttp.ImmutableRevision(request.Revision)}/{HubHttp.EncodedPath(request.File.Path)}");
        var cache = new RedirectCache();
        string etag;
        try { etag = await ProbeRangeAsync(origin, cache, request.File.Size, identity.Version == 2 ? identity.ETag : null, cancellationToken).ConfigureAwait(false); }
        catch (RangeIgnoredException)
        {
            await ResetForSequentialAsync(partPath, statePath, request, cancellationToken).ConfigureAwait(false);
            return null;
        }
        if (identity.Version != 2 || identity.HashFailed)
        {
            // Adaptive larger pieces cap the manifest at4096 entries for very large files.
            var pieceSize = Math.Max(options.SegmentBytes, (request.File.Size + 4095) / 4096);
            var map = new List<SegmentCheckpoint>();
            for (long start = 0; start < request.File.Size; start += pieceSize)
                map.Add(new SegmentCheckpoint(start, Math.Min(pieceSize, request.File.Size - start), 0));
            identity = TransferIdentity.For(request) with { Version = 2, ETag = etag, Segments = map.ToArray() };
        }
        identity.ValidateSegments();
        var existingLength = File.Exists(partPath) ? new FileInfo(partPath).Length : 0;
        var resumed = identity.CompletedBytes(existingLength);
        var reporter = new Reporter(request.File, progress, resumed);
        reporter.Report(resumed, resumed > 0 ? "Resuming segments" : "Starting segments", true);
        var additionalDisk = request.File.Size - DiskReservations.AllocatedBytes(partPath);
        using var disk = DiskReservations.Acquire(finalPath, additionalDisk);
        // Publish an empty/previously flushed v2 map before extending the file.
        // A crash can never leave a v1 prefix identity describing preallocated holes.
        await SaveIdentityAsync(statePath, identity, cancellationToken).ConfigureAwait(false);
        var ignored = false;
        await using (var partial = new FileStream(partPath, new FileStreamOptions
        {
            // .NET only permits PreallocationSize with Create/CreateNew. An existing
            // nonempty v2 file must be opened intact; an identified empty file is safe
            // to create anew under the already-held exclusive transfer lock.
            Mode = File.Exists(partPath) ? existingLength == 0 ? FileMode.Create : FileMode.Open : FileMode.CreateNew,
            Access = FileAccess.ReadWrite, Share = FileShare.None,
            Options = FileOptions.Asynchronous | FileOptions.RandomAccess, BufferSize = 1,
            PreallocationSize = existingLength == 0 ? request.File.Size : 0
        }))
        {
            if (existingLength != 0 && partial.Length != request.File.Size)
                throw new InvalidDataException("Preallocated segment file length differs from its manifest.");
            partial.SetLength(request.File.Size);
            RandomAccess.FlushToDisk(partial.SafeFileHandle);
            // The actual allocation now holds the space; retaining the lease would
            // double-charge later transfers for bytes already reflected in free space.
            disk.Consumed(additionalDisk);
            await SaveIdentityAsync(statePath, identity, cancellationToken).ConfigureAwait(false);
            var session = new SegmentSession(partial, statePath, identity, reporter);
            using var siblings = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            Exception? fatal = null;
            var next = -1;
            async Task WorkerAsync()
            {
                try
                {
                    while (true)
                    {
                        var index = Interlocked.Increment(ref next);
                        if (index >= session.Segments.Length) return;
                        if (session.Completed(index) >= session.Segments[index].Length) continue;
                        await DownloadSegmentAsync(origin, cache, session, index, siblings.Token).ConfigureAwait(false);
                    }
                }
                catch (Exception error)
                {
                    if (error is not OperationCanceledException || !siblings.IsCancellationRequested)
                        Interlocked.CompareExchange(ref fatal, error, null);
                    siblings.Cancel();
                    throw;
                }
            }
            try
            {
                var workers = Enumerable.Range(0, Math.Min(options.Connections, session.Segments.Length)).Select(_ => WorkerAsync()).ToArray();
                await Task.WhenAll(workers).ConfigureAwait(false);
            }
            catch
            {
                // All workers have settled before publishing any final checkpoint or
                // restarting. A cancelled sibling can never write into a sequential retry.
                await session.CheckpointAsync(CancellationToken.None).ConfigureAwait(false);
                if (fatal is RangeIgnoredException) ignored = true;
                else
                {
                    reporter.Report(session.TotalCompleted, cancellationToken.IsCancellationRequested ? "Paused" : "Interrupted", true);
                    cancellationToken.ThrowIfCancellationRequested();
                    ExceptionDispatchInfo.Capture(fatal ?? new IOException("Segmented download was interrupted.")).Throw();
                }
            }
            if (!ignored)
            {
                await session.CheckpointAsync(cancellationToken).ConfigureAwait(false);
                if (session.TotalCompleted != request.File.Size) throw new InvalidDataException("Segment completion map still contains holes.");
                reporter.Report(request.File.Size, "Verifying", true);
                partial.Position = 0;
                if (!await VerifyAsync(partial, request.File, cancellationToken).ConfigureAwait(false))
                {
                    await SaveIdentityAsync(statePath, session.Snapshot() with { HashFailed = true }, cancellationToken).ConfigureAwait(false);
                    throw new InvalidDataException("Segmented content hash verification failed. Retry restarts the identified partial file.");
                }
            }
        }
        if (ignored)
        {
            await ResetForSequentialAsync(partPath, statePath, request, cancellationToken).ConfigureAwait(false);
            return null;
        }
        cancellationToken.ThrowIfCancellationRequested();
        foreach (var path in new[] { finalPath, partPath, statePath }) DownloadPaths.RegularFile(path);
        File.Move(partPath, finalPath, overwrite: false);
        try { File.Delete(statePath); } catch (IOException) { }
        progress?.Report(new DownloadProgress(request.File.Path, request.File.Size, request.File.Size,
            timer.Elapsed.TotalSeconds > 0 ? (request.File.Size - resumed) / timer.Elapsed.TotalSeconds : 0, TimeSpan.Zero, "Completed"));
        return new DownloadResult(finalPath, request.File.Size, true, false, resumed, timer.Elapsed);
    }

    private async Task DownloadSegmentAsync(Uri origin, RedirectCache cache, SegmentSession session, int index, CancellationToken cancellationToken)
    {
        var segment = session.Segments[index];
        var failures = 0;
        var sinceCheckpoint = 0L;
        var buffer = ArrayPool<byte>.Shared.Rent(SegmentBufferSize);
        try
        {
            while (session.Completed(index) < segment.Length)
            {
                try
                {
                    using var connection = await DownloadConnections.AcquireAsync(cancellationToken).ConfigureAwait(false);
                    var start = segment.Start + session.Completed(index);
                    var end = segment.Start + segment.Length - 1;
                    using var response = await SendRangeAsync(origin, cache, start, end, session.Identity.ETag, cancellationToken).ConfigureAwait(false);
                    ValidateRange(response, start, end, session.Identity.Size, session.Identity.ETag);
                    await using var incoming = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
                    while (session.Completed(index) < segment.Length)
                    {
                        var offset = session.Completed(index);
                        var wanted = (int)Math.Min(SegmentBufferSize, segment.Length - offset);
                        var allowance = await _rate.AcquireAsync(wanted, cancellationToken).ConfigureAwait(false);
                        var count = await HubHttp.ReadWithTimeoutAsync(incoming, buffer.AsMemory(0, allowance), cancellationToken).ConfigureAwait(false);
                        _rate.Refund(allowance - count);
                        if (count == 0) throw new IOException("Segment body ended before its declared range.");
                        await RandomAccess.WriteAsync(session.Partial.SafeFileHandle, buffer.AsMemory(0, count), segment.Start + offset, cancellationToken).ConfigureAwait(false);
                        var completesSegment = offset + count == segment.Length;
                        if (completesSegment)
                        {
                            // The last write is only declared complete after the exact
                            // HTTP body has ended. Another worker's checkpoint cannot
                            // accidentally certify an invalid trailing body or cancellation.
                            if (await HubHttp.ReadWithTimeoutAsync(incoming, buffer.AsMemory(0, 1), cancellationToken).ConfigureAwait(false) != 0)
                                throw new InvalidDataException("Segment body exceeds its declared range.");
                        }
                        session.Add(index, count);
                        sinceCheckpoint += count;
                        if (sinceCheckpoint >= CheckpointInterval && session.Completed(index) < segment.Length)
                        {
                            await session.CheckpointAsync(cancellationToken).ConfigureAwait(false);
                            sinceCheckpoint = 0;
                        }
                    }
                    await session.CheckpointAsync(cancellationToken).ConfigureAwait(false);
                }
                catch (Exception error) when (IsTransient(error) && !cancellationToken.IsCancellationRequested)
                {
                    failures++;
                    await session.CheckpointAsync(cancellationToken).ConfigureAwait(false);
                    if (failures >= MaximumAttempts) throw new IOException("A segment failed after four attempts. Flushed checkpoints were retained for resume.");
                    await DelayRetryAsync(error, failures, cancellationToken).ConfigureAwait(false);
                }
            }
        }
        finally { ArrayPool<byte>.Shared.Return(buffer); }
    }

    private static async Task ResetForSequentialAsync(string partPath, string statePath, DownloadRequest request, CancellationToken cancellationToken)
    {
        DownloadPaths.RegularFile(partPath);
        // Identity was checked under the exclusive per-file lock before reaching here.
        if (File.Exists(partPath))
        {
            await using var file = new FileStream(partPath, FileMode.Open, FileAccess.Write, FileShare.None, 1, FileOptions.Asynchronous);
            file.SetLength(0);
            file.Flush(flushToDisk: true);
        }
        await SaveIdentityAsync(statePath, TransferIdentity.For(request), cancellationToken).ConfigureAwait(false);
    }

    private sealed class SegmentSession(FileStream partial, string statePath, TransferIdentity identity, Reporter reporter)
    {
        private readonly SemaphoreSlim _checkpoint = new(1, 1);
        private readonly long[] _completed = identity.Segments!.Select(segment => segment.CompletedBytes).ToArray();
        private long _totalCompleted = identity.Segments!.Sum(segment => segment.CompletedBytes);
        internal FileStream Partial { get; } = partial;
        internal TransferIdentity Identity { get; } = identity;
        internal SegmentCheckpoint[] Segments { get; } = identity.Segments!;
        internal long Completed(int index) => Interlocked.Read(ref _completed[index]);
        internal long TotalCompleted => Interlocked.Read(ref _totalCompleted);
        internal void Add(int index, int count)
        {
            Interlocked.Add(ref _completed[index], count);
            Interlocked.Add(ref _totalCompleted, count);
            reporter.Report(TotalCompleted, "Downloading segments");
        }
        internal TransferIdentity Snapshot() => Identity with
        {
            Segments = Segments.Select((segment, index) => segment with { CompletedBytes = Completed(index) }).ToArray()
        };
        internal async Task CheckpointAsync(CancellationToken cancellationToken)
        {
            await _checkpoint.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                var snapshot = Snapshot();
                RandomAccess.FlushToDisk(Partial.SafeFileHandle);
                await SaveIdentityAsync(statePath, snapshot, cancellationToken).ConfigureAwait(false);
            }
            finally { _checkpoint.Release(); }
        }
    }
}
