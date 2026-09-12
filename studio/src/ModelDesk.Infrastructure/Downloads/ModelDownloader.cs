using System.Diagnostics;
using System.Buffers;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using ModelDesk.Core;
using ModelDesk.Infrastructure.Hub;

namespace ModelDesk.Infrastructure.Downloads;

/// <summary>Resumable, hash-verified transfers. Only this downloader's identified partial files may be changed.</summary>
public sealed partial class ModelDownloader(HttpClient httpClient, ICredentialStore credentials) : IModelDownloader, IModelDownloadPlanner, IAdvancedDownloadOptions
{
    private const int BufferSize = 64 * 1024;
    private const int MaximumAttempts = 4;
    private readonly AggregateRateLimiter _rate = new();
    public long BytesPerSecondLimit { get => _rate.Limit; set => _rate.Limit = value; }

    /// <summary>Preflight the complete selection without network or filesystem mutations.
    /// RequiredBytes includes one 16-MiB reserve per drive. Existing matching-size finals
    /// count as zero here; DownloadAsync still verifies their content hash before skipping.</summary>
    public async Task<DownloadBatchEstimate> ValidateBatchAsync(IReadOnlyList<DownloadRequest> requests,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count > 100_000) throw new ArgumentException("A download selection cannot exceed 100,000 files.");
        var targets = new Dictionary<string, TransferIdentity>(StringComparer.OrdinalIgnoreCase);
        var volumes = new Dictionary<string, long>(StringComparer.OrdinalIgnoreCase);
        foreach (var request in requests)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ArgumentNullException.ThrowIfNull(request);
            ArgumentNullException.ThrowIfNull(request.File);
            var path = DownloadPaths.Resolve(request);
            var expected = TransferIdentity.For(request);
            if (targets.TryGetValue(path, out var other))
            {
                if (!other.Matches(expected)) throw new InvalidDataException("Selected files have conflicting identities at the same destination path.");
                continue;
            }
            targets.Add(path, expected);
            foreach (var candidate in new[] { path, path + ".part", path + ".part.json" }) DownloadPaths.RegularFile(candidate);
            long required = request.File.Size;
            if (File.Exists(path))
            {
                if (new FileInfo(path).Length != request.File.Size)
                    throw new IOException("An existing destination file differs in size. Choose another folder; existing files are never overwritten.");
                required = 0;
            }
            else if (File.Exists(path + ".part") || File.Exists(path + ".part.json"))
            {
                if (!File.Exists(path + ".part.json")) throw new InvalidDataException("A partial file has no verified resume identity.");
                var saved = await ReadIdentityAsync(path + ".part.json", cancellationToken).ConfigureAwait(false);
                if (!saved.Matches(expected)) throw new InvalidDataException("A partial file belongs to another model, revision or file identity.");
                var present = File.Exists(path + ".part") ? new FileInfo(path + ".part").Length : 0;
                if (present > request.File.Size) throw new InvalidDataException("A partial file exceeds its declared size.");
                var completed = saved.CompletedBytes(present);
                // Disk allocation is independent of network progress. A preallocated
                // v2 file can hold its disk space while its bitmap still contains holes.
                required -= saved.Version == 2 ? DiskReservations.AllocatedBytes(path + ".part") : completed;
            }
            var volume = Path.GetPathRoot(path)!;
            volumes[volume] = checked(volumes.GetValueOrDefault(volume) + required);
        }
        var estimates = volumes.OrderBy(item => item.Key, StringComparer.OrdinalIgnoreCase).Select(item =>
            new DownloadVolumeEstimate(item.Key, checked(item.Value + 16 * 1024 * 1024), new DriveInfo(item.Key).AvailableFreeSpace)).ToArray();
        foreach (var estimate in estimates)
            if (estimate.RequiredBytes > estimate.AvailableBytes)
                throw new IOException($"The complete selection requires {estimate.RequiredBytes:N0} bytes on {estimate.Volume}; only {estimate.AvailableBytes:N0} bytes are available.");
        return new DownloadBatchEstimate(estimates, targets.Count);
    }

    public async Task<DownloadResult> DownloadAsync(DownloadRequest request, IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(request.File);
        var finalPath = DownloadPaths.Resolve(request);
        var partPath = finalPath + ".part";
        var statePath = partPath + ".json";
        var lockPath = finalPath + ".download.lock";
        cancellationToken.ThrowIfCancellationRequested();
        DownloadPaths.CreateParent(finalPath);
        foreach (var path in new[] { finalPath, partPath, statePath, lockPath }) DownloadPaths.RegularFile(path);
        FileStream fileLock;
        try { fileLock = new FileStream(lockPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None, 1, FileOptions.DeleteOnClose); }
        catch (IOException) { throw new IOException("This file is already being downloaded by another window or CLI process."); }
        using (fileLock)
        {
            var timer = Stopwatch.StartNew();
            if (File.Exists(finalPath))
            {
                await using var existing = new FileStream(finalPath, FileMode.Open, FileAccess.Read, FileShare.Read, BufferSize, FileOptions.Asynchronous | FileOptions.SequentialScan);
                if (existing.Length != request.File.Size || !await VerifyAsync(existing, request.File, cancellationToken).ConfigureAwait(false))
                    throw new IOException("A different file already exists at the destination. Choose another folder; existing files are never overwritten.");
                progress?.Report(new DownloadProgress(request.File.Path, request.File.Size, request.File.Size, 0, TimeSpan.Zero, "Already present"));
                return new DownloadResult(finalPath, request.File.Size, true, true, 0, timer.Elapsed);
            }
            var identity = TransferIdentity.For(request);
            var existed = File.Exists(partPath);
            if (File.Exists(statePath))
            {
                var saved = await ReadIdentityAsync(statePath, cancellationToken).ConfigureAwait(false);
                if (!saved.Matches(identity)) throw new InvalidDataException("Partial download belongs to another model, revision or file identity. Choose a separate destination.");
                identity = saved;
            }
            else if (existed) throw new InvalidDataException("An unidentified partial file already exists. It cannot be overwritten or resumed.");
            else await SaveIdentityAsync(statePath, identity, cancellationToken).ConfigureAwait(false);

            var options = SnapshotOptions();
            if (identity.Version == 2 || !existed && options.Connections > 1 && request.File.Size >= options.Threshold)
            {
                var segmented = await TrySegmentedAsync(request, finalPath, partPath, statePath, identity, options,
                    progress, timer, cancellationToken).ConfigureAwait(false);
                if (segmented is not null) return segmented;
                identity = TransferIdentity.For(request); // The segmented fallback already discarded its owned data safely.
            }

            long resumedBytes;
            await using (var partial = new FileStream(partPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None,
                BufferSize, FileOptions.Asynchronous | FileOptions.SequentialScan))
            {
                if (partial.Length > request.File.Size) throw new InvalidDataException("Partial download is larger than its frozen file metadata.");
                if (identity.HashFailed)
                {
                    partial.SetLength(0);
                    identity = identity with { ETag = null, HashFailed = false };
                    await SaveIdentityAsync(statePath, identity, cancellationToken).ConfigureAwait(false);
                }
                resumedBytes = partial.Length;
                using var disk = DiskReservations.Acquire(finalPath, request.File.Size - partial.Length);
                var reporter = new Reporter(request.File, progress, partial.Length);
                reporter.Report(partial.Length, partial.Length > 0 ? "Resuming" : "Starting", true);
                var failures = 0;
                try
                {
                    while (partial.Length < request.File.Size)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        try
                        {
                            using var connection = await DownloadConnections.AcquireAsync(cancellationToken).ConfigureAwait(false);
                            // Without a strong ETag, restart our own partial instead of guessing a range identity.
                            if (partial.Length > 0 && identity.ETag is null)
                            {
                                partial.SetLength(0);
                                disk.Resize(request.File.Size);
                                resumedBytes = 0;
                                reporter.Restart();
                            }
                            var offset = partial.Length;
                            var uri = new Uri($"https://huggingface.co/{HubHttp.EncodedModel(request.ModelId)}/resolve/{HubHttp.ImmutableRevision(request.Revision)}/{HubHttp.EncodedPath(request.File.Path)}");
                            using var response = await HubHttp.SendAsync(httpClient, credentials, uri, cancellationToken,
                                offset, identity.ETag, allowCdn: true).ConfigureAwait(false);
                            EnsureDownloadSuccess(response);
                            if (response.StatusCode is not (HttpStatusCode.OK or HttpStatusCode.PartialContent))
                                throw new InvalidDataException("Download response must be HTTP 200 or 206.");
                            if (response.Content.Headers.ContentEncoding.Any(value => !value.Equals("identity", StringComparison.OrdinalIgnoreCase)))
                                throw new InvalidDataException("Encoded download bodies cannot be combined with byte-range metadata.");
                            var receivedTag = response.Headers.ETag is { IsWeak: false } tag ? tag.ToString() : null;
                            if (response.StatusCode == HttpStatusCode.PartialContent)
                            {
                                var range = response.Content.Headers.ContentRange;
                                if (range is null || !range.HasRange || !range.HasLength || range.Unit != "bytes" ||
                                    range.From != offset || range.To != request.File.Size - 1 || range.Length != request.File.Size)
                                    throw new InvalidDataException("Server returned an inconsistent Content-Range for the frozen file.");
                                if (offset > 0 && !string.Equals(identity.ETag, receivedTag, StringComparison.Ordinal))
                                    throw new InvalidDataException("Server ETag changed during a resumed download.");
                            }
                            else if (offset > 0)
                            {
                                // HTTP 200 means Range/If-Range was ignored. Never append it to old data.
                                partial.SetLength(0);
                                disk.Resize(request.File.Size);
                                offset = 0;
                                resumedBytes = 0;
                                reporter.Restart();
                            }
                            var expectedBody = request.File.Size - offset;
                            if (response.Content.Headers.ContentLength is { } bodySize && bodySize != expectedBody)
                                throw new InvalidDataException("Response length differs from the frozen file size.");
                            identity = identity with { ETag = receivedTag };
                            await SaveIdentityAsync(statePath, identity, cancellationToken).ConfigureAwait(false);
                            partial.Position = offset;
                            await using var incoming = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
                            var buffer = ArrayPool<byte>.Shared.Rent(BufferSize);
                            try
                            {
                            while (partial.Position < request.File.Size)
                            {
                                var wanted = (int)Math.Min(BufferSize, request.File.Size - partial.Position);
                                var allowance = await _rate.AcquireAsync(wanted, cancellationToken).ConfigureAwait(false);
                                var count = await HubHttp.ReadWithTimeoutAsync(incoming, buffer.AsMemory(0, allowance), cancellationToken).ConfigureAwait(false);
                                _rate.Refund(allowance - count);
                                if (count == 0) throw new IOException("Download body ended before the declared file size.");
                                await partial.WriteAsync(buffer.AsMemory(0, count), cancellationToken).ConfigureAwait(false);
                                disk.Consumed(count);
                                reporter.Report(partial.Position, "Downloading");
                            }
                            if (await HubHttp.ReadWithTimeoutAsync(incoming, buffer.AsMemory(0, 1), cancellationToken).ConfigureAwait(false) != 0)
                                throw new InvalidDataException("Download body exceeds the frozen file size.");
                            }
                            finally { ArrayPool<byte>.Shared.Return(buffer); }
                        }
                        catch (Exception error) when (IsTransient(error) && !cancellationToken.IsCancellationRequested)
                        {
                            failures++;
                            if (failures >= MaximumAttempts) throw new IOException("Download failed after four attempts. Its identified partial file was retained for resume.");
                            await partial.FlushAsync(cancellationToken).ConfigureAwait(false);
                            reporter.Report(partial.Length, $"Retrying ({failures}/{MaximumAttempts})", true);
                            await DelayRetryAsync(error, failures, cancellationToken).ConfigureAwait(false);
                        }
                    }
                    await partial.FlushAsync(cancellationToken).ConfigureAwait(false);
                    reporter.Report(partial.Length, "Verifying", true);
                    partial.Position = 0;
                    if (!await VerifyAsync(partial, request.File, cancellationToken).ConfigureAwait(false))
                    {
                        await SaveIdentityAsync(statePath, identity with { HashFailed = true }, cancellationToken).ConfigureAwait(false);
                        throw new InvalidDataException("Content hash verification failed. The final file was not published; retry restarts this identified partial.");
                    }
                }
                catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
                {
                    await partial.FlushAsync(CancellationToken.None).ConfigureAwait(false);
                    reporter.Report(partial.Length, "Paused", true);
                    throw;
                }
            }
            cancellationToken.ThrowIfCancellationRequested();
            foreach (var path in new[] { finalPath, partPath, statePath }) DownloadPaths.RegularFile(path);
            File.Move(partPath, finalPath, overwrite: false);
            try { File.Delete(statePath); } catch (IOException) { } // Verified final remains valid if state cleanup is interrupted.
            progress?.Report(new DownloadProgress(request.File.Path, request.File.Size, request.File.Size,
                timer.Elapsed.TotalSeconds > 0 ? (request.File.Size - resumedBytes) / timer.Elapsed.TotalSeconds : 0, TimeSpan.Zero, "Completed"));
            return new DownloadResult(finalPath, request.File.Size, true, false, resumedBytes, timer.Elapsed);
        }
    }

    private static bool IsTransient(Exception error) => error is IOException ||
        error is HttpRequestException request && (request.StatusCode is null or HttpStatusCode.RequestTimeout or (HttpStatusCode)429 || (int)request.StatusCode >= 500);

    private static async Task<bool> VerifyAsync(Stream stream, HubFile file, CancellationToken cancellationToken)
    {
        using var hash = IncrementalHash.CreateHash(file.LfsSha256 is not null ? HashAlgorithmName.SHA256 : HashAlgorithmName.SHA1);
        if (file.LfsSha256 is null) hash.AppendData(Encoding.UTF8.GetBytes($"blob {file.Size}\0"));
        var buffer = ArrayPool<byte>.Shared.Rent(BufferSize);
        long count = 0;
        try
        {
        while (true)
        {
            var read = await stream.ReadAsync(buffer.AsMemory(0, BufferSize), cancellationToken).ConfigureAwait(false);
            if (read == 0) break;
            count += read;
            if (count > file.Size) return false;
            hash.AppendData(buffer, 0, read);
        }
        return count == file.Size && Convert.ToHexString(hash.GetHashAndReset()).Equals(file.LfsSha256 ?? file.BlobId, StringComparison.OrdinalIgnoreCase);
        }
        finally { ArrayPool<byte>.Shared.Return(buffer); }
    }

    private static async Task<TransferIdentity> ReadIdentityAsync(string path, CancellationToken cancellationToken)
    {
        await using var input = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, FileOptions.Asynchronous);
        if (input.Length > 1024 * 1024) throw new InvalidDataException("Partial identity sidecar is too large.");
        try
        {
            var identity = await JsonSerializer.DeserializeAsync<TransferIdentity>(input, cancellationToken: cancellationToken).ConfigureAwait(false)
                ?? throw new InvalidDataException("Partial identity sidecar is empty.");
            identity.ValidateSegments();
            return identity;
        }
        catch (JsonException) { throw new InvalidDataException("Partial identity sidecar is invalid."); }
    }

    private static async Task SaveIdentityAsync(string path, TransferIdentity identity, CancellationToken cancellationToken)
    {
        DownloadPaths.RegularFile(path);
        var temporary = path + ".tmp-" + Guid.NewGuid().ToString("N");
        try
        {
            await using (var output = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None, 4096, FileOptions.Asynchronous))
            {
                await JsonSerializer.SerializeAsync(output, identity, cancellationToken: cancellationToken).ConfigureAwait(false);
                await output.FlushAsync(cancellationToken).ConfigureAwait(false);
                output.Flush(flushToDisk: true);
            }
            DownloadPaths.RegularFile(path);
            File.Move(temporary, path, overwrite: true);
        }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
    }

    private sealed record TransferIdentity(int Version, string ModelId, string Revision, string Path,
        long Size, string? LfsSha256, string? BlobId, string? ETag = null, bool HashFailed = false,
        SegmentCheckpoint[]? Segments = null)
    {
        internal static TransferIdentity For(DownloadRequest request) => new(1, request.ModelId,
            request.Revision.ToLowerInvariant(), request.File.Path, request.File.Size,
            request.File.LfsSha256?.ToLowerInvariant(), request.File.BlobId?.ToLowerInvariant());
        internal bool Matches(TransferIdentity other) => Version is 1 or 2 && ModelId == other.ModelId && Revision == other.Revision &&
            Path == other.Path && Size == other.Size && LfsSha256 == other.LfsSha256 && BlobId == other.BlobId &&
            (ETag is null || System.Net.Http.Headers.EntityTagHeaderValue.TryParse(ETag, out var tag) && !tag.IsWeak);

        internal void ValidateSegments()
        {
            if (Version == 1 && Segments is null) return;
            if (Version != 2 || Segments is null || Segments.Length is < 1 or > 4096 || ETag is null)
                throw new InvalidDataException("Unsupported segmented resume manifest.");
            long position = 0;
            foreach (var segment in Segments)
            {
                if (segment is null || segment.Start != position || segment.Length <= 0 || segment.Length > Size - position ||
                    segment.CompletedBytes < 0 || segment.CompletedBytes > segment.Length)
                    throw new InvalidDataException("Segmented resume manifest has invalid ranges or completed offsets.");
                position += segment.Length;
            }
            if (position != Size) throw new InvalidDataException("Segmented resume manifest does not cover the complete file.");
        }

        internal long CompletedBytes(long fileLength)
        {
            if (Version == 1) return fileLength;
            ValidateSegments();
            if (fileLength != Size && (fileLength != 0 || Segments!.Any(segment => segment.CompletedBytes != 0)))
                throw new InvalidDataException("Preallocated partial file does not match its segment manifest.");
            return HashFailed ? 0 : Segments!.Sum(segment => segment.CompletedBytes);
        }
    }

    private sealed class Reporter(HubFile file, IProgress<DownloadProgress>? progress, long initial)
    {
        private readonly Stopwatch _clock = Stopwatch.StartNew();
        private readonly object _sync = new();
        private long _initial = initial;
        private long _maximumCurrent = initial;
        private long _lastMilliseconds = -200;
        internal void Restart()
        {
            lock (_sync)
            {
            _initial = 0;
            _maximumCurrent = 0;
            _clock.Restart();
            _lastMilliseconds = -200;
            Report(0, "Restarting", true);
            }
        }
        internal void Report(long current, string stage, bool force = false)
        {
            lock (_sync)
            {
            current = _maximumCurrent = Math.Max(current, _maximumCurrent);
            if (progress is null || !force && _clock.ElapsedMilliseconds - _lastMilliseconds < 200) return;
            _lastMilliseconds = _clock.ElapsedMilliseconds;
            var speed = _clock.Elapsed.TotalSeconds > 0 ? Math.Max(0, current - _initial) / _clock.Elapsed.TotalSeconds : 0;
            var seconds = speed > 0 ? (file.Size - current) / speed : double.NaN;
            TimeSpan? eta = double.IsFinite(seconds) && seconds >= 0 && seconds < TimeSpan.MaxValue.TotalSeconds ? TimeSpan.FromSeconds(seconds) : null;
            progress.Report(new DownloadProgress(file.Path, current, file.Size, speed, eta, stage));
            }
        }
    }
}
