using System.Text.Json;
using ModelDesk.Core;

namespace ModelDesk.Infrastructure.Downloads;

public sealed class JsonDownloadQueueStore : IDownloadQueueStore
{
    private const int MaximumItems = 10_000;
    private readonly string _path;
    private readonly SemaphoreSlim _gate = new(1, 1);

    public JsonDownloadQueueStore(string stateDirectory)
    {
        _path = Path.Combine(DownloadPaths.Root(stateDirectory), "downloads.json");
        DownloadPaths.CreateParent(_path);
    }

    public async Task<IReadOnlyList<SavedDownload>> LoadAsync(CancellationToken cancellationToken = default)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            DownloadPaths.RegularFile(_path);
            if (!File.Exists(_path)) return [];
            await using var input = new FileStream(_path, FileMode.Open, FileAccess.Read, FileShare.Read, 64 * 1024, FileOptions.Asynchronous);
            if (input.Length > 8 * 1024 * 1024) throw new InvalidDataException("Saved download queue exceeds its size limit.");
            QueueDocument document;
            try { document = await JsonSerializer.DeserializeAsync<QueueDocument>(input, cancellationToken: cancellationToken).ConfigureAwait(false)
                ?? throw new InvalidDataException("Saved download queue is empty."); }
            catch (JsonException) { throw new InvalidDataException("Saved download queue is invalid."); }
            if (document.Version != 1 || document.Downloads is null || document.Downloads.Count > MaximumItems)
                throw new InvalidDataException("Unsupported download queue format.");
            Validate(document.Downloads);
            return document.Downloads.Select(item => item.State == DownloadState.Downloading
                ? item with { State = DownloadState.Paused, Error = null } : item).ToArray();
        }
        finally { _gate.Release(); }
    }

    public async Task SaveAsync(IReadOnlyList<SavedDownload> downloads, CancellationToken cancellationToken = default)
    {
        Validate(downloads);
        // Freeze caller-owned collections before awaiting; a changing UI queue cannot corrupt serialization.
        var document = new QueueDocument(1, downloads.ToArray());
        var encoded = JsonSerializer.SerializeToUtf8Bytes(document);
        if (encoded.Length > 8 * 1024 * 1024) throw new InvalidDataException("Saved download queue exceeds its size limit.");
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        var temporary = _path + ".tmp-" + Guid.NewGuid().ToString("N");
        try
        {
            DownloadPaths.RegularFile(_path);
            DownloadPaths.RegularFile(_path + ".lock");
            using var fileLock = new FileStream(_path + ".lock", FileMode.OpenOrCreate, FileAccess.Write, FileShare.None, 1, FileOptions.DeleteOnClose);
            await using (var output = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None, 64 * 1024, FileOptions.Asynchronous))
            {
                await output.WriteAsync(encoded, cancellationToken).ConfigureAwait(false);
                await output.FlushAsync(cancellationToken).ConfigureAwait(false);
            }
            DownloadPaths.RegularFile(_path);
            File.Move(temporary, _path, overwrite: true);
        }
        finally
        {
            if (File.Exists(temporary)) File.Delete(temporary);
            _gate.Release();
        }
    }

    private static void Validate(IReadOnlyList<SavedDownload> downloads)
    {
        ArgumentNullException.ThrowIfNull(downloads);
        if (downloads.Count > MaximumItems || downloads.Select(item => item.Id).Distinct().Count() != downloads.Count)
            throw new InvalidDataException("Download queue contains too many or duplicate items.");
        foreach (var item in downloads)
        {
            if (item.Id == Guid.Empty || !Enum.IsDefined(item.State) || item.DownloadedBytes < 0 || item.DownloadedBytes > item.Request.File.Size ||
                item.Error?.Length > 4096) throw new InvalidDataException("Download queue item is invalid.");
            DownloadPaths.Resolve(item.Request);
        }
    }

    private sealed record QueueDocument(int Version, IReadOnlyList<SavedDownload> Downloads);
}
