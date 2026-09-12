namespace ModelDesk.Core;

public sealed record HubModelSummary(string Id, string? Author, long Downloads, int Likes,
    string? PipelineTag, string? LibraryName, bool IsPrivate, bool IsGated,
    DateTimeOffset? LastModified, IReadOnlyList<string> Tags);

public sealed record HubSearchResult(IReadOnlyList<HubModelSummary> Models, string? NextPage = null);

public sealed record HubFile(string Path, long Size, string? LfsSha256 = null, string? BlobId = null)
{
    public bool IsLfs => LfsSha256 is not null;
}

public sealed record HubModelDetail(string Id, string Revision, IReadOnlyList<HubFile> Files,
    string Readme, string? PipelineTag, string? LibraryName, bool IsPrivate, bool IsGated,
    long Downloads, int Likes)
{
    public long TotalBytes => Files.Sum(file => file.Size);
    public Uri PageUri => new($"https://huggingface.co/{Id}");
}

public interface IHuggingFaceClient
{
    Task<HubSearchResult> SearchAsync(string query, string? nextPage = null, CancellationToken cancellationToken = default);
    Task<HubModelDetail> GetModelAsync(string modelId, string revision = "main", CancellationToken cancellationToken = default);
}

public sealed record DownloadRequest(string ModelId, string Revision, HubFile File, string DestinationDirectory);
public sealed record DownloadProgress(string FilePath, long DownloadedBytes, long TotalBytes,
    double BytesPerSecond, TimeSpan? Remaining, string Stage);
public sealed record DownloadResult(string FilePath, long Bytes, bool HashVerified,
    bool AlreadyPresent, long ResumedBytes, TimeSpan Elapsed);

public interface IModelDownloader
{
    long BytesPerSecondLimit { get; set; }
    Task<DownloadResult> DownloadAsync(DownloadRequest request, IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default);
}

/// <summary>Advanced transfer tuning; options are snapshotted when a file starts.</summary>
public interface IAdvancedDownloadOptions
{
    int ConnectionsPerFile { get; set; }
    long ParallelThresholdBytes { get; set; }
    long SegmentSizeBytes { get; set; }
}

public sealed record DownloadVolumeEstimate(string Volume, long RequiredBytes, long AvailableBytes);
public sealed record DownloadBatchEstimate(IReadOnlyList<DownloadVolumeEstimate> Volumes, int FileCount);
public interface IModelDownloadPlanner
{
    Task<DownloadBatchEstimate> ValidateBatchAsync(IReadOnlyList<DownloadRequest> requests,
        CancellationToken cancellationToken = default);
}

public enum DownloadState { Queued, Downloading, Paused, Completed, Failed, Cancelled }
public sealed record SavedDownload(Guid Id, DownloadRequest Request, DownloadState State,
    long DownloadedBytes = 0, string? Error = null);

public interface IDownloadQueueStore
{
    Task<IReadOnlyList<SavedDownload>> LoadAsync(CancellationToken cancellationToken = default);
    Task SaveAsync(IReadOnlyList<SavedDownload> downloads, CancellationToken cancellationToken = default);
}
