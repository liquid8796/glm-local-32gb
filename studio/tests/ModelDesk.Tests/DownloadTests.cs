using System.Diagnostics;
using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using ModelDesk.Core;
using ModelDesk.Infrastructure.Downloads;

namespace ModelDesk.Tests;

public sealed class DownloadTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "modeldesk-download-tests-" + Guid.NewGuid().ToString("N"));
    private static readonly byte[] Data = Enumerable.Range(0, 180_000).Select(index => (byte)(index * 17)).ToArray();

    public DownloadTests() => Directory.CreateDirectory(_root);
    public void Dispose() => Directory.Delete(_root, recursive: true);

    private DownloadRequest Request(byte[]? data = null, string path = "nested/weights.safetensors", bool git = false) => new("example/model", HubClientTests.Revision,
        new HubFile(path, (data ?? Data).LongLength, git ? null : Convert.ToHexString(SHA256.HashData(data ?? Data)).ToLowerInvariant(),
            git ? HubClientTests.GitHash(data ?? Data) : null), _root);

    [Fact]
    public async Task DownloadsBoundedStreamsVerifiesLfsAndSkipsIdenticalFinal()
    {
        var stream = new RecordingStream(Data);
        using var handler = new HubTestHandler((_, _, _) => Task.FromResult(Response(Data, stream: stream)));
        using var client = new HttpClient(handler);
        var downloader = new ModelDownloader(client, new HubTestCredentials());
        var progress = new List<DownloadProgress>();
        var first = await downloader.DownloadAsync(Request(), new InlineProgress<DownloadProgress>(progress.Add));
        Assert.True(first.HashVerified);
        Assert.False(first.AlreadyPresent);
        Assert.Equal(Data, await File.ReadAllBytesAsync(first.FilePath));
        Assert.All(stream.ReadSizes, count => Assert.InRange(count, 1, 65536));
        Assert.False(File.Exists(first.FilePath + ".part"));
        Assert.False(File.Exists(first.FilePath + ".part.json"));
        Assert.Equal("Completed", progress[^1].Stage);
        var second = await downloader.DownloadAsync(Request());
        Assert.True(second.AlreadyPresent);
        Assert.Single(handler.Calls);
    }

    [Fact]
    public async Task GitBlobHashIncludesGitObjectHeader()
    {
        var bytes = Encoding.UTF8.GetBytes("small config\n");
        using var handler = new HubTestHandler((_, _, _) => Task.FromResult(Response(bytes)));
        using var client = new HttpClient(handler);
        var result = await new ModelDownloader(client, new HubTestCredentials()).DownloadAsync(Request(bytes, "config.json", git: true));
        Assert.True(result.HashVerified);
        Assert.Equal(bytes, await File.ReadAllBytesAsync(result.FilePath));
    }

    [Fact]
    public async Task CancellationRetainsOwnedPartialAndResumeUsesExactRangeAndEtag()
    {
        using var cancelled = new CancellationTokenSource();
        using var handler = new HubTestHandler((request, _, number) => Task.FromResult(number == 1
            ? Response(Data, stream: new CancelAfterPrefixStream(Data, 65536, cancelled))
            : Response(Data, start: checked((int)request.Headers.Range!.Ranges.Single().From!.Value))));
        using var client = new HttpClient(handler);
        var downloader = new ModelDownloader(client, new HubTestCredentials());
        var request = Request();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => downloader.DownloadAsync(request, cancellationToken: cancelled.Token));
        var final = Path.Combine(_root, "nested", "weights.safetensors");
        Assert.Equal(65536, new FileInfo(final + ".part").Length);
        Assert.True(File.Exists(final + ".part.json"));
        var result = await downloader.DownloadAsync(request);
        Assert.Equal(65536, result.ResumedBytes);
        Assert.Equal("bytes=65536-", handler.Calls.Last().Range);
        Assert.Equal("\"stable\"", handler.Calls.Last().IfRange);
        Assert.Equal(Data, await File.ReadAllBytesAsync(final));
    }

    [Fact]
    public async Task Server200DuringResumeRestartsRatherThanAppending()
    {
        using var cancelled = new CancellationTokenSource();
        using var handler = new HubTestHandler((_, _, number) => Task.FromResult(number == 1
            ? Response(Data, stream: new CancelAfterPrefixStream(Data, 65536, cancelled)) : Response(Data)));
        using var client = new HttpClient(handler);
        var downloader = new ModelDownloader(client, new HubTestCredentials());
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => downloader.DownloadAsync(Request(), cancellationToken: cancelled.Token));
        var result = await downloader.DownloadAsync(Request());
        Assert.Equal(0, result.ResumedBytes);
        Assert.Equal(Data, await File.ReadAllBytesAsync(result.FilePath));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task ResumedEtagOrContentRangeMismatchCannotPublish(bool changeEtag)
    {
        using var cancelled = new CancellationTokenSource();
        using var handler = new HubTestHandler((_, _, number) =>
        {
            if (number == 1) return Task.FromResult(Response(Data, stream: new CancelAfterPrefixStream(Data, 65536, cancelled)));
            var response = Response(Data, start: 65536);
            if (changeEtag) response.Headers.ETag = new EntityTagHeaderValue("\"different\"");
            else response.Content.Headers.ContentRange = new ContentRangeHeaderValue(65535, Data.Length - 1, Data.Length);
            return Task.FromResult(response);
        });
        using var client = new HttpClient(handler);
        var downloader = new ModelDownloader(client, new HubTestCredentials());
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => downloader.DownloadAsync(Request(), cancellationToken: cancelled.Token));
        await Assert.ThrowsAsync<InvalidDataException>(() => downloader.DownloadAsync(Request()));
        Assert.False(File.Exists(Path.Combine(_root, "nested", "weights.safetensors")));
        Assert.Equal(2, handler.Calls.Count);
    }

    [Fact]
    public async Task CorruptChecksumCannotPublishAndRetryRestartsIdentifiedPartial()
    {
        using var handler = new HubTestHandler((_, _, number) => Task.FromResult(Response(number == 1 ? new byte[Data.Length] : Data)));
        using var client = new HttpClient(handler);
        var downloader = new ModelDownloader(client, new HubTestCredentials());
        await Assert.ThrowsAsync<InvalidDataException>(() => downloader.DownloadAsync(Request()));
        Assert.False(File.Exists(Path.Combine(_root, "nested", "weights.safetensors")));
        var result = await downloader.DownloadAsync(Request());
        Assert.True(result.HashVerified);
        Assert.Null(handler.Calls.Last().Range);
    }

    [Fact]
    public async Task ExistingDifferentFinalAndUnidentifiedPartAreNeverOverwritten()
    {
        using var handler = new HubTestHandler((_, _, _) => throw new InvalidOperationException("No request expected"));
        using var client = new HttpClient(handler);
        var downloader = new ModelDownloader(client, new HubTestCredentials());
        var first = Request(path: "existing.bin");
        await File.WriteAllTextAsync(Path.Combine(_root, first.File.Path), "user content");
        await Assert.ThrowsAsync<IOException>(() => downloader.DownloadAsync(first));
        Assert.Equal("user content", await File.ReadAllTextAsync(Path.Combine(_root, first.File.Path)));
        var second = Request(path: "other.bin");
        await File.WriteAllTextAsync(Path.Combine(_root, second.File.Path + ".part"), "unidentified bytes");
        await Assert.ThrowsAsync<InvalidDataException>(() => downloader.DownloadAsync(second));
        Assert.Equal("unidentified bytes", await File.ReadAllTextAsync(Path.Combine(_root, second.File.Path + ".part")));
        Assert.Empty(handler.Calls);
    }

    [Fact]
    public async Task PartialCannotResumeUnderAnotherRevision()
    {
        using var cancelled = new CancellationTokenSource();
        using var handler = new HubTestHandler((_, _, _) => Task.FromResult(Response(Data, stream: new CancelAfterPrefixStream(Data, 65536, cancelled))));
        using var client = new HttpClient(handler);
        var downloader = new ModelDownloader(client, new HubTestCredentials());
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => downloader.DownloadAsync(Request(), cancellationToken: cancelled.Token));
        await Assert.ThrowsAsync<InvalidDataException>(() => downloader.DownloadAsync(Request() with { Revision = new string('b', 40) }));
        Assert.Single(handler.Calls);
    }

    [Theory]
    [InlineData("../escape")]
    [InlineData("a/../../escape")]
    [InlineData("C:/absolute")]
    [InlineData("a\\b")]
    [InlineData("CON.txt")]
    [InlineData("a/LPT1")]
    [InlineData("a/COM¹.bin")]
    [InlineData("a/file.")]
    [InlineData("a/file ")]
    [InlineData("a/file:stream")]
    [InlineData("SHORT~1.bin")]
    [InlineData("a/file.part")]
    public async Task WindowsAliasesTraversalAndReservedStatePathsFailBeforeNetwork(string path)
    {
        using var handler = new HubTestHandler((_, _, _) => throw new InvalidOperationException("No request expected"));
        using var client = new HttpClient(handler);
        await Assert.ThrowsAsync<ArgumentException>(() => new ModelDownloader(client, new HubTestCredentials()).DownloadAsync(Request(path: path)));
        Assert.Empty(handler.Calls);
        Assert.Empty(Directory.EnumerateFileSystemEntries(_root));
    }

    [Fact]
    public async Task JunctionDestinationIsRejectedBeforeAnyNetworkRequest()
    {
        var target = Path.Combine(_root, "target");
        var link = Path.Combine(_root, "link");
        Directory.CreateDirectory(target);
        // Only create a junction between two test-owned directories. Delete the link
        // itself before recursive fixture cleanup, so cleanup cannot follow it.
        using var process = Process.Start(new ProcessStartInfo("cmd.exe", $"/d /c mklink /J \"{link}\" \"{target}\"")
        {
            UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true
        })!;
        var stdout = process.StandardOutput.ReadToEndAsync();
        var stderr = process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();
        Assert.True(process.ExitCode == 0, await stdout + await stderr);
        try
        {
            Assert.True((File.GetAttributes(link) & FileAttributes.ReparsePoint) != 0);
            using var handler = new HubTestHandler((_, _, _) => throw new InvalidOperationException("No request expected"));
            using var client = new HttpClient(handler);
            await Assert.ThrowsAsync<IOException>(() => new ModelDownloader(client, new HubTestCredentials()).DownloadAsync(Request(path: "link/file.bin")));
            Assert.Empty(handler.Calls);
            Assert.Empty(Directory.EnumerateFileSystemEntries(target));
        }
        finally { Directory.Delete(link); }
    }

    [Fact]
    public async Task CrossInstanceFileLockPreventsGuiCliCollision()
    {
        var started = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var cancellation = new CancellationTokenSource();
        using var handler = new HubTestHandler(async (_, token, _) =>
        {
            started.TrySetResult();
            await Task.Delay(Timeout.InfiniteTimeSpan, token);
            return Response(Data);
        });
        using var client = new HttpClient(handler);
        var first = new ModelDownloader(client, new HubTestCredentials()).DownloadAsync(Request(), cancellationToken: cancellation.Token);
        await started.Task.WaitAsync(TimeSpan.FromSeconds(5));
        await Assert.ThrowsAsync<IOException>(() => new ModelDownloader(client, new HubTestCredentials()).DownloadAsync(Request()));
        cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => first);
        Assert.Single(handler.Calls);
    }

    [Fact]
    public async Task RedirectToCdnDropsBearerAndErrorsDoNotRevealSignedQuery()
    {
        using var handler = new HubTestHandler((_, _, number) =>
        {
            if (number == 1)
            {
                var response = new HttpResponseMessage(HttpStatusCode.Redirect);
                response.Headers.Location = new Uri("https://cdn-lfs.hf.co/fixture?secret-signature=sample");
                return Task.FromResult(response);
            }
            return Task.FromResult(Response(Data));
        });
        using var client = new HttpClient(handler);
        await new ModelDownloader(client, new HubTestCredentials("hf_example_only")).DownloadAsync(Request());
        Assert.Equal("Bearer hf_example_only", handler.Calls.First().Authorization);
        Assert.Null(handler.Calls.Last().Authorization);
        using var failingHandler = new HubTestHandler((_, _, _) => throw new HttpRequestException("https://cdn-lfs.hf.co/?secret-signature=sample"));
        using var failingClient = new HttpClient(failingHandler);
        var error = await Assert.ThrowsAsync<IOException>(() => new ModelDownloader(failingClient, new HubTestCredentials()).DownloadAsync(Request(path: "failed.bin")));
        Assert.DoesNotContain("secret-signature", error.ToString());
        Assert.Equal(4, failingHandler.Calls.Count);
    }

    [Fact]
    public async Task AggregateSpeedLimitAppliesAcrossConcurrentFilesAndCanChangeWhileRunning()
    {
        var bytes = new byte[12_000];
        using var handler = new HubTestHandler((_, _, _) => Task.FromResult(Response(bytes)));
        using var client = new HttpClient(handler);
        var downloader = new ModelDownloader(client, new HubTestCredentials()) { BytesPerSecondLimit = 24_000 };
        var timer = Stopwatch.StartNew();
        await Task.WhenAll(downloader.DownloadAsync(Request(bytes, "one.bin")), downloader.DownloadAsync(Request(bytes, "two.bin")));
        Assert.True(timer.Elapsed >= TimeSpan.FromMilliseconds(850), $"Combined rate exceeded the limit: {timer.Elapsed}.");
        Assert.True(timer.Elapsed < TimeSpan.FromSeconds(10));
        downloader.BytesPerSecondLimit = 1;
        var pending = downloader.DownloadAsync(Request(bytes, "dynamic.bin"));
        await Task.Delay(80);
        Assert.False(pending.IsCompleted);
        downloader.BytesPerSecondLimit = 0;
        await pending.WaitAsync(TimeSpan.FromSeconds(3));
    }

    [Fact]
    public async Task QueuePersistenceRestoresInterruptedDownloadsAsPaused()
    {
        var state = Path.Combine(_root, "state");
        var store = new JsonDownloadQueueStore(state);
        var original = new[] { new SavedDownload(Guid.NewGuid(), Request(), DownloadState.Downloading, 42),
            new SavedDownload(Guid.NewGuid(), Request(path: "done.bin"), DownloadState.Completed, Data.Length) };
        await store.SaveAsync(original);
        var restored = await new JsonDownloadQueueStore(state).LoadAsync();
        Assert.Equal(DownloadState.Paused, restored[0].State);
        Assert.Equal(42, restored[0].DownloadedBytes);
        Assert.Equal(DownloadState.Completed, restored[1].State);
        Assert.Equal(original[0].Request, restored[0].Request);
        await File.WriteAllTextAsync(Path.Combine(state, "downloads.json"), "{invalid");
        await Assert.ThrowsAsync<InvalidDataException>(() => store.LoadAsync());
    }

    [Fact]
    public async Task WholeSelectionDiskPreflightRejectsCombinedFilesBeforeAnyRequestOrMutation()
    {
        using var handler = new HubTestHandler((_, _, _) => throw new InvalidOperationException("No request expected"));
        using var client = new HttpClient(handler);
        var downloader = new ModelDownloader(client, new HubTestCredentials());
        var free = new DriveInfo(Path.GetPathRoot(_root)!).AvailableFreeSpace;
        var size = free * 3 / 5;
        var first = Request(path: "big-one.bin") with { File = new HubFile("big-one.bin", size, new string('a', 64)) };
        var second = first with { File = first.File with { Path = "big-two.bin" } };
        Assert.Single((await downloader.ValidateBatchAsync([first])).Volumes);
        await Assert.ThrowsAsync<IOException>(() => downloader.ValidateBatchAsync([first, second]));
        Assert.Empty(handler.Calls);
        Assert.Empty(Directory.EnumerateFileSystemEntries(_root));
    }

    [Fact]
    public async Task BatchPreflightSubtractsIdentifiedPartialAndDeduplicatesExactTarget()
    {
        using var cancelled = new CancellationTokenSource();
        using var handler = new HubTestHandler((_, _, _) => Task.FromResult(Response(Data, stream: new CancelAfterPrefixStream(Data, 65536, cancelled))));
        using var client = new HttpClient(handler);
        var downloader = new ModelDownloader(client, new HubTestCredentials());
        var request = Request();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => downloader.DownloadAsync(request, cancellationToken: cancelled.Token));
        var estimate = await downloader.ValidateBatchAsync([request, request]);
        Assert.Equal(1, estimate.FileCount);
        Assert.Equal(Data.Length - 65536 + 16 * 1024 * 1024, estimate.Volumes.Single().RequiredBytes);
        await Assert.ThrowsAsync<InvalidDataException>(() => downloader.ValidateBatchAsync([request, request with { Revision = new string('b', 40) }]));
        Assert.Single(handler.Calls);
    }

    private static HttpResponseMessage Response(byte[] bytes, int start = 0, Stream? stream = null)
    {
        var response = new HttpResponseMessage(start > 0 ? HttpStatusCode.PartialContent : HttpStatusCode.OK)
        { Content = stream is null ? new ByteArrayContent(bytes[start..]) : new StreamContent(stream) };
        response.Headers.ETag = new EntityTagHeaderValue("\"stable\"");
        response.Content.Headers.ContentLength = bytes.Length - start;
        if (start > 0) response.Content.Headers.ContentRange = new ContentRangeHeaderValue(start, bytes.Length - 1, bytes.Length);
        return response;
    }

    private sealed class RecordingStream(byte[] bytes) : MemoryStream(bytes, writable: false)
    {
        internal List<int> ReadSizes { get; } = [];
        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            ReadSizes.Add(buffer.Length);
            return base.ReadAsync(buffer, cancellationToken);
        }
    }

    private sealed class CancelAfterPrefixStream(byte[] bytes, int prefix, CancellationTokenSource cancellation) : MemoryStream(bytes, writable: false)
    {
        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            if (Position >= prefix)
            {
                cancellation.Cancel();
                cancellationToken.ThrowIfCancellationRequested();
            }
            return base.ReadAsync(buffer[..Math.Min(buffer.Length, prefix - (int)Position)], cancellationToken);
        }
    }
}

internal sealed class InlineProgress<T>(Action<T> action) : IProgress<T>
{
    public void Report(T value) => action(value);
}
