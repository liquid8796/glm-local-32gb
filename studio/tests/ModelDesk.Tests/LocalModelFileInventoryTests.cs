using System.Diagnostics;
using ModelDesk.Core;
using ModelDesk.Infrastructure.Downloads;

namespace ModelDesk.Tests;

public sealed class LocalModelFileInventoryTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "modeldesk-inventory-" + Guid.NewGuid().ToString("N"));
    private readonly LocalModelFileInventory _inventory = new();
    public LocalModelFileInventoryTests() => Directory.CreateDirectory(_root);
    public void Dispose() => Directory.Delete(_root, recursive: true);

    [Fact]
    public async Task ExactFinalLengthIsDownloadedWithoutOpeningEvenAnExclusivelyLockedPayload()
    {
        Directory.CreateDirectory(Path.Combine(_root, "nested"));
        var path = Path.Combine(_root, "nested", "weights.safetensors");
        await using var locked = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);
        await locked.WriteAsync(new byte[] { 1, 2, 3, 4 });
        await locked.FlushAsync();
        var result = await _inventory.ScanAsync(_root, [new HubFile("nested/weights.safetensors", 4)]);
        Assert.Equal(LocalModelFileState.Downloaded, result[0].State);
        Assert.Equal(4, result[0].LocalBytes);
        Assert.Contains("does not verify the hash", result[0].Message);
        Assert.Contains("not read", result[0].Message);
        Assert.Equal(4, locked.Length);
    }

    [Fact]
    public async Task PartialFilesIncludingFullLengthPreallocationNeverCountAsDownloaded()
    {
        var partial = Path.Combine(_root, "weights.bin.part");
        await using var locked = new FileStream(partial, FileMode.CreateNew, FileAccess.Write, FileShare.None);
        locked.SetLength(4096);
        await locked.FlushAsync();
        await File.WriteAllTextAsync(partial + ".json", "This is intentionally not JSON; inventory must never parse it.");
        var result = await _inventory.ScanAsync(_root, [new HubFile("weights.bin", 4096)]);
        Assert.Equal(LocalModelFileState.Partial, result[0].State);
        Assert.Null(result[0].LocalBytes); // A v2 .part length cannot certify downloaded bytes.
        Assert.Contains("preallocated holes", result[0].Message);
        Assert.False(File.Exists(Path.Combine(_root, "weights.bin")));
        Assert.Equal("This is intentionally not JSON; inventory must never parse it.", await File.ReadAllTextAsync(partial + ".json"));
    }

    [Fact]
    public async Task FinalFilesTakePrecedenceOverStaleTransferMarkersAndSizeMismatchIsExplicit()
    {
        await File.WriteAllBytesAsync(Path.Combine(_root, "complete.bin"), [1, 2]);
        await File.WriteAllBytesAsync(Path.Combine(_root, "complete.bin.part"), [9]);
        await File.WriteAllBytesAsync(Path.Combine(_root, "wrong.bin"), [1, 2, 3]);
        await File.WriteAllBytesAsync(Path.Combine(_root, "wrong.bin.part"), new byte[5]);
        await File.WriteAllTextAsync(Path.Combine(_root, "marker.bin.part.json"), "{}");
        var result = await _inventory.ScanAsync(_root,
            [new HubFile("complete.bin", 2), new HubFile("wrong.bin", 5), new HubFile("marker.bin", 10), new HubFile("missing.bin", 10)]);
        Assert.Equal([LocalModelFileState.Downloaded, LocalModelFileState.SizeMismatch, LocalModelFileState.Partial, LocalModelFileState.Missing],
            result.Select(item => item.State).ToArray());
        Assert.Equal(3, result[1].LocalBytes);
        Assert.Contains("not overwritten", result[1].Message);
        Assert.Equal(0, result[3].LocalBytes);
    }

    [Fact]
    public async Task MissingDirectoryIsNeverCreatedAndZeroSizeRequiresARealFinalFile()
    {
        var absentDirectory = Path.Combine(_root, "never-created");
        var absent = await _inventory.ScanAsync(absentDirectory, [new HubFile("zero.bin", 0)]);
        Assert.Equal(LocalModelFileState.Missing, absent[0].State);
        Assert.False(Directory.Exists(absentDirectory));
        await File.WriteAllBytesAsync(Path.Combine(_root, "zero.bin"), []);
        var present = await _inventory.ScanAsync(_root, [new HubFile("zero.bin", 0)]);
        Assert.Equal(LocalModelFileState.Downloaded, present[0].State);
    }

    [Fact]
    public async Task ScanPreservesUnrelatedFilesAndOnlyChecksRequestedNames()
    {
        var unrelated = Path.Combine(_root, "personal.txt");
        await File.WriteAllTextAsync(unrelated, "unrelated user data");
        var before = File.GetLastWriteTimeUtc(unrelated);
        var result = await _inventory.ScanAsync(_root, [new HubFile("model.bin", 4)]);
        Assert.Single(result);
        Assert.Equal(LocalModelFileState.Missing, result[0].State);
        Assert.Equal("unrelated user data", await File.ReadAllTextAsync(unrelated));
        Assert.Equal(before, File.GetLastWriteTimeUtc(unrelated));
        Assert.Single(Directory.GetFileSystemEntries(_root));
    }

    [Fact]
    public async Task InvalidPathsAndFilesystemConflictsArePerFileUnavailable()
    {
        await File.WriteAllBytesAsync(Path.Combine(_root, "good.bin"), [1]);
        await File.WriteAllBytesAsync(Path.Combine(_root, "file-parent"), [1]);
        Directory.CreateDirectory(Path.Combine(_root, "directory.bin"));
        var paths = new[] { "../outside.bin", "a/../../outside.bin", "C:/outside.bin", "a\\b", "CON.txt", "a/LPT1",
            "file.", "file ", "file:stream", "SHORT~1.bin", "name.part", "file-parent/child.bin", "directory.bin" };
        var files = paths.Select(path => new HubFile(path, 1)).Append(new HubFile("bad-size.bin", -1)).Append(new HubFile("good.bin", 1)).ToArray();
        var result = await _inventory.ScanAsync(_root, files);
        Assert.All(result.Take(result.Count - 1), item => Assert.Equal(LocalModelFileState.Unavailable, item.State));
        Assert.Equal(LocalModelFileState.Downloaded, result[^1].State);
        Assert.Equal(paths.Append("bad-size.bin").Append("good.bin"), result.Select(item => item.Path));
    }

    [Theory]
    [InlineData("")]
    [InlineData("relative/folder")]
    [InlineData("\\\\server\\share")]
    public async Task InvalidRootProducesStatusesRatherThanFailingTheWholeUiScan(string directory)
    {
        var result = await _inventory.ScanAsync(directory, [new HubFile("a.bin", 1), new HubFile("b.bin", 1)]);
        Assert.Equal(2, result.Count);
        Assert.All(result, item => Assert.Equal(LocalModelFileState.Unavailable, item.State));
    }

    [Fact]
    public async Task CancellationStopsQueuedAndRunningScansWithoutCreatingAnything()
    {
        using var alreadyCancelled = new CancellationTokenSource();
        alreadyCancelled.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => _inventory.ScanAsync(_root,
            [new HubFile("missing.bin", 1)], alreadyCancelled.Token));
        var many = Enumerable.Range(0, 100_000).Select(index => new HubFile($"missing-{index}.bin", 1)).ToArray();
        using var cancel = new CancellationTokenSource(TimeSpan.FromMilliseconds(10));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => _inventory.ScanAsync(_root, many, cancel.Token));
        Assert.Empty(Directory.GetFileSystemEntries(_root));
    }

    [Fact]
    public async Task InputListIsSnapshottedBeforeBackgroundWork()
    {
        var files = new List<HubFile> { new("original.bin", 1) };
        var scan = _inventory.ScanAsync(_root, files);
        files.Clear();
        files.Add(new HubFile("different.bin", 2));
        var result = await scan;
        Assert.Single(result);
        Assert.Equal("original.bin", result[0].Path);
    }

    [Fact]
    public async Task DirectoryJunctionCannotMakeExternalFinalFilesLookDownloaded()
    {
        var target = Path.Combine(_root, "target");
        var link = Path.Combine(_root, "link");
        Directory.CreateDirectory(target);
        await File.WriteAllTextAsync(Path.Combine(target, "model.bin"), "safe");
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
            var result = await _inventory.ScanAsync(_root, [new HubFile("link/model.bin", 4)]);
            Assert.Equal(LocalModelFileState.Unavailable, result[0].State);
            Assert.Equal("safe", await File.ReadAllTextAsync(Path.Combine(target, "model.bin")));
        }
        finally { Directory.Delete(link); }
    }
}
