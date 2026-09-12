using System.Security;
using ModelDesk.Core;

namespace ModelDesk.Infrastructure.Downloads;

/// <summary>Read-only filesystem metadata inventory. Downloaded means final-file
/// presence and expected length only; it does not certify its contents or hash.</summary>
public sealed class LocalModelFileInventory : ILocalModelFileInventory
{
    public Task<IReadOnlyList<LocalModelFileStatus>> ScanAsync(string directory, IReadOnlyList<HubFile> files,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(files);
        if (files.Count > 100_000) throw new ArgumentException("A local model scan cannot exceed 100,000 files.");
        var snapshot = files.ToArray();
        return Task.Run<IReadOnlyList<LocalModelFileStatus>>(() =>
        {
            var statuses = new LocalModelFileStatus[snapshot.Length];
            for (var index = 0; index < snapshot.Length; index++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                statuses[index] = Inspect(directory, snapshot[index]);
            }
            cancellationToken.ThrowIfCancellationRequested();
            return statuses;
        }, cancellationToken);
    }

    private static LocalModelFileStatus Inspect(string directory, HubFile file)
    {
        if (file is null) return new("", LocalModelFileState.Unavailable, Message: "Hub file metadata is absent.");
        try
        {
            if (file.Size is < 0 or > 8_796_093_022_208) throw new ArgumentException("Hub file size is invalid.");
            var final = DownloadPaths.ResolveFile(directory, file.Path);
            if (LengthOfRegularFile(final) is { } length)
            {
                return length == file.Size
                    ? new(file.Path, LocalModelFileState.Downloaded, length,
                        "Final file is present and its size matches Hugging Face metadata. Contents were not read; this scan does not verify the hash.")
                    : new(file.Path, LocalModelFileState.SizeMismatch, length,
                        $"Existing final file has {length:N0} bytes; Hugging Face expects {file.Size:N0}. This scan reads no contents, and existing files are not overwritten automatically.");
            }
            if (LengthOfRegularFile(final + ".part") is { } partialLength)
                return new(file.Path, LocalModelFileState.Partial, Message:
                    $"A partial transfer file exists with a length of {partialLength:N0} bytes. This may include preallocated holes and is not completed download progress. The final file is absent; no contents or hashes were read.");
            if (LengthOfRegularFile(final + ".part.json") is not null)
                return new(file.Path, LocalModelFileState.Partial, Message:
                    "Transfer metadata exists, but the final file is absent. No payload or resume manifest contents were read.");
            return new(file.Path, LocalModelFileState.Missing, 0, "No final file or partial transfer was found at this path.");
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException or SecurityException)
        {
            return new(file.Path ?? "", LocalModelFileState.Unavailable, Message: $"Cannot inspect this local file: {error.Message}");
        }
    }

    private static long? LengthOfRegularFile(string path)
    {
        try
        {
            DownloadPaths.CheckExisting(path);
            CheckDirectoryAncestors(path);
            var attributes = File.GetAttributes(path);
            if ((attributes & (FileAttributes.Directory | FileAttributes.ReparsePoint | FileAttributes.Device)) != 0)
                throw new IOException("The path is not a regular file or contains a symbolic link/reparse point.");
            var length = new FileInfo(path).Length;
            DownloadPaths.CheckExisting(path);
            return length;
        }
        catch (FileNotFoundException) { return null; }
        catch (DirectoryNotFoundException) { return null; }
    }

    private static void CheckDirectoryAncestors(string filename)
    {
        for (var parent = Path.GetDirectoryName(filename); parent is not null; parent = Path.GetDirectoryName(parent))
        {
            try
            {
                if ((File.GetAttributes(parent) & FileAttributes.Directory) == 0)
                    throw new IOException("A parent directory path is occupied by a regular file.");
            }
            catch (FileNotFoundException) { }
            catch (DirectoryNotFoundException) { }
        }
    }
}
