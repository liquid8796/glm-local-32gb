using System.Security.Cryptography;
using System.Text.Json;

namespace ModelDesk.Infrastructure.Settings;

internal static class LocalFiles
{
    internal static string StateDirectory(string? value) => Path.GetFullPath(value ??
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "ModelDesk"));

    internal static string CheckPath(string path)
    {
        var full = Path.GetFullPath(path);
        var current = Path.GetPathRoot(full)!;
        foreach (var part in full[current.Length..].Split(Path.DirectorySeparatorChar, StringSplitOptions.RemoveEmptyEntries))
        {
            current = Path.Combine(current, part);
            if ((File.Exists(current) || Directory.Exists(current)) &&
                (File.GetAttributes(current) & FileAttributes.ReparsePoint) != 0)
                throw new IOException("Links and reparse points are not supported for application data or reports.");
        }
        return full;
    }

    internal static string Within(string path, string root)
    {
        var full = CheckPath(path);
        var relative = Path.GetRelativePath(CheckPath(root), full);
        if (Path.IsPathRooted(relative) || relative == ".." || relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal))
            throw new IOException("The requested path is outside the allowed directory.");
        return full;
    }

    internal static async Task<byte[]> ReadBoundedAsync(string path, int maximum, CancellationToken cancellationToken = default)
    {
        CheckPath(path);
        await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete,
            65536, FileOptions.Asynchronous | FileOptions.SequentialScan);
        if (stream.Length > maximum) throw new InvalidDataException($"File exceeds the {maximum:N0}-byte limit.");
        using var destination = new MemoryStream((int)stream.Length);
        var buffer = new byte[65536];
        int count;
        while ((count = await stream.ReadAsync(buffer, cancellationToken)) != 0)
        {
            if (destination.Length + count > maximum) throw new InvalidDataException("File grew beyond its size limit.");
            await destination.WriteAsync(buffer.AsMemory(0, count), cancellationToken);
        }
        return destination.ToArray();
    }

    internal static async Task AtomicWriteAsync(string path, byte[] bytes, CancellationToken cancellationToken = default)
    {
        path = CheckPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var temporary = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            await File.WriteAllBytesAsync(temporary, bytes, cancellationToken);
            cancellationToken.ThrowIfCancellationRequested();
            File.Move(temporary, path, true);
        }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
    }

    internal static string Hash(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes));
    internal static JsonSerializerOptions JsonOptions { get; } = new() { WriteIndented = true, PropertyNameCaseInsensitive = true, MaxDepth = 64 };
}
