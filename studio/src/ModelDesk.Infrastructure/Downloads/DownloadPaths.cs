using System.Text.RegularExpressions;
using ModelDesk.Core;
using ModelDesk.Infrastructure.Hub;

namespace ModelDesk.Infrastructure.Downloads;

internal static partial class DownloadPaths
{
    internal static string Resolve(DownloadRequest request)
    {
        HubHttp.ModelId(request.ModelId);
        HubHttp.ImmutableRevision(request.Revision);
        if (request.File.Size is < 0 or > 8_796_093_022_208 ||
            !(HubHttp.IsSha256(request.File.LfsSha256) || request.File.LfsSha256 is null && HubHttp.IsSha1(request.File.BlobId)))
            throw new ArgumentException("A bounded file size and LFS SHA-256 or Git blob SHA-1 are required.");
        var relative = request.File.Path;
        if (string.IsNullOrWhiteSpace(relative) || relative.Length > 4096 || relative.Contains('\\') || Path.IsPathRooted(relative))
            throw new ArgumentException("Download file path must be relative to its repository.");
        foreach (var part in relative.Split('/'))
        {
            if (part is "" or "." or ".." || part.Length > 255 || part.EndsWith(' ') || part.EndsWith('.') ||
                part.Any(character => char.IsControl(character) || "<>:\"|?*~".Contains(character)) ||
                ReservedName().IsMatch(part.Split('.')[0]))
                throw new ArgumentException("Download file path contains a Windows alias, reserved name or traversal segment.");
        }
        if (relative.EndsWith(".part", StringComparison.OrdinalIgnoreCase) || relative.EndsWith(".part.json", StringComparison.OrdinalIgnoreCase) ||
            relative.EndsWith(".download.lock", StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("Download path collides with a reserved transfer-state filename.");
        var root = Root(request.DestinationDirectory);
        var destination = Path.GetFullPath(Path.Combine(root, relative.Replace('/', Path.DirectorySeparatorChar)));
        if (!destination.StartsWith(Path.TrimEndingDirectorySeparator(root) + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("Download file resolves outside its destination directory.");
        CheckExisting(destination);
        return destination;
    }

    internal static string Root(string directory)
    {
        if (string.IsNullOrWhiteSpace(directory) || !Path.IsPathFullyQualified(directory) || directory.StartsWith("\\", StringComparison.Ordinal))
            throw new ArgumentException("Choose an absolute local drive directory for downloads.");
        var root = Path.GetFullPath(directory);
        CheckExisting(root);
        return root;
    }

    internal static void CheckExisting(string path)
    {
        var absolute = Path.GetFullPath(path);
        var current = Path.GetPathRoot(absolute) ?? throw new ArgumentException("Path has no volume root.");
        foreach (var segment in absolute[current.Length..].Split(Path.DirectorySeparatorChar, StringSplitOptions.RemoveEmptyEntries))
        {
            current = Path.Combine(current, segment);
            try
            {
                if ((File.GetAttributes(current) & FileAttributes.ReparsePoint) != 0)
                    throw new IOException("Download paths cannot contain symbolic links or reparse points.");
            }
            catch (FileNotFoundException) { }
            catch (DirectoryNotFoundException) { }
        }
    }

    internal static void CreateParent(string filename)
    {
        CheckExisting(filename);
        Directory.CreateDirectory(Path.GetDirectoryName(filename)!);
        CheckExisting(filename);
    }

    internal static void RegularFile(string filename)
    {
        CheckExisting(filename);
        if (File.Exists(filename) && (File.GetAttributes(filename) & FileAttributes.Directory) != 0 || Directory.Exists(filename))
            throw new IOException("A download file path is occupied by a directory.");
    }

    [GeneratedRegex("^(CON|PRN|AUX|NUL|CONIN\\$|CONOUT\\$|COM[1-9¹²³]|LPT[1-9¹²³])$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex ReservedName();
}
