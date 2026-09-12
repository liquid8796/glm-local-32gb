using System.Text;
using System.Text.Json;
using ModelDesk.Core;
using ModelDesk.Infrastructure.Settings;

namespace ModelDesk.Infrastructure.Reports;

public sealed class ReportService : IReportService
{
    public const int MaximumReadBytes = 4 * 1024 * 1024;
    private readonly HashSet<string> _roots = new(StringComparer.OrdinalIgnoreCase);

    public IReadOnlyList<ReportEntry> List(string projectRoot, string? profileReportDirectory = null)
    {
        var reportRoot = LocalFiles.CheckPath(Path.Combine(Path.GetFullPath(projectRoot), "reports"));
        var directory = profileReportDirectory is null ? reportRoot : LocalFiles.Within(
            Path.GetFullPath(profileReportDirectory, projectRoot), reportRoot);
        lock (_roots) _roots.Add(reportRoot);
        if (!Directory.Exists(directory)) return [];
        var pending = new Queue<(string Path, int Depth)>();
        pending.Enqueue((directory, 0));
        var results = new List<ReportEntry>();
        var visited = 0;
        while (pending.Count > 0 && visited++ < 512 && results.Count < 2048)
        {
            var current = pending.Dequeue();
            foreach (var path in Directory.EnumerateFileSystemEntries(current.Path).Take(4096))
            {
                try
                {
                    var attributes = File.GetAttributes(path);
                    if ((attributes & FileAttributes.ReparsePoint) != 0) continue;
                    if ((attributes & FileAttributes.Directory) != 0)
                    {
                        if (current.Depth < 5 && pending.Count < 512) pending.Enqueue((path, current.Depth + 1));
                        continue;
                    }
                    var extension = Path.GetExtension(path);
                    if (extension is not (".json" or ".md" or ".txt" or ".log")) continue;
                    var info = new FileInfo(path);
                    string? status = null;
                    if (extension == ".json" && info.Length <= 65536)
                    {
                        try
                        {
                            using var document = JsonDocument.Parse(LocalFiles.ReadBoundedAsync(
                                LocalFiles.Within(path, reportRoot), 65536).GetAwaiter().GetResult());
                            if (document.RootElement.ValueKind == JsonValueKind.Object &&
                                document.RootElement.TryGetProperty("status", out var value) && value.ValueKind == JsonValueKind.String)
                                status = value.GetString();
                        }
                        catch (JsonException) { status = "Invalid JSON"; }
                    }
                    results.Add(new ReportEntry(Path.GetRelativePath(reportRoot, path), Path.GetFullPath(path),
                        new DateTimeOffset(info.LastWriteTimeUtc), info.Length, status));
                    if (results.Count >= 2048) break;
                }
                catch (Exception exception) when (exception is IOException or UnauthorizedAccessException) { }
            }
        }
        return results.OrderByDescending(result => result.ModifiedAt).Take(512).ToArray();
    }

    public async Task<string> ReadAsync(string path, CancellationToken cancellationToken = default)
    {
        string[] roots;
        lock (_roots) roots = _roots.ToArray();
        if (!roots.Any(root => IsWithin(path, root)))
            throw new IOException("Select a report from this project's report list before reading it.");
        if (Path.GetExtension(path) is not (".json" or ".md" or ".txt" or ".log"))
            throw new InvalidDataException("Only text and JSON report files can be opened.");
        return new UTF8Encoding(false, true).GetString(await LocalFiles.ReadBoundedAsync(path, MaximumReadBytes, cancellationToken));
    }

    private static bool IsWithin(string path, string root)
    {
        try { LocalFiles.Within(path, root); return true; }
        catch (Exception exception) when (exception is IOException or ArgumentException) { return false; }
    }
}
