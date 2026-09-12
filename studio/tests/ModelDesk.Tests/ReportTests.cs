using ModelDesk.Infrastructure.Reports;

namespace ModelDesk.Tests;

public sealed class ReportTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ModelDesk-report-tests", Guid.NewGuid().ToString("N"));
    private readonly ReportService _service = new();

    [Fact]
    public async Task ListsStatusesAndReadsOnlyBoundedTextReports()
    {
        var directory = Path.Combine(_root, "reports", "nvfp4");
        Directory.CreateDirectory(directory);
        var path = Path.Combine(directory, "latest.json");
        await File.WriteAllTextAsync(path, "{\"status\":\"BLOCKED\"}");
        await File.WriteAllTextAsync(Path.Combine(directory, "tensor.safetensors"), "not a report");
        var entries = _service.List(_root, directory);
        Assert.Single(entries);
        Assert.Equal("BLOCKED", entries[0].Status);
        Assert.Contains("BLOCKED", await _service.ReadAsync(path));
    }

    [Fact]
    public async Task EscapingOrUnselectedRootCannotBeRead()
    {
        Directory.CreateDirectory(_root);
        var path = Path.Combine(_root, "secret.json");
        await File.WriteAllTextAsync(path, "synthetic-private-file");
        _service.List(_root);
        await Assert.ThrowsAsync<IOException>(() => _service.ReadAsync(path));
        Assert.Throws<IOException>(() => _service.List(_root, Path.Combine(_root, "..")));
    }

    [Fact]
    public async Task LargeReportIsNotLoaded()
    {
        var directory = Path.Combine(_root, "reports");
        Directory.CreateDirectory(directory);
        var path = Path.Combine(directory, "large.log");
        await using (var stream = File.Create(path)) stream.SetLength(ReportService.MaximumReadBytes + 1);
        _service.List(_root);
        await Assert.ThrowsAsync<InvalidDataException>(() => _service.ReadAsync(path));
    }

    [Fact]
    public void ReparseDirectoryCannotEscapeTheReportTree()
    {
        var directory = Path.Combine(_root, "reports");
        Directory.CreateDirectory(directory);
        var outside = Path.Combine(_root, "outside");
        Directory.CreateDirectory(outside);
        File.WriteAllText(Path.Combine(outside, "private.json"), "{}");
        var link = Path.Combine(directory, "linked");
        try { Directory.CreateSymbolicLink(link, outside); }
        catch (UnauthorizedAccessException) { return; } // Windows without Developer Mode cannot create the test link.
        Assert.Empty(_service.List(_root));
        Assert.Throws<IOException>(() => _service.List(_root, link));
    }

    public void Dispose() { if (Directory.Exists(_root)) Directory.Delete(_root, true); }
}
