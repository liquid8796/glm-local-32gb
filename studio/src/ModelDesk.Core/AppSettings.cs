namespace ModelDesk.Core;

public sealed record AppSettings
{
    public string ProjectRoot { get; init; } = "";
    public string PythonExecutable { get; init; } = "";
    public string DefaultDownloadDirectory { get; init; } = "";
    public string RuntimeModelDirectory { get; init; } = "";
    public string Profile { get; init; } = "nvfp4";
    public string Theme { get; init; } = "Dark";
    public long DownloadBytesPerSecond { get; init; }
    public int ConcurrentDownloads { get; init; } = 2;
    public int DownloadConnectionsPerFile { get; init; } = 4;
    public long RamBudgetBytes { get; init; } = 32_000_000_000;
    public int CpuLimitPercent { get; init; } = 70;
    public int GpuIndex { get; init; }
    public double GpuTargetPercent { get; init; } = 60;
    public int GpuWindowSeconds { get; init; } = 10;

    public void Validate()
    {
        if (DownloadBytesPerSecond is < 0 or > 1_099_511_627_776) throw new ArgumentOutOfRangeException(nameof(DownloadBytesPerSecond));
        if (ConcurrentDownloads is < 1 or > 8) throw new ArgumentOutOfRangeException(nameof(ConcurrentDownloads), "Choose 1 to 8 parallel downloads.");
        if (DownloadConnectionsPerFile is < 1 or > 4) throw new ArgumentOutOfRangeException(nameof(DownloadConnectionsPerFile), "Choose 1 to 4 connections per large file.");
        if (RamBudgetBytes is < 1_000_000 or > 32_000_000_000) throw new ArgumentOutOfRangeException(nameof(RamBudgetBytes), "The Python core supports at most 32 GB.");
        if (CpuLimitPercent is < 1 or > 70) throw new ArgumentOutOfRangeException(nameof(CpuLimitPercent));
        if (!double.IsFinite(GpuTargetPercent) || GpuTargetPercent is < 1 or > 60) throw new ArgumentOutOfRangeException(nameof(GpuTargetPercent));
        if (GpuWindowSeconds is < 1 or > 3600 || GpuIndex < 0) throw new ArgumentOutOfRangeException(nameof(GpuWindowSeconds));
        if (Theme is not ("Dark" or "Light" or "System")) throw new ArgumentException("Unknown theme.");
        if (string.IsNullOrWhiteSpace(Profile)) throw new ArgumentException("Choose a model profile.");
    }
}

public interface ISettingsStore
{
    string FilePath { get; }
    Task<AppSettings> LoadAsync(CancellationToken cancellationToken = default);
    Task SaveAsync(AppSettings settings, CancellationToken cancellationToken = default);
}

public interface ICredentialStore
{
    Task<string?> GetTokenAsync(CancellationToken cancellationToken = default);
    Task SetTokenAsync(string? token, CancellationToken cancellationToken = default);
}
