using System.Text;
using ModelDesk.Core;
using ModelDesk.Infrastructure.Settings;

namespace ModelDesk.Tests;

public sealed class SettingsTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ModelDesk-settings-tests", Guid.NewGuid().ToString("N"));

    [Fact]
    public async Task DefaultsDoNotCreateFilesAndSavedValuesRoundTripAtomically()
    {
        var store = new JsonSettingsStore(_directory);
        var defaults = await store.LoadAsync();
        Assert.Equal("nvfp4", defaults.Profile);
        Assert.False(string.IsNullOrWhiteSpace(defaults.PythonExecutable));
        Assert.False(File.Exists(store.FilePath));
        var changed = defaults with { CpuLimitPercent = 20, ConcurrentDownloads = 3, GpuTargetPercent = 45,
            DownloadBytesPerSecond = 1024 * 1024, Theme = "Light", RuntimeModelDirectory = "D:\\Models with spaces" };
        await store.SaveAsync(changed);
        Assert.Equal(changed, await store.LoadAsync());
        Assert.Empty(Directory.GetFiles(_directory, "*.tmp"));
    }

    [Fact]
    public async Task MalformedAndInvalidSettingsAreReportedAndNeverOverwritten()
    {
        var store = new JsonSettingsStore(_directory);
        Directory.CreateDirectory(_directory);
        foreach (var invalid in new[] { "{broken", "{\"CpuLimitPercent\":90}", "{\"Theme\":\"Unknown\"}" })
        {
            await File.WriteAllTextAsync(store.FilePath, invalid);
            await Assert.ThrowsAsync<InvalidDataException>(() => store.LoadAsync());
            Assert.Equal(invalid, await File.ReadAllTextAsync(store.FilePath));
        }
    }

    [Fact]
    public async Task InvalidOrCancelledSaveLeavesPreviousSettingsIntact()
    {
        var store = new JsonSettingsStore(_directory);
        var original = new AppSettings { CpuLimitPercent = 15 };
        await store.SaveAsync(original);
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => store.SaveAsync(original with { CpuLimitPercent = 99 }));
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => store.SaveAsync(original with { CpuLimitPercent = 25 }, cancellation.Token));
        Assert.Equal(original, await store.LoadAsync());
        Assert.Empty(Directory.GetFiles(_directory, "*.tmp"));
    }

    [Fact]
    public async Task CurrentUserTokenIsEncryptedAndCanBeCleared()
    {
        var store = new WindowsCredentialStore(_directory, useEnvironment: false);
        const string token = "hf_synthetic_modeldesk_unit_test_only";
        await store.SetTokenAsync(token);
        var bytes = await File.ReadAllBytesAsync(Path.Combine(_directory, "huggingface.token.dpapi"));
        Assert.DoesNotContain(token, Encoding.UTF8.GetString(bytes));
        Assert.Equal(token, await store.GetTokenAsync());
        await store.SetTokenAsync(null);
        Assert.Null(await store.GetTokenAsync());
    }

    [Fact]
    public async Task TokenControlCharactersAreRejectedWithoutWritingASecret()
    {
        var store = new WindowsCredentialStore(_directory, useEnvironment: false);
        await Assert.ThrowsAsync<ArgumentException>(() => store.SetTokenAsync("hf_fake\r\nheader"));
        Assert.False(File.Exists(Path.Combine(_directory, "huggingface.token.dpapi")));
    }

    public void Dispose() { if (Directory.Exists(_directory)) Directory.Delete(_directory, true); }
}
