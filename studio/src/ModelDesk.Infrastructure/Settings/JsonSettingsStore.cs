using System.Text.Json;
using ModelDesk.Core;

namespace ModelDesk.Infrastructure.Settings;

public sealed class JsonSettingsStore : ISettingsStore
{
    public JsonSettingsStore(string? stateDirectory = null) => FilePath = Path.Combine(LocalFiles.StateDirectory(stateDirectory), "settings.json");
    public string FilePath { get; }

    public async Task<AppSettings> LoadAsync(CancellationToken cancellationToken = default)
    {
        if (!File.Exists(FilePath)) return CreateDefaults();
        try
        {
            var bytes = await LocalFiles.ReadBoundedAsync(FilePath, 65536, cancellationToken);
            var settings = JsonSerializer.Deserialize<AppSettings>(bytes, LocalFiles.JsonOptions)
                ?? throw new InvalidDataException("Settings JSON is null.");
            settings.Validate();
            return settings;
        }
        catch (Exception exception) when (exception is JsonException or ArgumentException)
        {
            throw new InvalidDataException($"Settings are invalid: {FilePath}. The file was preserved; correct it or choose a new settings directory.", exception);
        }
    }

    public Task SaveAsync(AppSettings settings, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(settings);
        settings.Validate();
        return LocalFiles.AtomicWriteAsync(FilePath, JsonSerializer.SerializeToUtf8Bytes(settings, LocalFiles.JsonOptions), cancellationToken);
    }

    public static string DiscoverProjectRoot()
    {
        var configured = Environment.GetEnvironmentVariable("MODEL_DESK_ROOT");
        if (!string.IsNullOrWhiteSpace(configured))
        {
            var full = Path.GetFullPath(configured);
            if (!File.Exists(Path.Combine(full, "glm_local", "__main__.py")))
                throw new DirectoryNotFoundException("MODEL_DESK_ROOT does not contain the Python core (glm_local/__main__.py).");
            return full;
        }
        foreach (var start in new[] { AppContext.BaseDirectory, Environment.CurrentDirectory })
        {
            for (var directory = new DirectoryInfo(start); directory is not null; directory = directory.Parent)
            {
                if (File.Exists(Path.Combine(directory.FullName, "glm_local", "__main__.py"))) return directory.FullName;
                var bundled = Path.Combine(directory.FullName, "core");
                if (File.Exists(Path.Combine(bundled, "glm_local", "__main__.py"))) return bundled;
            }
        }
        return "";
    }

    public static AppSettings CreateDefaults()
    {
        var root = DiscoverProjectRoot();
        var python = Path.Combine(root, ".venv-reference", "Scripts", "python.exe");
        var models = string.IsNullOrEmpty(root)
            ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "ModelDesk")
            : Path.Combine(root, "models");
        return new AppSettings { ProjectRoot = root, PythonExecutable = File.Exists(python) ? python : "python",
            DefaultDownloadDirectory = models, RuntimeModelDirectory = "", Profile = "nvfp4" };
    }
}
