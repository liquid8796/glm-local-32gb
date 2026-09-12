using System.Diagnostics;
using System.Text.RegularExpressions;

namespace ModelDesk.Desktop.Presentation;

public static partial class DiagnosticLog
{
    private static readonly object Gate = new();
    private static string? file;
    public static void Initialize(string stateDirectory)
    {
        Directory.CreateDirectory(stateDirectory);
        file = Path.Combine(stateDirectory, "startup.log");
        PresentationTraceSources.DataBindingSource.Switch.Level = SourceLevels.Error;
        PresentationTraceSources.DataBindingSource.Listeners.Add(new BindingListener());
        Write("Application startup");
    }
    public static void Write(string message)
    {
        if (file is null) return;
        lock (Gate)
        {
            try
            {
                if (File.Exists(file) && new FileInfo(file).Length > 262_144) File.Move(file, file + ".previous", true);
                var safe = TokenPattern().Replace(message, "[redacted]");
                if (safe.Length > 12_000) safe = safe[..12_000];
                File.AppendAllText(file, $"{DateTimeOffset.Now:O} {safe}{Environment.NewLine}");
            }
            catch (Exception exception) when (exception is IOException or UnauthorizedAccessException) { Debug.WriteLine("Unable to write ModelDesk diagnostics."); }
        }
    }
    [GeneratedRegex(@"hf_[A-Za-z0-9]+|(?i:Bearer)\s+[^\s\""']+")]
    private static partial Regex TokenPattern();
    private sealed class BindingListener : TraceListener
    {
        public override void Write(string? message) { if (!string.IsNullOrWhiteSpace(message)) DiagnosticLog.Write("Binding: " + message); }
        public override void WriteLine(string? message) => Write(message);
    }
}
