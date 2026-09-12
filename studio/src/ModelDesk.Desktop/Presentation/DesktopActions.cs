using System.Diagnostics;
using Microsoft.Win32;

namespace ModelDesk.Desktop.Presentation;

public static class DesktopActions
{
    public static string? ChooseFolder(string title, string? initial = null)
    {
        var dialog = new OpenFolderDialog { Title = title, Multiselect = false };
        if (Directory.Exists(initial)) dialog.InitialDirectory = initial;
        return dialog.ShowDialog() == true ? dialog.FolderName : null;
    }
    public static string? ChooseFile(string title, string filter)
    {
        var dialog = new OpenFileDialog { Title = title, Filter = filter, CheckFileExists = true };
        return dialog.ShowDialog() == true ? dialog.FileName : null;
    }
    public static void OpenFolder(string path)
    {
        var directory = File.Exists(path) ? Path.GetDirectoryName(path)! : path;
        if (!Directory.Exists(directory)) throw new DirectoryNotFoundException("Thư mục chưa tồn tại: " + directory);
        Process.Start(new ProcessStartInfo(directory) { UseShellExecute = true });
    }
}
