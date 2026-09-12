using System.Diagnostics;
using System.Security.Cryptography;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using ModelDesk.Core;
using ModelDesk.Desktop;
using ModelDesk.Desktop.Presentation;
using ModelDesk.Desktop.ViewModels;
using Xunit.Abstractions;

namespace ModelDesk.Tests;

[CollectionDefinition("Headless WPF rendering", DisableParallelization = true)]
public sealed class HeadlessWpfRenderingCollection;

[Collection("Headless WPF rendering")]
public sealed class DesktopRenderingTests(ITestOutputHelper output)
{
    [Fact(Timeout = 60_000)]
    public Task Every_tab_renders_headlessly_with_initialized_profiles_at_both_sizes_and_themes()
        => OnApplicationDispatcher(async application =>
        {
            application.Resources.MergedDictionaries.Add(new ResourceDictionary
            {
                Source = new Uri("/ModelDesk;component/Themes/Controls.xaml", UriKind.Relative)
            });
            ThemeManager.Apply("Dark");
            var errors = new List<string>();
            var listener = new BindingErrors(errors);
            var bindingSource = PresentationTraceSources.DataBindingSource;
            var priorLevel = bindingSource.Switch.Level;
            bindingSource.Switch.Level = SourceLevels.Error;
            bindingSource.Listeners.Add(listener);
            DispatcherUnhandledExceptionEventHandler exceptionHandler = (_, args) =>
            {
                errors.Add(args.Exception.ToString()); args.Handled = true;
            };
            application.Dispatcher.UnhandledException += exceptionHandler;

            var root = FindProjectRoot();
            var destination = RenderDirectory(root);
            var core = new RenderCore(root);
            var hub = new RenderHub();
            var downloader = new NoDownloads();
            var settings = new AppSettings
            {
                ProjectRoot = root, PythonExecutable = Path.Combine(root, ".venv-reference", "Scripts", "python.exe"),
                DefaultDownloadDirectory = Path.Combine(root, "models"), Profile = "nvfp4"
            };
            var shell = new ShellViewModel(new RenderSettings(settings), new NoCredentials(), core,
                new RenderReports(root), hub, downloader, new RenderQueue(root));
            MainWindow? window = null;
            try
            {
                await shell.InitializeAsync();
                Assert.Equal(2, shell.Profiles.Count);
                Assert.Equal("nvfp4", shell.SelectedProfile?.Key);
                Assert.False(string.IsNullOrWhiteSpace(shell.PythonPath));
                Assert.Empty(shell.Error);

                // Only fake service calls populate reviewable rows; no HTTP/process adapter exists here.
                shell.Hub.SearchCommand.Execute(null);
                await DrainAsync();
                shell.Hub.OpenCommand.Execute(null);
                await DrainAsync();
                Assert.Equal(3, shell.Hub.Files.Count);
                Assert.All(shell.Downloads.Items.Where(item => item.State != DownloadState.Completed),
                    item => Assert.Equal(DownloadState.Paused, item.State));

                window = new MainWindow(shell);
                Assert.False(window.IsVisible);
                Assert.Null(PresentationSource.FromVisual(window));
                window.ApplyTemplate();

                // A never-shown Window has no HWND. Render its real XAML content in a detached
                // in-memory visual root so Window.Visibility does not suppress its drawing.
                var content = Assert.IsAssignableFrom<FrameworkElement>(window.Content);
                content.DataContext = shell;
                window.Content = null;
                var surface = new Border { Child = content };
                var frames = 0;
                string[] views = ["OverviewView", "RunView", "ValidationView", "HubView", "DownloadsView", "ReportsView", "SettingsView"];
                string[] slugs = ["overview", "run", "validation", "hub", "downloads", "reports", "settings"];
                var sharedPanelBrush = (Brush)application.FindResource("PanelBrush");
                foreach (var theme in new[] { "Dark", "Light" })
                {
                    ThemeManager.Apply(theme);
                    await DrainAsync();
                    Assert.Same(sharedPanelBrush, application.FindResource("PanelBrush"));
                    Assert.False(sharedPanelBrush.IsFrozen);
                    var expectedBackground = theme == "Dark" ? Color.FromRgb(0x12, 0x1A, 0x17) : Color.FromRgb(0xF1, 0xF4, 0xF1);
                    Assert.Equal(expectedBackground, Solid(window.Background));
                    Assert.True(Contrast(Solid(window.Foreground), expectedBackground) >= 4.5,
                        $"{theme}: window text contrast is insufficient.");
                    surface.Background = window.Background;
                    TextElement.SetForeground(surface, window.Foreground);
                    TextElement.SetFontFamily(surface, window.FontFamily);
                    TextElement.SetFontSize(surface, window.FontSize);

                    foreach (var size in new[] { new Size(1280, 800), new Size(1100, 720) })
                    {
                        Layout(surface, size);
                        var tabs = Assert.Single(Visuals<TabControl>(surface));
                        Assert.Equal(7, tabs.Items.Count);
                        var tabHashes = new HashSet<string>(StringComparer.Ordinal);
                        for (var index = 0; index < views.Length; index++)
                        {
                            tabs.SelectedIndex = index;
                            await DrainAsync();
                            Layout(surface, size);
                            await DrainAsync();
                            Layout(surface, size);

                            var page = Assert.Single(Visuals<UserControl>(surface), item => item.GetType().Namespace == "ModelDesk.Desktop.Views");
                            Assert.Equal(views[index], page.GetType().Name);
                            Assert.True(page.ActualWidth >= 800 && page.ActualHeight >= 200,
                                $"{theme} {size}: {views[index]} did not receive a usable layout ({page.ActualWidth} × {page.ActualHeight}).");
                            Assert.False(window.IsVisible);
                            Assert.Null(PresentationSource.FromVisual(window));
                            AssertProfileLabel(surface, shell);
                            AssertThemeSurfaces(surface, theme);
                            if (index == 3)
                            {
                                var files = Assert.Single(Visuals<DataGrid>(surface), grid => grid.Name == "ModelFilesGrid");
                                var position = files.TransformToAncestor(surface).Transform(new Point());
                                Assert.True(files.ActualHeight >= 100 && position.Y + files.ActualHeight <= size.Height - 50,
                                    $"{theme}/{size}: model files are below the initial viewport ({position.Y}, {files.ActualHeight}).");
                            }

                            var bitmap = new RenderTargetBitmap((int)size.Width, (int)size.Height, 96, 96, PixelFormats.Pbgra32);
                            bitmap.Render(surface);
                            var pixels = new byte[(int)size.Width * (int)size.Height * 4];
                            bitmap.CopyPixels(pixels, (int)size.Width * 4, 0);
                            Assert.Equal(expectedBackground.B, pixels[0]);
                            Assert.Equal(expectedBackground.G, pixels[1]);
                            Assert.Equal(expectedBackground.R, pixels[2]);
                            Assert.Equal(255, pixels[3]);
                            Assert.True(DistinctSampleColors(pixels) >= 12, $"{theme} {views[index]} rendered a blank/flat image.");
                            tabHashes.Add(Convert.ToHexString(SHA256.HashData(pixels)));

                            if (destination is not null)
                            {
                                var filename = $"headless-{theme.ToLowerInvariant()}-{(int)size.Width}x{(int)size.Height}-{index + 1:00}-{slugs[index]}.png";
                                var encoder = new PngBitmapEncoder(); encoder.Frames.Add(BitmapFrame.Create(bitmap));
                                using var stream = new FileStream(Path.Combine(destination, filename), FileMode.Create, FileAccess.Write, FileShare.None);
                                encoder.Save(stream);
                            }
                            AssertReadablePrimaryLabels(surface, theme, views[index]);
                            frames++;
                        }
                        Assert.Equal(7, tabHashes.Count);
                    }
                }
                Assert.Equal(28, frames);
                Assert.Equal(0, core.RunCalls);
                Assert.Equal(0, downloader.Calls);
                Assert.Equal(1, hub.SearchCalls);
                Assert.Equal(1, hub.DetailCalls);
                Assert.Empty(shell.Error);
                Assert.True(errors.Count == 0, "WPF binding/render errors:\n" + string.Join("\n", errors));
                output.WriteLine($"Rendered {frames} headless WPF content frames with fake model/report/download data. No Window.Show, HWND, real HTTP or Python process. PNG directory: {destination ?? "disabled"}.");
            }
            finally
            {
                await shell.CloseAsync();
                if (window is not null) { window.Close(); await DrainAsync(); }
                bindingSource.Listeners.Remove(listener);
                bindingSource.Switch.Level = priorLevel;
                application.Dispatcher.UnhandledException -= exceptionHandler;
                application.Resources.MergedDictionaries.Clear();
                application.Resources.Clear();
            }
        });

    private static void Layout(FrameworkElement surface, Size size)
    {
        surface.Width = size.Width; surface.Height = size.Height;
        surface.Measure(size); surface.Arrange(new Rect(size)); surface.UpdateLayout();
        foreach (var control in Visuals<Control>(surface)) control.ApplyTemplate();
        surface.UpdateLayout();
    }

    private static async Task DrainAsync()
    {
        await Dispatcher.CurrentDispatcher.InvokeAsync(() => { }, DispatcherPriority.ApplicationIdle);
        await Dispatcher.CurrentDispatcher.InvokeAsync(() => { }, DispatcherPriority.ApplicationIdle);
    }

    private static IEnumerable<T> Visuals<T>(DependencyObject root) where T : DependencyObject
    {
        for (var index = 0; index < VisualTreeHelper.GetChildrenCount(root); index++)
        {
            var child = VisualTreeHelper.GetChild(root, index);
            if (child is T match) yield return match;
            foreach (var descendant in Visuals<T>(child)) yield return descendant;
        }
    }

    private static Color Solid(Brush brush) => Assert.IsType<SolidColorBrush>(brush).Color;
    private static double Contrast(Color first, Color second)
    {
        static double Channel(byte value) { var s = value / 255d; return s <= .04045 ? s / 12.92 : Math.Pow((s + .055) / 1.055, 2.4); }
        static double Luminance(Color value) => .2126 * Channel(value.R) + .7152 * Channel(value.G) + .0722 * Channel(value.B);
        var a = Luminance(first); var b = Luminance(second);
        return (Math.Max(a, b) + .05) / (Math.Min(a, b) + .05);
    }

    private static int DistinctSampleColors(byte[] pixels)
    {
        var colors = new HashSet<uint>();
        for (var index = 0; index < pixels.Length; index += 4 * 17) colors.Add(BitConverter.ToUInt32(pixels, index));
        return colors.Count;
    }

    private static void AssertReadablePrimaryLabels(DependencyObject surface, string theme, string view)
    {
        var accent = Solid((Brush)Application.Current.FindResource("AccentBrush"));
        foreach (var button in Visuals<Button>(surface).Where(button => button.Background is SolidColorBrush brush && brush.Color == accent))
        {
            Assert.True(Contrast(Solid(button.Foreground), accent) >= 4.5, $"{theme}/{view}: primary button foreground is unreadable.");
            foreach (var label in Visuals<TextBlock>(button).Where(label => !string.IsNullOrWhiteSpace(label.Text)))
                Assert.True(Contrast(Solid(label.Foreground), accent) >= 4.5,
                    $"{theme}/{view}: primary label '{label.Text}' ignores its control foreground.");
        }
    }

    private static void AssertProfileLabel(DependencyObject surface, ShellViewModel shell)
    {
        var picker = Assert.Single(Visuals<ComboBox>(surface), combo => combo.Name == "ProfilePicker");
        var labels = Visuals<TextBlock>(picker).Select(label => label.Text).Where(text => !string.IsNullOrWhiteSpace(text)).ToArray();
        Assert.Contains(shell.SelectedProfile!.DisplayName, labels);
        Assert.DoesNotContain(labels, text => text.Contains("ModelProfile {", StringComparison.Ordinal) || text.Contains("DisplayName =", StringComparison.Ordinal));
        Assert.True(picker.ActualHeight <= 45, "The checkpoint label must remain a single line.");
        Assert.DoesNotContain(Visuals<TextBlock>(surface), label => label.Text.Contains("CoreOperation {", StringComparison.Ordinal));
    }

    private static void AssertThemeSurfaces(DependencyObject surface, string theme)
    {
        var panelColor = theme == "Dark" ? Color.FromRgb(0x1A, 0x25, 0x20) : Color.FromRgb(0xFC, 0xFD, 0xFC);
        var backgroundColor = theme == "Dark" ? Color.FromRgb(0x12, 0x1A, 0x17) : Color.FromRgb(0xF1, 0xF4, 0xF1);
        Assert.Equal(panelColor, Solid((Brush)Application.Current.FindResource("PanelBrush")));
        var panelStyle = (Style)Application.Current.FindResource("Panel");
        var panels = Visuals<Border>(surface).Where(border => ReferenceEquals(border.Style, panelStyle)).ToArray();
        var tables = Visuals<DataGrid>(surface).ToArray();
        Assert.True(panels.Length + tables.Length > 0, "The view must expose real panel/table surfaces.");
        foreach (var panel in panels) Assert.Equal(panelColor, Solid(panel.Background));
        foreach (var table in tables) Assert.Equal(panelColor, Solid(table.Background));
        foreach (var textBox in Visuals<TextBox>(surface)) Assert.Equal(backgroundColor, Solid(textBox.Background));
    }

    private static string FindProjectRoot()
    {
        foreach (var start in new[] { AppContext.BaseDirectory, Environment.CurrentDirectory })
            for (var directory = new DirectoryInfo(start); directory is not null; directory = directory.Parent)
                if (File.Exists(Path.Combine(directory.FullName, "glm_local", "__main__.py"))) return directory.FullName;
        throw new DirectoryNotFoundException("Headless rendering tests must run from the source repository.");
    }

    private static string? RenderDirectory(string root)
    {
        var requested = Environment.GetEnvironmentVariable("MODEL_DESK_RENDER_DIR");
        if (string.IsNullOrWhiteSpace(requested)) return null;
        var path = Path.GetFullPath(requested, root);
        var allowed = Path.GetFullPath(Path.Combine(root, "reports", "studio", "ui"));
        if (!path.Equals(allowed, StringComparison.OrdinalIgnoreCase) && !path.StartsWith(allowed + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("MODEL_DESK_RENDER_DIR must remain inside this repository's reports/studio/ui directory.");
        for (var directory = new DirectoryInfo(path); directory is not null && directory.FullName.Length >= root.Length; directory = directory.Parent)
            if (directory.Exists && (directory.Attributes & FileAttributes.ReparsePoint) != 0)
                throw new IOException("Render output cannot pass through directory links or reparse points.");
        Directory.CreateDirectory(path); return path;
    }

    private static async Task OnApplicationDispatcher(Func<Application, Task> test)
    {
        var completed = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var thread = new Thread(() =>
        {
            var dispatcher = Dispatcher.CurrentDispatcher;
            SynchronizationContext.SetSynchronizationContext(new DispatcherSynchronizationContext(dispatcher));
            Application? application = null;
            dispatcher.BeginInvoke(async () =>
            {
                try
                {
                    if (Application.Current is not null) throw new InvalidOperationException("Run the headless rendering test in a test host without an existing WPF Application.");
                    application = new Application { ShutdownMode = ShutdownMode.OnExplicitShutdown };
                    await test(application); completed.TrySetResult();
                }
                catch (Exception exception) { completed.TrySetException(exception); }
                finally
                {
                    application?.Shutdown();
                    if (!dispatcher.HasShutdownStarted) dispatcher.BeginInvokeShutdown(DispatcherPriority.Background);
                }
            });
            Dispatcher.Run();
        }) { IsBackground = true, Name = "ModelDesk headless renderer" };
        thread.SetApartmentState(ApartmentState.STA); thread.Start();
        try { await completed.Task.WaitAsync(TimeSpan.FromSeconds(50)); }
        finally { if (!thread.Join(TimeSpan.FromSeconds(5))) throw new TimeoutException("Headless WPF dispatcher did not shut down."); }
    }

    private sealed class BindingErrors(List<string> errors) : TraceListener
    {
        public override void Write(string? message) { if (!string.IsNullOrWhiteSpace(message)) errors.Add(message); }
        public override void WriteLine(string? message) => Write(message);
    }
    private sealed class RenderSettings(AppSettings settings) : ISettingsStore
    {
        public string FilePath => Path.Combine(settings.ProjectRoot, "reports", "studio", "ui", "fixture-settings.json");
        public Task<AppSettings> LoadAsync(CancellationToken cancellationToken = default) => Task.FromResult(settings);
        public Task SaveAsync(AppSettings value, CancellationToken cancellationToken = default) => Task.CompletedTask;
    }
    private sealed class NoCredentials : ICredentialStore
    {
        public Task<string?> GetTokenAsync(CancellationToken cancellationToken = default) => Task.FromResult<string?>(null);
        public Task SetTokenAsync(string? token, CancellationToken cancellationToken = default) => throw new InvalidOperationException("Rendering cannot change credentials.");
    }
    private sealed class RenderCore(string root) : IPythonCoreService
    {
        public int RunCalls { get; private set; }
        public IReadOnlyList<ModelProfile> GetProfiles(string projectRoot) => [
            new("nvfp4", "Render fixture · NVFP4", "render-fixture/GLM-NVFP4", new string('a', 40), "config.json", Path.Combine(root, "models", "render-fixture-nvfp4"), Path.Combine(root, "reports", "nvfp4")),
            new("fp8", "Render fixture · FP8", "render-fixture/GLM-FP8", new string('b', 40), "config.json", Path.Combine(root, "models", "render-fixture-fp8"), Path.Combine(root, "reports"))];
        public Task<CoreRunResult> RunAsync(CoreRunRequest request, IProgress<CoreOutput>? output = null, CancellationToken cancellationToken = default)
        { RunCalls++; throw new InvalidOperationException("Rendering cannot launch Python."); }
    }
    private sealed class RenderHub : IHuggingFaceClient
    {
        public int SearchCalls { get; private set; }
        public int DetailCalls { get; private set; }
        public Task<HubSearchResult> SearchAsync(string query, string? nextPage = null, CancellationToken cancellationToken = default)
        {
            SearchCalls++;
            return Task.FromResult(new HubSearchResult([new("render-fixture/GLM-NVFP4", "render-fixture", 0, 0, "text-generation", "transformers", false, false, null, ["render-test"])]));
        }
        public Task<HubModelDetail> GetModelAsync(string modelId, string revision = "main", CancellationToken cancellationToken = default)
        {
            DetailCalls++;
            return Task.FromResult(new HubModelDetail("render-fixture/GLM-NVFP4", new string('a', 40),
                [new("config.json", 4096, BlobId: new string('b', 40)), new("tokenizer_config.json", 2048, BlobId: new string('c', 40)), new("model.safetensors", 67_108_864, new string('d', 64))],
                "# Headless rendering fixture\nThese rows are test data; no remote model was queried or downloaded.", "text-generation", "transformers", false, false, 0, 0));
        }
    }
    private sealed class NoDownloads : IModelDownloader
    {
        public int Calls { get; private set; }
        public long BytesPerSecondLimit { get; set; }
        public Task<DownloadResult> DownloadAsync(DownloadRequest request, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
        { Calls++; throw new InvalidOperationException("Rendering cannot download files."); }
    }
    private sealed class RenderQueue(string root) : IDownloadQueueStore
    {
        public Task<IReadOnlyList<SavedDownload>> LoadAsync(CancellationToken cancellationToken = default) => Task.FromResult<IReadOnlyList<SavedDownload>>([
            new(Guid.Parse("11111111-1111-1111-1111-111111111111"), new("render-fixture/GLM-NVFP4", new string('a', 40), new("model.safetensors", 67_108_864, new string('d', 64)), Path.Combine(root, "models", "render-fixture")), DownloadState.Downloading, 16_777_216),
            new(Guid.Parse("22222222-2222-2222-2222-222222222222"), new("render-fixture/GLM-NVFP4", new string('a', 40), new("config.json", 4096, BlobId: new string('b', 40)), Path.Combine(root, "models", "render-fixture")), DownloadState.Completed, 4096)]);
        public async Task SaveAsync(IReadOnlyList<SavedDownload> downloads, CancellationToken cancellationToken = default) => await Task.Yield();
    }
    private sealed class RenderReports(string root) : IReportService
    {
        public IReadOnlyList<ReportEntry> List(string projectRoot, string? profileReportDirectory = null) => [
            new("nvfp4/latest.json", Path.Combine(root, "reports", "nvfp4", "latest.json"), new DateTimeOffset(2026, 9, 12, 12, 0, 0, TimeSpan.Zero), 512, "RENDER FIXTURE"),
            new("nvfp4/metadata-latest.json", Path.Combine(root, "reports", "nvfp4", "metadata-latest.json"), new DateTimeOffset(2026, 9, 12, 11, 0, 0, TimeSpan.Zero), 1024, "RENDER FIXTURE")];
        public Task<string> ReadAsync(string path, CancellationToken cancellationToken = default) => Task.FromResult("{\n  \"scope\": \"headless rendering fixture\"\n}");
    }
}
