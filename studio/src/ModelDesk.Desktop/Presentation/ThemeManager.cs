using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.ComponentModel;
using System.Windows.Data;
using Microsoft.Win32;

namespace ModelDesk.Desktop.Presentation;

public static class ThemeManager
{
    private static string selected = "Dark";
    private static readonly Dictionary<string, PaletteColor> Palette = [];
    static ThemeManager() => SystemEvents.UserPreferenceChanged += (_, _) => Application.Current?.Dispatcher.Invoke(() => Apply(selected));
    public static void Apply(string theme)
    {
        selected = theme;
        bool light = theme == "Light";
        if (theme == "System")
        {
            using var key = Registry.CurrentUser.OpenSubKey(@"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize");
            light = key?.GetValue("AppsUseLightTheme") is int value && value != 0;
        }
        string[] names = ["BackgroundBrush", "PanelBrush", "RaisedBrush", "BorderBrush", "TextBrush", "MutedBrush", "AccentBrush", "AccentSoftBrush", "DangerBrush", "AccentTextBrush"];
        string[] colors = light ? ["#F1F4F1", "#FCFDFC", "#E6EDE8", "#CCD9D0", "#23372D", "#596F61", "#247653", "#DCEEE2", "#AB3F31", "#FFFFFF"]
            : ["#121A17", "#1A2520", "#24312A", "#34463B", "#E7EFE9", "#9CB1A3", "#85CEAB", "#2B4435", "#F3AA98", "#142D20"];
        for (int index = 0; index < names.Length; index++)
        {
            var color = (Color)ColorConverter.ConvertFromString(colors[index]);
            if (!Palette.TryGetValue(names[index], out var entry))
                Palette[names[index]] = entry = new PaletteColor(color);
            // A bound Color prevents WPF from freezing a style's shared brush. Updating
            // the source changes every existing brush consumer without replacing it.
            var brush = Application.Current.Resources[names[index]] as SolidColorBrush;
            var replace = brush is null || brush.IsFrozen;
            if (replace) brush = new SolidColorBrush();
            if (!BindingOperations.IsDataBound(brush, SolidColorBrush.ColorProperty))
                BindingOperations.SetBinding(brush, SolidColorBrush.ColorProperty,
                    new Binding(nameof(PaletteColor.Value)) { Source = entry, Mode = BindingMode.OneWay });
            entry.Value = color;
            if (replace) Application.Current.Resources[names[index]] = brush;
        }
    }

    private sealed class PaletteColor(Color initial) : INotifyPropertyChanged
    {
        private Color value = initial;
        public Color Value
        {
            get => value;
            set { if (this.value == value) return; this.value = value; PropertyChanged?.Invoke(this, new(nameof(Value))); }
        }
        public event PropertyChangedEventHandler? PropertyChanged;
    }
}

public static class HoverMotion
{
    public static readonly DependencyProperty EnabledProperty = DependencyProperty.RegisterAttached("Enabled", typeof(bool), typeof(HoverMotion), new PropertyMetadata(false, Changed));
    public static void SetEnabled(DependencyObject element, bool value) => element.SetValue(EnabledProperty, value);
    public static bool GetEnabled(DependencyObject element) => (bool)element.GetValue(EnabledProperty);
    private static void Changed(DependencyObject source, DependencyPropertyChangedEventArgs args)
    {
        if (source is not UIElement element || args.NewValue is not true) return;
        element.MouseEnter += (_, _) => Animate(element, .88);
        element.MouseLeave += (_, _) => Animate(element, 1);
    }
    private static void Animate(UIElement element, double value) => element.BeginAnimation(UIElement.OpacityProperty,
        new DoubleAnimation(value, TimeSpan.FromMilliseconds(SystemParameters.ClientAreaAnimation ? 150 : 0)));
}
