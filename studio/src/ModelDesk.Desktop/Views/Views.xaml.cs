using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using ModelDesk.Desktop.ViewModels;

namespace ModelDesk.Desktop.Views;

public partial class OverviewView : UserControl { public OverviewView() => InitializeComponent(); }
public partial class RunView : UserControl { public RunView() => InitializeComponent(); }
public partial class ValidationView : UserControl { public ValidationView() => InitializeComponent(); }
public partial class HubView : UserControl
{
    public HubView()
    {
        InitializeComponent();
        Loaded += async (_, _) =>
        {
            if (DataContext is HubViewModel { Detail: not null, IsBusy: false, IsScanning: false } model)
                await model.RefreshLocalFilesAsync();
        };
    }
    private void FileSelectionMouseDown(object sender, MouseButtonEventArgs args)
    {
        if (DataContext is not HubViewModel model || args.OriginalSource is not DependencyObject source) return;
        var row = Ancestor<DataGridRow>(source);
        if (row?.Item is not HubFileViewModel item) return;
        var checkbox = Ancestor<CheckBox>(source);
        var extend = Keyboard.Modifiers.HasFlag(ModifierKeys.Shift);
        if (checkbox is null && !extend) { model.SelectRange(item, item.Selected, false); return; }
        if (checkbox is { IsEnabled: false }) return;
        args.Handled = true;
        ModelFilesGrid.SelectedItem = item;
        checkbox?.Focus();
        model.SelectRange(item, checkbox is null || !item.Selected, extend);
    }
    private void FileSelectionKeyDown(object sender, KeyEventArgs args)
    {
        if (args.Key != Key.Space || DataContext is not HubViewModel model) return;
        var item = args.OriginalSource is DependencyObject source ? Ancestor<DataGridRow>(source)?.Item as HubFileViewModel : null;
        item ??= ModelFilesGrid.SelectedItem as HubFileViewModel;
        if (item is null) return;
        args.Handled = true;
        model.SelectRange(item, !item.Selected, Keyboard.Modifiers.HasFlag(ModifierKeys.Shift));
    }
    private static T? Ancestor<T>(DependencyObject? element) where T : DependencyObject
    {
        while (element is not null)
        {
            if (element is T match) return match;
            element = element is Visual ? VisualTreeHelper.GetParent(element) : LogicalTreeHelper.GetParent(element);
        }
        return null;
    }
}
public partial class DownloadsView : UserControl { public DownloadsView() => InitializeComponent(); }
public partial class ReportsView : UserControl { public ReportsView() => InitializeComponent(); }
public partial class SettingsView : UserControl
{
    public SettingsView() => InitializeComponent();
    private async void SaveToken(object sender, RoutedEventArgs args)
    {
        if (DataContext is not SettingsViewModel model) return;
        var token = TokenInput.Password; TokenInput.Clear(); await model.SaveTokenAsync(token);
    }
}
