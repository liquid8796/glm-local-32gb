using System.Windows;
using System.Windows.Controls;
using ModelDesk.Desktop.ViewModels;

namespace ModelDesk.Desktop.Views;

public partial class OverviewView : UserControl { public OverviewView() => InitializeComponent(); }
public partial class RunView : UserControl { public RunView() => InitializeComponent(); }
public partial class ValidationView : UserControl { public ValidationView() => InitializeComponent(); }
public partial class HubView : UserControl { public HubView() => InitializeComponent(); }
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
