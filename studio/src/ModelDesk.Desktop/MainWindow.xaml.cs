using System.ComponentModel;
using System.Windows;
using ModelDesk.Desktop.ViewModels;

namespace ModelDesk.Desktop;

public partial class MainWindow : Window
{
    private readonly ShellViewModel model;
    private bool closing;
    private bool stopping;
    public MainWindow(ShellViewModel model)
    {
        InitializeComponent(); this.model = model; DataContext = model;
        var work = SystemParameters.WorkArea;
        Width = Math.Min(1340, Math.Max(900, work.Width - 24)); Height = Math.Min(870, Math.Max(600, work.Height - 24));
        MinWidth = Math.Min(1100, work.Width - 24); MinHeight = Math.Min(720, work.Height - 24);
        Closing += OnClosing;
    }
    private async void OnClosing(object? sender, CancelEventArgs args)
    {
        if (closing) return; args.Cancel = true;
        if (stopping) return;
        stopping = true;
        try { await model.CloseAsync(); } catch (Exception exception) { model.ShowError(exception); }
        closing = true;
        // CloseAsync can finish synchronously for an idle window. Schedule the
        // final close after this Closing event has unwound instead of re-entering it.
        _ = Dispatcher.BeginInvoke(new Action(Close));
    }
}
