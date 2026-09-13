using System.Windows;
using System.Windows.Controls;

namespace ModelDesk.Desktop.Presentation;

/// <summary>Follow appended conversation text only while the reader is at the end.</summary>
public static class ConversationScroll
{
    public static readonly DependencyProperty FollowLatestProperty = DependencyProperty.RegisterAttached(
        "FollowLatest", typeof(bool), typeof(ConversationScroll), new PropertyMetadata(false, Changed));
    private static readonly DependencyProperty FollowingProperty = DependencyProperty.RegisterAttached(
        "Following", typeof(bool), typeof(ConversationScroll), new PropertyMetadata(true));
    public static bool GetFollowLatest(DependencyObject value) => (bool)value.GetValue(FollowLatestProperty);
    public static void SetFollowLatest(DependencyObject value, bool enabled) => value.SetValue(FollowLatestProperty, enabled);
    private static void Changed(DependencyObject value, DependencyPropertyChangedEventArgs args)
    {
        if (value is not ScrollViewer viewer) return;
        viewer.ScrollChanged -= Scrolled;
        if ((bool)args.NewValue) { viewer.SetValue(FollowingProperty, true); viewer.ScrollChanged += Scrolled; }
    }
    private static void Scrolled(object sender, ScrollChangedEventArgs args)
    {
        if (sender is not ScrollViewer viewer || args.OriginalSource != viewer) return;
        if (args.ExtentHeightChange != 0 || args.ViewportHeightChange != 0)
        {
            if ((bool)viewer.GetValue(FollowingProperty)) viewer.ScrollToEnd();
        }
        else viewer.SetValue(FollowingProperty, viewer.ScrollableHeight - viewer.VerticalOffset <= 2);
    }
}
