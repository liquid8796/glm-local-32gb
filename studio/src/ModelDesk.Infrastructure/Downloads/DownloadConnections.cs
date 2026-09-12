namespace ModelDesk.Infrastructure.Downloads;

/// <summary>Process-wide transport admission shared by sequential and segmented transfers.</summary>
internal static class DownloadConnections
{
    internal const int Maximum = 8;
    private static readonly SemaphoreSlim Slots = new(Maximum, Maximum);

    internal static async ValueTask<IDisposable> AcquireAsync(CancellationToken cancellationToken)
    {
        await Slots.WaitAsync(cancellationToken).ConfigureAwait(false);
        return new Lease();
    }

    private sealed class Lease : IDisposable
    {
        private int _disposed;
        public void Dispose() { if (Interlocked.Exchange(ref _disposed, 1) == 0) Slots.Release(); }
    }
}
