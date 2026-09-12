using System.Reflection;
using System.Runtime.ExceptionServices;
using ModelDesk.Infrastructure.Downloads;

namespace ModelDesk.Tests;

/// <summary>Exercise exact lease interleavings without scheduler/timing dependencies.</summary>
public sealed class DownloadDiskReservationTests
{
    private static readonly Type Reservations = typeof(ModelDownloader).Assembly.GetType(
        "ModelDesk.Infrastructure.Downloads.DiskReservations", throwOnError: true)!;
    private static readonly string Destination = Path.Combine(Path.GetTempPath(), "modeldesk-reservation-test.bin");

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FullyConsumedConcurrentLeasesCanDisposeInEitherOrder(bool reverse)
    {
        using var first = Acquire(65536);
        using var second = Acquire(65536);
        Consume(first, 65536);
        Consume(second, 65536);
        if (reverse) { second.Dispose(); first.Dispose(); }
        else { first.Dispose(); second.Dispose(); }
        // Idempotent finally/using cleanup must not mask a download's real exception.
        first.Dispose();
        second.Dispose();
        using var subsequent = Acquire(1024);
        Consume(subsequent, 1024);
    }

    [Fact]
    public void FullyConsumedSurvivorCanResizeAfterItsPeerIsDisposed()
    {
        using var first = Acquire(4096);
        using var second = Acquire(8192);
        Consume(first, 4096);
        Consume(second, 8192);
        first.Dispose();
        Resize(second, 65536);
        Consume(second, 32768);
        var available = new DriveInfo(Path.GetPathRoot(Destination)!).AvailableFreeSpace;
        Assert.Throws<IOException>(() => Resize(second, available + 1));
        // A rejected resize leaves the live lease valid and its accounting unchanged.
        Resize(second, 4096);
        Consume(second, 4096);
        second.Dispose();
    }

    [Fact]
    public void ZeroBytePreallocationLeasesRetainAccountingUntilTheirLastOwnerExits()
    {
        using var first = Acquire(0);
        using var second = Acquire(0);
        first.Dispose();
        using var third = Acquire(4096);
        Resize(second, 8192);
        Consume(third, 4096);
        third.Dispose();
        Consume(second, 8192);
        second.Dispose();
        Assert.Throws<ObjectDisposedException>(() => Resize(second, 1));
        Assert.Throws<ObjectDisposedException>(() => Consume(second, 1));
    }

    private static IDisposable Acquire(long bytes) => (IDisposable)Invoke(
        Reservations.GetMethod("Acquire", BindingFlags.Static | BindingFlags.NonPublic)!, null, [Destination, bytes])!;
    private static void Consume(IDisposable lease, long bytes) => Invoke(
        lease.GetType().GetMethod("Consumed", BindingFlags.Instance | BindingFlags.NonPublic)!, lease, [bytes]);
    private static void Resize(IDisposable lease, long bytes) => Invoke(
        lease.GetType().GetMethod("Resize", BindingFlags.Instance | BindingFlags.NonPublic)!, lease, [bytes]);

    private static object? Invoke(MethodInfo method, object? instance, object?[] arguments)
    {
        try { return method.Invoke(instance, arguments); }
        catch (TargetInvocationException error) when (error.InnerException is not null)
        {
            ExceptionDispatchInfo.Capture(error.InnerException).Throw();
            throw;
        }
    }
}
