using System.ComponentModel;
using System.Runtime.InteropServices;

namespace ModelDesk.Infrastructure.Downloads;

internal static class DiskReservations
{
    private static readonly object Sync = new();
    private static readonly Dictionary<string, long> Remaining = new(StringComparer.OrdinalIgnoreCase);
    private static readonly Dictionary<string, int> ActiveLeases = new(StringComparer.OrdinalIgnoreCase);
    private const long Reserve = 16 * 1024 * 1024;

    internal static long AllocatedBytes(string path)
    {
        if (!File.Exists(path)) return 0;
        DownloadPaths.RegularFile(path);
        Marshal.SetLastPInvokeError(0);
        var low = GetCompressedFileSize(path, out var high);
        if (low == uint.MaxValue && Marshal.GetLastPInvokeError() != 0)
            throw new IOException("Could not determine the partial file's allocated disk space.", new Win32Exception(Marshal.GetLastPInvokeError()));
        return Math.Min(new FileInfo(path).Length, ((long)high << 32) | low);
    }

    [DllImport("kernel32.dll", EntryPoint = "GetCompressedFileSizeW", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern uint GetCompressedFileSize(string filename, out uint highSize);

    internal static Lease Acquire(string path, long bytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(bytes);
        var volume = Path.GetPathRoot(path)!;
        lock (Sync)
        {
            var existing = Remaining.GetValueOrDefault(volume);
            if (new DriveInfo(volume).AvailableFreeSpace < checked(existing + bytes + Reserve))
                throw new IOException("The destination drive does not have enough free space for the selected active downloads.");
            Remaining[volume] = checked(existing + bytes);
            ActiveLeases[volume] = checked(ActiveLeases.GetValueOrDefault(volume) + 1);
            return new Lease(volume, bytes);
        }
    }

    internal sealed class Lease(string volume, long bytes) : IDisposable
    {
        private long _bytes = bytes;
        private bool _disposed;

        internal void Consumed(long count)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(count);
            lock (Sync)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                var consumed = Math.Min(count, _bytes);
                _bytes -= consumed;
                Remaining[volume] -= consumed;
            }
        }

        internal void Resize(long count)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(count);
            lock (Sync)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                var others = Remaining[volume] - _bytes;
                if (new DriveInfo(volume).AvailableFreeSpace < checked(others + count + Reserve))
                    throw new IOException("Insufficient disk space to restart the partial download.");
                Remaining[volume] = checked(others + count);
                _bytes = count;
            }
        }

        public void Dispose()
        {
            lock (Sync)
            {
                if (_disposed) return;
                Remaining[volume] -= _bytes;
                // A fully consumed lease still owns a lifetime: a parallel transfer
                // may dispose or resize another zero-byte lease after this one ends.
                // Byte totals alone cannot tell whether the shared volume is unused.
                var active = ActiveLeases[volume] - 1;
                if (active == 0)
                {
                    Remaining.Remove(volume);
                    ActiveLeases.Remove(volume);
                }
                else ActiveLeases[volume] = active;
                _bytes = 0;
                _disposed = true;
            }
        }
    }
}
