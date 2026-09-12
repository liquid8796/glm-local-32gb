using System.Diagnostics;

namespace ModelDesk.Infrastructure.Downloads;

/// <summary>One shared token bucket for every transfer using a downloader instance.</summary>
internal sealed class AggregateRateLimiter
{
    private readonly object _sync = new();
    private readonly SemaphoreSlim _turn = new(1, 1);
    private long _limit;
    private long _last = Stopwatch.GetTimestamp();
    private double _credit;

    internal long Limit
    {
        get { lock (_sync) return _limit; }
        set
        {
            if (value is < 0 or > 1_099_511_627_776) throw new ArgumentOutOfRangeException(nameof(value));
            lock (_sync)
            {
                if (_limit == value) return;
                _limit = value;
                _credit = 0;
                _last = Stopwatch.GetTimestamp();
            }
        }
    }

    internal async ValueTask<int> AcquireAsync(int requested, CancellationToken cancellationToken)
    {
        if (Limit == 0) return requested;
        await _turn.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            lock (_sync)
            {
                if (_limit == 0) return requested;
                var now = Stopwatch.GetTimestamp();
                var burst = Math.Min(64 * 1024, Math.Max(1, _limit / 10.0));
                _credit = Math.Min(burst, _credit + Stopwatch.GetElapsedTime(_last, now).TotalSeconds * _limit);
                _last = now;
                var granted = (int)Math.Min(requested, Math.Floor(_credit));
                if (granted > 0)
                {
                    _credit -= granted;
                    return granted;
                }
            }
            // Short waits make a changed limit or pause responsive, including very low limits.
            await Task.Delay(20, cancellationToken).ConfigureAwait(false);
        }
        }
        finally { _turn.Release(); }
    }

    internal void Refund(int bytes)
    {
        if (bytes <= 0) return;
        lock (_sync)
        {
            if (_limit != 0) _credit = Math.Min(Math.Min(64 * 1024, Math.Max(1, _limit / 10.0)), _credit + bytes);
        }
    }
}
