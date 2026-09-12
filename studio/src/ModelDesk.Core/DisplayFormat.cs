using System.Globalization;

namespace ModelDesk.Core;

public static class DisplayFormat
{
    public static string Bytes(long value)
    {
        if (value < 0) return "—";
        string[] units = ["B", "KiB", "MiB", "GiB", "TiB"];
        double number = value;
        var index = 0;
        while (number >= 1024 && index < units.Length - 1) { number /= 1024; index++; }
        return $"{number.ToString(index == 0 ? "0" : "0.##", CultureInfo.InvariantCulture)} {units[index]}";
    }
    public static string Speed(double bytesPerSecond) => Bytes((long)Math.Max(0, bytesPerSecond)) + "/s";
}
