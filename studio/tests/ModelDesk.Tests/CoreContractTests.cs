using ModelDesk.Core;

namespace ModelDesk.Tests;

public sealed class CoreContractTests
{
    [Fact]
    public void ArgumentsKeepPathsUnicodeAndShellCharactersLiteral()
    {
        Assert.Equal(["--prompt", "Xin chào $(whoami) & echo", "--model-directory", @"D:\Mô hình\test", ""],
            ArgumentTokenizer.Parse("--prompt \"Xin chào $(whoami) & echo\" --model-directory 'D:\\Mô hình\\test' \"\""));
    }

    [Theory]
    [InlineData("\"not closed")]
    [InlineData("'not closed")]
    [InlineData("a\nb")]
    [InlineData("a\0b")]
    public void InvalidArgumentInputFailsBeforeLaunching(string input) =>
        Assert.Throws<ArgumentException>(() => ArgumentTokenizer.Parse(input));

    [Fact]
    public void OptionalCapsCannotExceedThePythonContract()
    {
        new AppSettings().Validate();
        Assert.Throws<ArgumentOutOfRangeException>(() => new AppSettings { CpuLimitPercent = 71 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new AppSettings { RamBudgetBytes = 33_000_000_000 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new AppSettings { GpuTargetPercent = double.NaN }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new AppSettings { DownloadBytesPerSecond = -1 }.Validate());
    }

    [Fact]
    public void CurrentCoreOperationsAreExplicitAndDistinct()
    {
        Assert.Equal(16, CoreOperations.All.Count);
        Assert.Equal(CoreOperations.All.Count, CoreOperations.All.Select(item => item.Id).Distinct().Count());
        Assert.Equal(CoreOperationKind.Python, CoreOperations.Get("generate").Kind);
        Assert.Throws<ArgumentException>(() => CoreOperations.Get("arbitrary-shell-command"));
    }
}
