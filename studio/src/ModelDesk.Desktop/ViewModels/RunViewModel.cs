using System.Globalization;
using System.Windows.Input;
using ModelDesk.Core;
using ModelDesk.Desktop.Presentation;

namespace ModelDesk.Desktop.ViewModels;

public sealed class RunViewModel : ObservableObject
{
    private readonly CoreTaskViewModel task;
    private string directory = "", prompt = "", tokens = "", backend = "cpu", context = "4096", generate = "32", timeout = "1800", advanced = "";
    private bool useTokens;
    public RunViewModel(CoreTaskViewModel task)
    {
        this.task = task;
        PlanCommand = new AsyncCommand(_ => task.RunAsync("runtime-plan", Arguments(false)), task.ShowError, _ => task.IsAvailable && !task.IsBusy);
        GenerateCommand = new AsyncCommand(_ => task.RunAsync("generate", Arguments(true)), task.ShowError, _ => task.IsAvailable && !task.IsBusy);
        BrowseCommand = new RelayCommand(_ => ModelDirectory = DesktopActions.ChooseFolder("Chọn thư mục trọng số", ModelDirectory) ?? ModelDirectory);
    }
    public string ModelDirectory { get => directory; set => Set(ref directory, value); }
    public string Prompt { get => prompt; set => Set(ref prompt, value); }
    public string Tokens { get => tokens; set => Set(ref tokens, value); }
    public bool UseTokens { get => useTokens; set => Set(ref useTokens, value); }
    public string Backend { get => backend; set => Set(ref backend, value); }
    public string Context { get => context; set => Set(ref context, value); }
    public string Generate { get => generate; set => Set(ref generate, value); }
    public string Timeout { get => timeout; set => Set(ref timeout, value); }
    public string Advanced { get => advanced; set => Set(ref advanced, value); }
    public string[] Backends { get; } = ["cpu", "hybrid"];
    public ICommand PlanCommand { get; }
    public ICommand GenerateCommand { get; }
    public ICommand BrowseCommand { get; }
    private IReadOnlyList<string> Arguments(bool generation)
    {
        Positive(Context, "Context"); Positive(Generate, "Số token");
        List<string> arguments = ["--backend", Backend, "--context", Context, "--generate", Generate];
        if (generation)
        {
            if (string.IsNullOrWhiteSpace(ModelDirectory)) throw new ArgumentException("Chọn thư mục trọng số cục bộ.");
            Positive(Timeout, "Timeout");
            var value = UseTokens ? Tokens : Prompt;
            if (string.IsNullOrWhiteSpace(value)) throw new ArgumentException("Nhập nội dung hoặc danh sách token ID.");
            arguments.AddRange(["--model-directory", ModelDirectory, UseTokens ? "--tokens" : "--prompt", value, "--timeout", Timeout]);
        }
        arguments.AddRange(ArgumentTokenizer.Parse(Advanced));
        return arguments;
    }
    private static void Positive(string value, string name)
    {
        if (!int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out var number) || number < 1) throw new ArgumentException(name + " phải là số nguyên dương.");
    }
}
