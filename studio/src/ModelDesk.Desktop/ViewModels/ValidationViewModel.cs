using System.Windows.Input;
using ModelDesk.Core;
using ModelDesk.Desktop.Presentation;

namespace ModelDesk.Desktop.ViewModels;

public sealed class ValidationViewModel : ObservableObject
{
    private CoreOperation operation;
    private string backend = "cpu", evidenceMode = "Trực tuyến", evidencePath = "", maxShards = "512", budget = "64", tensor = "", directory = "", advanced = "", seed = "7", lengths = "8,32,64", seconds = "10";
    private bool online;
    public ValidationViewModel(CoreTaskViewModel task)
    {
        Operations = CoreOperations.All.Where(item => item.Id is not ("generate" or "runtime-plan")).ToArray(); operation = Operations.First(item => item.Id == "metadata-check");
        ExecuteCommand = new AsyncCommand(_ => task.RunAsync(Operation.Id, Arguments()), task.ShowError, _ => task.IsAvailable && !task.IsBusy);
        BrowseEvidenceCommand = new RelayCommand(_ => EvidencePath = DesktopActions.ChooseFolder("Chọn thư mục evidence", EvidencePath) ?? EvidencePath);
        BrowseModelCommand = new RelayCommand(_ => ModelDirectory = DesktopActions.ChooseFolder("Chọn thư mục model", ModelDirectory) ?? ModelDirectory);
    }
    public IReadOnlyList<CoreOperation> Operations { get; }
    public CoreOperation Operation
    {
        get => operation;
        set
        {
            if (value is null || !Set(ref operation, value)) return;
            if (value.Id != "probe" && Backend == "gpu") Backend = "cpu";
            Raise(nameof(IsMetadata)); Raise(nameof(IsProjection)); Raise(nameof(IsTokenizer)); Raise(nameof(IsBackend));
            Raise(nameof(IsSynthetic)); Raise(nameof(IsMonitor)); Raise(nameof(Description)); Raise(nameof(Backends));
        }
    }
    public string Description => Operation.Description;
    public bool IsMetadata => Operation.Id == "metadata-check";
    public bool IsProjection => Operation.Id == "projection-check";
    public bool IsTokenizer => Operation.Id == "tokenizer-check";
    public bool IsBackend => Operation.Id is "projection-check" or "probe" or "mini" or "parity" or "storage-check";
    public bool IsSynthetic => Operation.Id is "probe" or "mini" or "parity" or "storage-check";
    public bool IsMonitor => Operation.Id == "monitor";
    public string[] Backends => Operation.Id == "probe" ? ["cpu", "gpu", "hybrid"] : ["cpu", "hybrid"];
    public string[] EvidenceModes { get; } = ["Trực tuyến", "Offline", "Resume"];
    public string Backend { get => backend; set => Set(ref backend, value); }
    public string EvidenceMode { get => evidenceMode; set => Set(ref evidenceMode, value); }
    public string EvidencePath { get => evidencePath; set => Set(ref evidencePath, value); }
    public string MaxShards { get => maxShards; set => Set(ref maxShards, value); }
    public string Budget { get => budget; set => Set(ref budget, value); }
    public string Tensor { get => tensor; set => Set(ref tensor, value); }
    public string ModelDirectory { get => directory; set => Set(ref directory, value); }
    public string Advanced { get => advanced; set => Set(ref advanced, value); }
    public string Seed { get => seed; set => Set(ref seed, value); }
    public string Lengths { get => lengths; set => Set(ref lengths, value); }
    public string Seconds { get => seconds; set => Set(ref seconds, value); }
    public bool Online { get => online; set => Set(ref online, value); }
    public ICommand ExecuteCommand { get; }
    public ICommand BrowseEvidenceCommand { get; }
    public ICommand BrowseModelCommand { get; }
    private IReadOnlyList<string> Arguments()
    {
        List<string> result = [];
        if (IsMetadata)
        {
            result.AddRange(["--max-shards", MaxShards, "--budget-mib", Budget]);
            if (EvidenceMode != "Trực tuyến")
            {
                if (string.IsNullOrWhiteSpace(EvidencePath)) throw new ArgumentException("Chọn thư mục evidence để tiếp tục.");
                result.AddRange([EvidenceMode == "Offline" ? "--offline" : "--resume", EvidencePath]);
            }
        }
        if (IsBackend) result.AddRange(["--backend", Backend]);
        if (IsProjection)
        {
            result.AddRange(["--budget-mib", Budget]);
            if (!string.IsNullOrWhiteSpace(Tensor)) result.AddRange(["--tensor", Tensor]);
        }
        if (IsProjection || IsTokenizer)
        {
            if (Online) result.Add("--online");
            if (!string.IsNullOrWhiteSpace(ModelDirectory)) result.AddRange(["--model-directory", ModelDirectory]);
        }
        if (IsSynthetic) result.AddRange(["--seed", Seed]);
        if (Operation.Id is "mini" or "parity") result.AddRange(["--lengths", Lengths]);
        if (IsMonitor) result.AddRange(["--seconds", Seconds]);
        result.AddRange(ArgumentTokenizer.Parse(Advanced));
        return result;
    }
}
