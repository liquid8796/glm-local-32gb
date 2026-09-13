namespace ModelDesk.Core;

public sealed record ModelProfile(string Key, string DisplayName, string ModelId, string Revision,
    string ConfigPath, string ModelDirectory, string ReportDirectory);

public enum CoreOperationKind { Python, BuildNative, SetupReference, TestReference }
public sealed record CoreOperation(string Id, string Name, string Description, string Group,
    CoreOperationKind Kind = CoreOperationKind.Python);

public static class CoreOperations
{
    public static IReadOnlyList<CoreOperation> All { get; } =
    [
        new("doctor", "Kiểm tra sẵn sàng", "Kiểm tra phần cứng, dữ liệu và các điều kiện chạy model.", "overview"),
        new("policy-check", "Kiểm tra giới hạn Windows", "Xác minh giới hạn CPU và committed memory.", "validation"),
        new("monitor", "Quan sát GPU", "Quan sát GPU và khuyến nghị điều tiết.", "overview"),
        new("generate", "Sinh văn bản", "Chạy decoder Python với trọng số cục bộ.", "run"),
        new("runtime-plan", "Lập kế hoạch bộ nhớ", "Ước tính RAM, VRAM và cache theo context.", "run"),
        new("metadata-check", "Kiểm tra metadata", "Đọc config, index và header; hỗ trợ offline/resume.", "validation"),
        new("architecture-check", "Kiểm tra kiến trúc", "Đối chiếu catalogue với profile đã ghim.", "validation"),
        new("projection-check", "Đối chiếu projection", "Kiểm chứng một projection FP8/NVFP4 trên CPU/GPU.", "validation"),
        new("tokenizer-check", "Kiểm tra tokenizer", "Đối chiếu tokenizer với manifest checkpoint.", "validation"),
        new("probe", "Thử kernel FP8", "Phép thử ma trận giả lập có giới hạn.", "validation"),
        new("mini", "Decoder thu nhỏ", "Kiểm tra decoder bằng dữ liệu nhỏ.", "validation"),
        new("parity", "Đối chiếu Transformers", "So sánh fixture với graph tham chiếu chính thức.", "validation"),
        new("storage-check", "Kiểm tra safetensors", "Kiểm chứng reader và scale theo khối.", "validation"),
        new("build-native", "Build native kernels", "Build DLL CPU qua Visual Studio C++ tools.", "tools", CoreOperationKind.BuildNative),
        new("setup-reference", "Tạo môi trường Python", "Cài môi trường tham chiếu theo lock của core.", "tools", CoreOperationKind.SetupReference),
        new("test-reference", "Chạy bộ hồi quy Python", "Chạy kiểm thử đầy đủ của core hiện tại.", "tools", CoreOperationKind.TestReference)
    ];
    public static CoreOperation Get(string id) => All.FirstOrDefault(item => item.Id == id)
        ?? throw new ArgumentException($"Unknown core operation: {id}");
}

public sealed record CoreRunRequest(AppSettings Settings, string Operation,
    IReadOnlyList<string> Arguments, string? ConfigPath = null, ChatRunOptions? Chat = null);
public sealed record CoreOutput(DateTimeOffset Timestamp, string Text, bool IsError = false, ModelStreamEvent? StreamEvent = null);
public sealed record CoreRunResult(string RunId, int ExitCode, bool Cancelled, string LogPath,
    string? ReportPath, DateTimeOffset StartedAt, DateTimeOffset FinishedAt, bool TimedOut = false,
    GenerationResult? Generation = null)
{
    public string Status => Cancelled ? "Cancelled" : TimedOut ? "Timed out" : ExitCode == 0 ? "Completed" : ExitCode == 2 ? "Needs review" : "Failed";
}

public interface IPythonCoreService
{
    IReadOnlyList<ModelProfile> GetProfiles(string projectRoot);
    Task<CoreRunResult> RunAsync(CoreRunRequest request, IProgress<CoreOutput>? output = null,
        CancellationToken cancellationToken = default);
}

public sealed record ReportEntry(string Name, string Path, DateTimeOffset ModifiedAt, long Bytes, string? Status);
public interface IReportService
{
    IReadOnlyList<ReportEntry> List(string projectRoot, string? profileReportDirectory = null);
    Task<string> ReadAsync(string path, CancellationToken cancellationToken = default);
}
