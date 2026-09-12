# ModelDesk

ModelDesk là ứng dụng WPF và CLI cho việc tải model từ Hugging Face, quản lý checkpoint cục bộ và sử dụng Python core GLM hiện có. GUI và CLI dùng chung các service C#; phần Python được giữ nguyên. Trình tải có thể tải repository ngoài hai profile GLM, nhưng việc tải đủ tệp không xác nhận model đó có thể suy luận bằng Python core này.

## Mở ứng dụng

Trong bộ phát hành, chạy `ModelDesk.exe` để mở GUI hoặc `modeldesk-cli.exe help` để xem CLI. Giữ nguyên các tệp đi kèm và thư mục `core/`; không chỉ sao chép riêng một tệp EXE.

Trong repository, mở `ModelDesk.sln` bằng Visual Studio 2026 và chọn `ModelDesk.Desktop` làm startup project. Hoặc chạy các lệnh sau tại thư mục chứa solution:

```powershell
.\build-studio.bat -Test
.\modeldesk.bat
.\modeldesk-cli.bat help
```

`clean-project.bat` xem trước các cache có thể dọn; thêm `-Apply` để xóa file sinh tự động. Công cụ giữ nguyên source Python/native, DLL đang dùng, môi trường Python, trọng số, metadata evidence và bản publish. Output folder có file đang được ứng dụng mở sẽ được bỏ qua. Không cần submodule Kimi để build hoặc chạy.

Các wrapper ưu tiên bản đã publish ở `artifacts/ModelDesk/win-x64/` nếu có. Khi cần chạy đúng source vừa sửa, dùng trực tiếp:

```powershell
dotnet run --project studio/src/ModelDesk.Desktop -c Release
dotnet run --project studio/src/ModelDesk.Cli -c Release -- profiles
```

## Windows, .NET và Python

Ứng dụng nhắm Windows 10/11 x64 với .NET 10 và nhận biết DPI theo từng màn hình. Việc kiểm chứng thực tế của dự án được thực hiện trên Windows 11 x64; Windows 10 chưa có lượt nghiệm thu tương đương. Phạm vi hỗ trợ .NET 10 của Microsoft trên Windows 10 hiện giới hạn ở các phiên bản LTSC/Enterprise còn được hỗ trợ, không phải mọi bản Home/Pro. Visual Studio 2026 từ 18.0 hỗ trợ .NET 10. Xem [hướng dẫn Windows chính thức của Microsoft](https://learn.microsoft.com/en-us/dotnet/core/install/windows#supported-versions).

Build source cần .NET 10 SDK; `global.json` chọn SDK ổn định từ `10.0.100`, cho phép roll forward trong các feature band .NET 10. Trong Visual Studio, cài workload **.NET desktop development**. Bản publish mặc định tự chứa runtime .NET; bản `-FrameworkDependent` cần **.NET Desktop Runtime 10 x64** trên máy chạy.

Hugging Face và tải tệp không cần Python. Các lệnh của core cần Python x64 3.10 trở lên và một thư mục có `glm_local/__main__.py`. Môi trường tham chiếu, native CPU kernels và GPU có yêu cầu riêng:

- `setup-reference` tạo `.venv-reference` và cài các dependency được ghim của Python core. Tác vụ này cần truy cập mạng để cài thư viện, không tải trọng số model.
- `build-native` cần Visual Studio C++ x64 build tools. Ba DLL CPU là `fp8_cpu.dll`, `topk_cpu.dll` và `nvfp4_cpu.dll`.
- Các phép thử hybrid cần GPU NVIDIA và driver tương thích. Các tệp PTX đi kèm được driver nạp trực tiếp; không cần CUDA Toolkit cho đường PTX hiện tại.
- Chạy inference cần trọng số và tokenizer cục bộ phù hợp đúng model/revision. Xem tài liệu của Python core trong `docs/` của repository hoặc `core/docs/` của bộ phát hành.

## Cấu hình lần đầu

Mở **Cài đặt**, chọn thư mục Python core, Python executable và thư mục tải mặc định, rồi bấm **Lưu cấu hình**. **Tự tìm đường dẫn** tìm core đi kèm hoặc repository và giữ lại đường dẫn Python tùy chỉnh. **Lưu và nạp lại profile** quét lại các profile trong `config/models/`.

Khi chưa có cấu hình đã lưu, việc tìm core ưu tiên biến môi trường `MODEL_DESK_ROOT`, sau đó kiểm tra các thư mục cha của vị trí ứng dụng và thư mục làm việc, kể cả thư mục con `core/`. Python mặc định là `.venv-reference/Scripts/python.exe` trong core nếu có, nếu không dùng `python` trên PATH. `MODEL_DESK_ROOT` là đầu vào cho việc khám phá mặc định; một cấu hình đã lưu vẫn dùng `ProjectRoot` của nó.

Hai profile đi kèm là `nvfp4` và `fp8`. Profile quyết định model ID, revision, config, trọng số và vùng báo cáo. Đường dẫn trọng số riêng trong Cài đặt có thể thay đường dẫn mặc định cho các tác vụ mới. Cấu hình tài nguyên hiện giữ trần của Python core: tối đa **32 GB RAM thập phân**, **70% CPU**, mục tiêu **60% GPU**. Đây là giá trị cấu hình; đọc báo cáo kiểm chứng để biết giới hạn nào đã được xác minh trên máy.

## Bảy trang làm việc

| Trang | Công việc |
|---|---|
| Tổng quan | Xem checkpoint/revision đang chọn, đường dẫn Python, giới hạn cấu hình và báo cáo sẵn sàng gần nhất; chạy doctor hoặc quan sát GPU. |
| Chạy model | Nhập thư mục trọng số, prompt hoặc token ID, backend, context, số token sinh và timeout; lập kế hoạch bộ nhớ trước khi sinh văn bản. |
| Kiểm chứng | Chạy toàn bộ operation của core: metadata, kiến trúc, projection, tokenizer, kernel, mini, parity, storage, Windows policy, build native, setup và bộ kiểm thử tham chiếu. |
| Hugging Face | Tìm model, mở trực tiếp `owner/model`, chọn revision, đọc README dạng văn bản, lọc/chọn tệp và chọn thư mục tải. |
| Tải xuống | Theo dõi số byte, tốc độ, thời gian còn lại và kết quả; tạm dừng, tiếp tục hoặc hủy từng tệp hay cả hàng đợi. |
| Báo cáo | Lọc, đọc JSON/Markdown và mở thư mục bằng chứng của profile đang chọn. |
| Cài đặt | Cấu hình core, Python, thư mục, tài nguyên, giao diện Dark/Light/System, tốc độ tải, số tệp và số kết nối. |

Nhật ký tác vụ Python nằm trong thanh có thể mở rộng ở cuối cửa sổ. Một tác vụ core chạy tại một thời điểm; hàng đợi tải vẫn hoạt động độc lập. Có thể dừng tác vụ đang chạy, và ứng dụng chờ các tác vụ được hủy kết thúc khi đóng cửa sổ.

Trang Kiểm chứng hỗ trợ metadata **trực tuyến**, **offline** hoặc **resume**, với thư mục evidence, số shard và ngân sách đọc riêng. Projection trực tuyến và tokenizer trực tuyến là các lựa chọn rõ ràng trong form. Ô đối số nâng cao hỗ trợ dấu nháy cho giá trị có khoảng trắng; các đối số Python được truyền bằng `ProcessStartInfo.ArgumentList`. Các tác vụ build/setup/test dùng wrapper cố định và không nhận đối số shell tùy ý.

## Tìm và tải model

1. Tìm theo tên, chọn kết quả rồi bấm **Xem model đã chọn**, hoặc nhập `owner/model` và revision rồi bấm **Mở**. Branch như `main` được phân giải thành một commit cụ thể trước khi tạo yêu cầu tải.
2. Xem tổng kích thước, danh sách tệp và hash. Chọn từng tệp, chọn tất cả, hoặc lọc tên rồi **Chọn tệp đang lọc**.
3. Kiểm tra thư mục đích và tổng số byte đã chọn. **Tải các tệp đã chọn** chỉ tải lựa chọn đó; **Tải toàn bộ model** là yêu cầu tải mọi tệp trong revision đang mở.
4. Theo dõi trong **Tải xuống**. Ứng dụng kiểm tra dung lượng cho cả lựa chọn trước khi bắt đầu; tệp đích có nội dung khác và tệp dở không xác định được nguồn sẽ không bị ghi đè.

Mặc định tải **2 tệp đồng thời**, tối đa **4 kết nối cho mỗi tệp lớn**, và không giới hạn tốc độ. Tệp mới từ **64 MiB** được thử tải theo các đoạn **16 MiB** khi máy chủ cung cấp Range/ETag phù hợp. Bộ điều phối giới hạn **8 kết nối tải trong mỗi tiến trình**; GUI và một CLI chạy riêng là hai tiến trình riêng. Tệp nhỏ, cấu hình một kết nối hoặc máy chủ không phù hợp dùng đường tải tuần tự.

Giới hạn MiB/s áp dụng cho tổng lưu lượng của downloader, không phải cho từng kết nối. Thay đổi tốc độ trong GUI được áp dụng khi lưu; số tệp đồng thời ảnh hưởng việc nhận tác vụ mới, còn tùy chọn kết nối được chốt khi một tệp bắt đầu. Giảm số tệp không tự hủy các tệp đã chạy. Không có thiết lập nào bảo đảm tốc độ internet: đường truyền, CDN, ổ đĩa và bước kiểm tra hash đều ảnh hưởng thời gian hoàn tất.

### Tiếp tục tệp dở

Trình tải giữ `tên-tệp.part` và `tên-tệp.part.json` trong thư mục đích. Giữ cả hai để tiếp tục đúng model, revision, đường dẫn tệp, kích thước và hash:

- Resume **v1** dùng phần đầu đã tải tuần tự và ETag mạnh. Khi không thể bảo đảm danh tính Range, downloader tải lại phần dữ liệu do chính nó quản lý.
- Resume **v2** lưu phạm vi và số byte đã hoàn tất của từng đoạn. Một tệp `.part` v2 có thể đã được cấp trước toàn bộ kích thước; kích thước trên đĩa không phải là số byte đã tải. Tiến độ lấy từ các checkpoint của đoạn.

Các checkpoint được ghi sau khi dữ liệu tương ứng được flush. Khi hoàn tất, toàn bộ tệp được đối chiếu **SHA-256 cho LFS** hoặc **Git blob SHA-1** cho tệp thường rồi mới đổi thành tên đích. Tệp đích có sẵn cũng phải qua kiểm tra hash trước khi được bỏ qua. Hash lỗi không được báo thành hoàn tất.

Đóng ứng dụng, tạm dừng hoặc hủy giữ lại dữ liệu dở được nhận diện. Khi mở lại GUI, tác vụ đang chờ/đang tải trước đó trở thành **Tạm dừng**, không tự tải tiếp. CLI tiếp tục bằng cách chạy lại cùng yêu cầu với đúng revision và thư mục. CLI không thêm tác vụ vào hàng đợi GUI; hai bên dùng chung định dạng tệp dở và có khóa độc quyền cho mỗi tệp đích.

## CLI

Các ví dụ sau chạy trong thư mục bộ phát hành. Trong repository, thay `.\modeldesk-cli.exe` bằng `.\modeldesk-cli.bat`.

```powershell
.\modeldesk-cli.exe hub search GLM-5.3
.\modeldesk-cli.exe --json hub details dealignai/GLM-5.3-ABLITERATED-NVFP4
.\modeldesk-cli.exe profiles
.\modeldesk-cli.exe settings show
```

Tải riêng các tệp cấu hình nhỏ vào thư mục bạn chọn:

```powershell
.\modeldesk-cli.exe download dealignai/GLM-5.3-ABLITERATED-NVFP4 `
  --revision 371bdb985d0124e76348c91e4a8fcf3a9d719d09 `
  --folder "D:\Models\GLM-5.3-ABLITERATED-NVFP4" `
  --file config.json --file tokenizer_config.json --parallel 2 --limit-mib 20
```

Để chủ động tải toàn bộ repository, dùng `--all` thay cho các `--file`; hai lựa chọn này loại trừ nhau. `--connections 1..4` hoặc `--connections-per-file 1..4` điều chỉnh kết nối mỗi tệp, `--parallel 1..8` điều chỉnh số tệp và `--limit-mib 0` bỏ giới hạn tốc độ.

```powershell
.\modeldesk-cli.exe settings set root "D:\Project\llm\glm-local-32gb"
.\modeldesk-cli.exe settings set python "D:\Python\python.exe"
.\modeldesk-cli.exe settings set download-folder "D:\Models"
.\modeldesk-cli.exe settings set parallel 2
.\modeldesk-cli.exe settings set connections 4
.\modeldesk-cli.exe settings set limit-mib 0

.\modeldesk-cli.exe core --profile nvfp4 doctor
.\modeldesk-cli.exe core --profile fp8 runtime-plan --backend cpu --context 4096
.\modeldesk-cli.exe core --profile nvfp4 metadata-check --resume "D:\Evidence\nvfp4"
.\modeldesk-cli.exe core --config "config\models\abliterated-nvfp4.json" architecture-check
.\modeldesk-cli.exe reports list
.\modeldesk-cli.exe reports read "reports\nvfp4\metadata-latest.json"
```

`core --profile` hoặc `core --config` áp dụng cho một lần chạy, đặt **trước tên operation**, và không đổi lựa chọn đã lưu. Config tùy chỉnh phải nằm dưới `config/` của Python project. Sau operation, mọi đối số thuộc Python, kể cả một giá trị có nội dung `--json`; đặt `--json` ở đầu lệnh ModelDesk khi cần output có cấu trúc. Tiến độ khi đó đi ra stderr. Ctrl+C hủy tác vụ và trả mã 130; mã thoát khác của core được giữ nguyên, bao gồm mã 2 cho trường hợp blocked/review.

Các khóa `settings set` còn có `profile`, `runtime-folder`, `ram-mib`, `cpu-percent`, `gpu-index`, `gpu-percent`, `gpu-window` và `theme`. Đơn vị RAM của CLI là MiB; giới hạn 32 GB của core vẫn được áp dụng. `theme` nhận `Dark`, `Light` hoặc `System`.

## Trạng thái và token

GUI và CLI lưu cấu hình theo tài khoản Windows tại `%LOCALAPPDATA%\ModelDesk`:

| Tệp | Nội dung |
|---|---|
| `settings.json` | Đường dẫn, profile, giao diện và các giới hạn; không chứa token. |
| `downloads.json` | Hàng đợi GUI, danh tính tệp và trạng thái phục hồi. |
| `huggingface.token.dpapi` | Token được mã hóa bằng Windows DPAPI cho tài khoản hiện tại. |
| `startup.log` | Chẩn đoán khởi động/binding có giới hạn, lọc token Hugging Face; file cũ được chuyển sang `.previous`. |

Nhập token bằng ô mật khẩu trong **Cài đặt**, hoặc cung cấp `HF_TOKEN` qua môi trường chạy đã được quản lý. `HF_TOKEN` có ưu tiên cao hơn token DPAPI đã lưu. Vì vậy, xóa token trong GUI không vô hiệu hóa một `HF_TOKEN` vẫn tồn tại. Không đưa token vào đối số CLI; CLI không có khóa cài token bằng giá trị trên command line. Một token hợp lệ vẫn cần quyền truy cập repository gated/private tương ứng.

Log và kết quả tác vụ core nằm dưới `reports/modeldesk/runs/<run-id>/` của Python project. Mỗi run giữ config phát sinh, `output.log` và `run.json`; báo cáo nghiệp vụ của Python tiếp tục dùng namespace profile. Đọc trạng thái và nội dung báo cáo, không suy diễn một lần kiểm tra nhỏ thành nghiệm thu full model.

## Build, kiểm thử và publish

Chạy tại thư mục chứa `ModelDesk.sln`:

```powershell
.\build-studio.bat -Configuration Release -Test
.\publish-studio.bat
```

`build-studio.ps1 -Test` build solution và chạy test C#; TRX được ghi vào `reports/studio/tests/`. Bộ test Python tham chiếu là một bước riêng, có thể chạy từ trang Kiểm chứng hoặc `core test-reference` sau khi chuẩn bị môi trường/native kernels.

`publish-studio.ps1` mặc định kiểm thử trước khi publish hai ứng dụng x64 tự chứa runtime vào `artifacts/ModelDesk/win-x64/`. `-FrameworkDependent` tạo bản cần runtime cài sẵn; `-SkipTests` chỉ dành cho lần đóng gói đã có kết quả kiểm thử phù hợp.

Gói chứa GUI, CLI, runtime C# nếu self-contained và bản sao nguyên trạng của Python core, native source/PTX, config, tài liệu và test. Các DLL native CPU có trong `build/` được chép vào `core/build/` nếu tồn tại. Gói **không chứa Python interpreter, virtualenv hay trọng số model**. Trên máy mới, chuẩn bị Python/môi trường và chọn lại các đường dẫn trong Cài đặt. Không xem `publish-receipt.json` là bằng chứng suy luận full checkpoint.

## Đo tốc độ có thể lặp lại

Benchmark tại `studio/benchmarks/ModelDesk.DownloadBench` dùng downloader thật với HTTP loopback trên `127.0.0.1`. Lượt 64 MiB, độ trễ header 100 ms và giới hạn máy chủ 8 MiB/s **cho mỗi kết nối** đo được:

| Chế độ | Thời gian gồm kiểm tra hash | Tốc độ |
|---|---:|---:|
| Một kết nối | 8,373 giây | 7,64 MiB/s |
| Bốn kết nối | 2,338 giây | 27,37 MiB/s |

Chênh lệch **3,58 lần** thuộc mô phỏng có kiểm soát này, không phải phép đo tốc độ Hugging Face hay cam kết cho mọi đường truyền. Lượt hủy/tiếp tục cũng xác minh tổng byte đã lưu cộng byte cần yêu cầu tiếp bằng kích thước tệp, rồi kiểm tra hash hoàn chỉnh. Báo cáo cục bộ là `reports/studio/download-benchmark-64mib.json`; README của benchmark có lệnh tái chạy. Các kiểm tra ModelDesk dùng dữ liệu nhỏ và mô phỏng; không tải full checkpoint trong quá trình kiểm chứng ứng dụng.

## Khi cần xử lý lỗi

- Không thấy profile: kiểm tra thư mục có `glm_local/__main__.py`, bấm tự tìm hoặc chọn lại, rồi lưu/nạp lại profile. Hub và hàng đợi vẫn dùng được khi core cần cấu hình lại.
- Không tìm thấy Python: chọn executable thật, tạo môi trường tham chiếu nếu cần và lưu lại. Python tùy chỉnh không bị tự đổi khi chỉ dò core.
- Không đủ đĩa: chọn thư mục ở ổ khác hoặc giảm số tệp; preflight kiểm tra toàn bộ lựa chọn trước khi tải payload.
- Xung đột tệp đích/partial: chọn thư mục khác hoặc kiểm tra đúng revision. Giữ nguyên tệp dở và sidecar nếu muốn tiếp tục; không tự sửa các byte tiến độ.
- HTTP 401/403: kiểm tra token và quyền model. Giới hạn 429/503 có retry hữu hạn, tôn trọng `Retry-After`; một yêu cầu chờ quá lâu được để lại cho lần tiếp tục sau.
- Core trả mã 2 hoặc 3: mở báo cáo để xem điều kiện chưa đạt. Trạng thái này được truyền lại, không đổi thành thông báo thành công.

Tài liệu dành cho người mở rộng ứng dụng nằm ở `studio/ARCHITECTURE.md` trong source repository.
