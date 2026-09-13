# Kiến trúc ModelDesk

ModelDesk thêm một lớp C# cho GUI, CLI, trạng thái người dùng và truyền tệp. Python tiếp tục sở hữu việc diễn giải checkpoint, kiểm tra kiến trúc, lập kế hoạch, tokenizer và thực thi model. Adapter gọi các entrypoint hiện có; không chèn code, sửa module hay đổi thuật toán Python để phục vụ giao diện.

## Các project và chiều phụ thuộc

| Project | Trách nhiệm |
|---|---|
| `ModelDesk.Core` | Record, cấu hình có validation, danh mục `CoreOperations`, parser đối số và các interface service. Không phụ thuộc WPF hay HTTP cụ thể. |
| `ModelDesk.Infrastructure` | HTTP Hugging Face, downloader, rate/connection/disk coordination, JSON/DPAPI, process Python và đọc báo cáo. Phụ thuộc Core. |
| `ModelDesk.Desktop` | WPF, view model, command, theme, folder picker và điều phối hàng đợi. Phụ thuộc Core; composition root trong `App.xaml.cs` nối các adapter Infrastructure. |
| `ModelDesk.Cli` | Phân tích câu lệnh, stdout/stderr và Ctrl+C. `Program.cs` là composition root; `CliApplication` nhận các interface. |
| `ModelDesk.Tests` | Test contract, HTTP giả lập, download/resume, process, cấu hình/báo cáo và view model trên STA dispatcher. |
| `ModelDesk.DownloadBench` | Chương trình độc lập chạy downloader thật qua HTTP loopback để đo truyền dữ liệu và hủy/tiếp tục. |

Các port chính là `IHuggingFaceClient`, `IModelDownloader`, `IModelDownloadPlanner`, `IAdvancedDownloadOptions`, `ISettingsStore`, `ICredentialStore`, `IDownloadQueueStore`, `IPythonCoreService` và `IReportService`. Core dùng các kiểu bất biến cho request/result, `CancellationToken` cho vòng đời và `IProgress<T>` cho tiến độ. Service có thể được thay bằng fake trong test mà không cần dựng GUI hay mở mạng.

## WPF và vòng đời ứng dụng

Bảy trang có view riêng. `ShellViewModel` giữ profile, cấu hình đang áp dụng và các view model con; `CoreTaskViewModel` giữ một tác vụ Python; `DownloadsViewModel` giữ một hàng đợi độc lập. `ObservableObject`, `RelayCommand` và `AsyncCommand` triển khai thông báo thay đổi/can-execute, chống gọi lặp và đưa lỗi về trạng thái hiển thị.

View dùng control WPF thật cho keyboard/focus, DataGrid virtualization và vùng cuộn khi nội dung vượt chiều cao. Theme dùng resource brush để đổi Dark/Light/System. Chuyển opacity kéo dài 150 ms khi Windows cho phép animation. Cửa sổ giới hạn kích thước ban đầu theo work area, khai báo PerMonitorV2 và không yêu cầu quyền quản trị.

App nạp workspace trước khi tạo binding của cửa sổ. Nếu phần core chưa sẵn sàng, hàng đợi và token vẫn được khôi phục độc lập. Tác vụ core bị vô hiệu hóa khi chưa có profile. Chẩn đoán startup/binding được ghi có giới hạn vào state directory, không âm thầm biến lỗi khởi tạo thành một màn hình thành công.

Output Python được gom cho UI theo nhịp khoảng 150 ms và giới hạn buffer hiển thị; adapter vẫn ghi log riêng có giới hạn. Tiến độ tải được điều tiết để tránh cập nhật toàn bộ bảng trên từng buffer mạng. Lựa chọn hàng loạt tệp dùng bộ đếm tăng/giảm và một lần cập nhật tổng; không quét lại toàn bộ danh sách sau từng checkbox. Lượt refresh báo cáo cũ không được ghi đè một lượt refresh mới hơn.

Hàng đợi dùng semaphore để tuần tự hóa thay đổi, lưu trạng thái và cấp tác vụ. Số file đang chạy được giới hạn theo cấu hình 1–8; điều chỉnh giảm không hủy tác vụ đang chạy. Khi đóng cửa sổ, shell hủy và chờ cả Python task lẫn download task hoàn tất cleanup. Các mục đang chờ/đang tải khi khôi phục GUI được chuyển sang Paused, không tự mở kết nối.

## Cầu nối Python

`PythonCoreService` đọc profile dưới `config/models/`, giữ model ID/revision và vùng báo cáo, rồi tạo một bản config cho mỗi run. Các giá trị RAM, CPU, GPU và đường dẫn trọng số từ AppSettings được đưa vào config đó, thay vì ghi đè profile nguồn.

Một operation Python chạy với thư mục làm việc là project core, theo dạng:

```text
<python> -u -m glm_local --config <run-directory>/config.json <operation> <arguments...>
```

`ArgumentList` giữ nguyên ranh giới đối số, bao gồm prompt Unicode và đường dẫn có khoảng trắng. `ArgumentTokenizer` chỉ chia chuỗi nâng cao thành các đối số; không mở rộng môi trường, thực thi shell substitution hay đánh giá nội dung prompt. Lệnh CLI `core --profile KEY` hoặc `core --config FILE` chỉ áp dụng cho request đó. Custom config phải nằm trong vùng `config/` của project.

Build native, setup reference và reference tests dùng các wrapper cố định của Python project. Những operation này không chấp nhận đối số shell bổ sung. Adapter ghi `config.json`, `output.log`, `run.json` dưới `reports/modeldesk/runs/<id>/`, giữ mã thoát của core và chỉ liên kết báo cáo phù hợp với tác vụ/model/revision hiện tại. `WindowsProcessJob` quản lý vòng đời cây process; các kiểm tra quota nghiệp vụ của Python vẫn thuộc Python core.

Hội thoại dùng `ChatRunOptions` và `ChatMessage` được validate/snapshot trong Core. Service tạo `messages.json` riêng cho từng lượt, không nhận đường dẫn lịch sử do view tự tạo. `MODELDESK_EVENT` được parse thành sự kiện có kiểu, cập nhật theo vị trí UTF-16 và đối chiếu với báo cáo cuối cùng thuộc đúng run. `GenerationResult` phân biệt phần assistant, suy luận, kết thúc hợp lệ và nội dung chưa hoàn tất. `RunViewModel` chỉ lưu các cặp hoàn tất vào context; lỗi/hủy vẫn giữ phần hiển thị và câu hỏi để thử lại. CLI dùng chung port và DTO. Lịch sử ở cấp ứng dụng, còn mỗi lượt vẫn khởi động một worker Python riêng.

## Hub, truyền tệp và resume

`HubHttpClientFactory` tạo HttpClient dùng lại, tắt automatic redirects. `HubHttp` quản lý endpoint/redirect, giới hạn response, timeout, HTTP status và phạm vi gửi credential. `HuggingFaceClient` trả model summary, revision đã phân giải, manifest tệp và README dạng văn bản. README được xem như dữ liệu; không tải hoặc thực thi code của repository.

Trước khi enqueue, `IModelDownloadPlanner` kiểm tra cả tập đích: đường dẫn hợp lệ, identity không xung đột, tệp dở đã nhận diện và dung lượng theo ổ. Dung lượng được ước tính theo phần còn cần cấp phát, có reserve; tệp đủ kích thước vẫn cần kiểm tra hash trước khi được bỏ qua trong `DownloadAsync`. `DiskReservations` phối hợp cấp phát của các transfer đang chạy. Khóa độc quyền `<file>.download.lock` ngăn GUI/CLI cùng viết một tệp.

Downloader có hai đường dùng chung identity, rate limiter và kết quả:

- Tuần tự: buffer 64 KiB, có thể nối tiếp một prefix v1 đã nhận diện với ETag mạnh.
- Chia đoạn: mặc định thử với file mới từ 64 MiB, tối đa 4 kết nối/file, đoạn mặc định 16 MiB và buffer thuê 128 KiB/worker. `RandomAccess.WriteAsync` ghi đúng offset vào một tệp đã cấp trước; checkpoint v2 lưu tiến độ cho từng đoạn.

`DownloadConnections` là semaphore **8 slot trong phạm vi tiến trình**, dùng chung cho đường tuần tự/chia đoạn. `AggregateRateLimiter` giới hạn tổng byte của downloader; không nhân hạn mức theo số worker. Số kết nối/ngưỡng/độ dài đoạn được chốt khi file bắt đầu, còn tốc độ có thể cập nhật khi đang chạy. Nhiều tiến trình GUI/CLI riêng không chia sẻ semaphore/rate limiter này.

Range, ETag, byte count và kết thúc body phải khớp trước khi ghi nhận hoàn tất. Tiến độ v2 dựa trên checkpoint, không dựa vào `FileInfo.Length` của `.part` đã cấp trước. Dữ liệu được flush trước khi metadata tiến độ tương ứng được thay thế nguyên tử. Sau hủy hoặc lỗi, các byte chưa được checkpoint có thể cần tải lại; offset đã checkpoint không bị đếm thành lỗ đã tải.

Máy chủ không cung cấp điều kiện chia đoạn có thể dẫn về đường tuần tự cho phần dữ liệu do downloader quản lý. Retry là hữu hạn với backoff/jitter, tôn trọng `Retry-After` của 429/503; cooldown quá dài được trả thành lỗi có thể resume thay vì giữ worker vô thời hạn. Signed CDN URL chỉ là cache trong bộ nhớ và không được ghi vào resume state/log.

Tệp chỉ chuyển từ `.part` sang tên cuối sau khi toàn bộ nội dung qua SHA-256 LFS hoặc Git blob SHA-1 và đúng kích thước. Tệp khác đã tồn tại không bị ghi đè. Resume identity khác model/revision/path/hash bị từ chối; sidecar hỏng hay `.part` không xác định không được coi là dữ liệu có thể nối tiếp.

## Trạng thái và thông tin xác thực

`JsonSettingsStore` và `JsonDownloadQueueStore` đọc cấu trúc có giới hạn, validate trước khi dùng, đóng băng collection trước serialize và thay file qua tệp tạm. Cấu hình lỗi được giữ lại để người dùng sửa. App và CLI dùng `%LOCALAPPDATA%/ModelDesk`; sidecar transfer nằm cạnh chính tệp đang tải.

`WindowsCredentialStore` ưu tiên `HF_TOKEN`, sau đó giải mã file DPAPI theo current user. Token không nằm trong AppSettings, không có tham số CLI để nhận token và không được bind thành chuỗi hiển thị trong view model. GUI dùng PasswordBox. Các buffer byte giải mã được xóa sau khi dùng; lưu token không biến private/gated repository thành tài nguyên công khai.

`ReportService` chỉ đọc các loại tệp văn bản được chấp nhận trong cây reports đã xác định, có giới hạn kích thước và kiểm tra đường dẫn. Một báo cáo cũ, output fixture hoặc kết quả download không được dùng để suy ra full-model inference đã đạt.

## Mở rộng và kiểm chứng

Để thêm thao tác của core, khai báo `CoreOperation` trong `CoreOperations`, thêm mapping report/launcher nếu cần và thêm form hoặc dùng đối số nâng cao. Kiểm tra sự tương thích ở Python riêng; không dùng tên operation mới để lách validation hay thay một graph chưa hỗ trợ. Profile mới cần config được ghim và evidence phù hợp, không chỉ một tên repository tải được.

Để thêm adapter, triển khai port trong Core và nối tại composition root của GUI/CLI. Giữ HTTP, filesystem, credential và process ra khỏi view. Kiểm tra cancellation, lỗi và sự khôi phục trước khi thêm trạng thái thành công. Thay đổi định dạng `.part.json` cần validation version và test tương thích v1/v2; không tự suy luận tiến độ v2 từ kích thước file.

`build-studio.ps1 -Test` chạy bộ test C#; Python reference suite chạy riêng qua wrapper gốc. `ModelDesk.DownloadBench` bổ sung đường HTTP socket thật trên loopback, kiểm tra checksum và kế toán byte sau hủy/tiếp tục. Lượt mô phỏng 64 MiB với server giới hạn 8 MiB/s mỗi kết nối và header 100 ms cho tỷ lệ 3,58 lần giữa bốn/một kết nối. Chỉ số đó mô tả điều kiện mô phỏng đã ghi trong report; không phải benchmark internet hoặc suy luận model.

Publish mặc định đóng gói GUI/CLI tự chứa .NET runtime và chép nguyên trạng các file Python core được chọn. Python interpreter, virtualenv và model weights nằm ngoài gói. Việc phát hành hoặc đủ test giao diện không thay thế nghiệm thu full checkpoint, quota và output parity của Python runtime.
