# Checkpoint metadata audit — 0.7.1

## Mục tiêu và mốc hiện tại

Bước này **thu thập và kiểm tra config/index/header của checkpoint đã ghim**, không chạy model. Baseline 0.6.1 đã có bằng chứng Windows official hybrid với fixture synthetic; bằng chứng đó không xác minh tensor của checkpoint thật.

Lệnh mới giữ `dealignai/GLM-5.3-CYBERSECURITY-FP8`, revision `5915c1b88f998a9c1e1a0c83688e285a08ae3ca5`. Không chuyển sang `main`, model khác, GGUF, tải tokenizer, thực thi remote code hoặc gọi `from_pretrained`. Cấu hình và revision lock không bị sửa.

**Lần chạy tại môi trường phát triển:** DNS không phân giải được Hugging Face; dừng ở `fetch_model`, 0 byte metadata và 0 Range request. Vì thế bản phát hành này **không kèm kết luận PASS cho metadata checkpoint thật**. Xem [report truy cập](verification/metadata-online-attempt-linux-v0.7.1.json).

## Chạy trên máy người dùng

Chép source mới, giữ `.venv-reference` và `build`. Không cài thêm gói và không build lại DLL: lệnh metadata chỉ dùng Python standard library. `test-reference.bat` vẫn dùng môi trường đã ghim để chạy bộ hồi quy cũ/mới.

```powershell
.\test-reference.bat 2>&1 | Tee-Object -FilePath .\reports\test-reference-v0.7.1.log
.\glm.bat metadata-check
```

Dừng để đọc log khi test lỗi. Double-click `metadata-check.bat` chạy cùng lệnh mặc định và giữ cửa sổ mở. Script `glm.bat` giữ mã thoát để dùng trong terminal/automation.

Có thể chủ động kiểm tra ít shard trước:

```powershell
.\glm.bat metadata-check --max-shards 3
```

Nếu checkpoint có nhiều hơn ba shard, kết quả **PARTIAL / exit 2**, không phải đã kiểm chứng toàn bộ. Việc chọn shard theo tên tăng dần, không phải mẫu đại diện kiến trúc; scale ở shard chưa đọc được ghi deferred. Không suy ra đúng/sai của header chưa đọc.

Trên Windows, launcher tạo worker qua cơ chế Job Object sẵn có, cài CPU hard cap và committed-memory limit trước khi worker chạy. Nó dùng Python đang chạy `glm.bat`, không bắt buộc venv. Trên Linux, audit có thể chạy để phát triển nhưng `job_policy_verified=false`; không giả lập nghiệm thu Windows. Không có công việc CUDA trong audit này.

## Các bước thực hiện

1. Đọc model API tại revision cố định; yêu cầu `id` và `sha` đúng settings, shard có tên/kích thước hợp lệ.
2. Đọc `config.json` và `model.safetensors.index.json` tại cùng revision. Đối chiếu các field/manifest đã ghi trong `docs/model-metadata.json`, không tự cập nhật baseline cho khớp dữ liệu mới.
3. Mỗi shard: yêu cầu `Range: bytes=0-7`, lấy độ dài header; sau đó chỉ yêu cầu `Range: bytes=8-(7+header_length)`. Không yêu cầu vùng payload.
4. Dùng cùng `parse_header_bytes` với reader local để xác minh dtype/shape/offset, vùng dữ liệu liên tục, không overlap/gap/trailing bytes và tổng kích thước. So sánh chính xác tập tên tensor của header với mapping index; kiểm tra `metadata.total_size` khi đủ header.
5. Ghi catalogue quan sát thực tế; đối chiếu các tên GLM đã nhận diện với kích thước khai báo trong config. Tên hoặc chiều config chưa biết ghi `not_reviewed`, không lấy default từ mô hình thu nhỏ.
6. Đối với tên `*.weight` dtype `F8_E4M3`, kiểm tra profile ứng viên `*.weight_scale_inv`, lưới `ceil(rows/128) × ceil(cols/128)` và F32 scale mà adapter hiện tại yêu cầu. Ghi riêng scale chéo shard, missing/deferred, kiểu scale chưa hỗ trợ và ma trận không phải 2D. Không đọc giá trị scale hoặc suy luận chúng dương/hữu hạn.

Các công thức kiểm tra tên đã biết tham chiếu cấu trúc trong mapping official miniature hiện có (embedding, norm, MLA, dense/expert/shared MLP, router và indexer), dùng **giá trị config được đọc**, không dùng kích thước miniature. Đây không phải bảng mapping hoàn chỉnh của checkpoint thực: không chứng minh đủ mọi layer/expert, schedule/shared index, MTP, RoPE hoặc tokenizer. `architecture_mapping_verified` luôn false ở bước này.

## HTTP và giới hạn cục bộ

| Giới hạn | Giá trị |
|---|---:|
| Model API JSON | 8 MiB |
| Config JSON | 1 MiB |
| Index JSON cho audit | 32 MiB |
| Tổng body metadata được application đọc | 64 MiB mặc định; `--budget-mib` cho phép 1–128 MiB |
| Header một shard | 1 MiB, giữ nguyên policy reader |
| Mỗi lần đọc body/file | Tối đa 64 KiB |
| Tensor mỗi header | 4.096, giữ nguyên policy reader |
| Tensor trong index audit | 262.144 |
| Shard | 512 |
| Request, tính cả redirect | Tối đa 4.096 |
| Redirect mỗi request | Tối đa 5 |
| Socket timeout | 30 giây; tổng elapsed budget 1.800 giây |

Timeout tổng được kiểm tra ở ranh giới mở kết nối/đọc; một read đang chờ socket có thể kết thúc sau ranh giới đó theo socket timeout. Windows launcher có timeout worker riêng và cleanup cây tiến trình như cơ chế cũ.

Chỉ HTTPS tới Hugging Face và các host con của `huggingface.co`/`hf.co`; không đọc token môi trường hoặc chuyển credential. Redirect/error body được đóng mà không consume. Không ghi signed redirect URL vào report. `X-Repo-Commit` nếu server cung cấp phải khớp. ETag nếu có ở prefix được đối chiếu ở header; ETag mạnh được gửi thêm qua `If-Match`.

**Shard bắt buộc HTTP 206 và Content-Range đúng chính xác vùng yêu cầu + kích thước manifest.** HTTP 200, thiếu/sai Range, sai length, content encoding nén, header quá lớn hoặc budget cạn đều dừng. Không fallback sang tải cả file và không tự retry. JSON không có Content-Length được đọc tới tối đa limit+1 byte để phát hiện quá giới hạn; các byte đã đọc vẫn tính vào tổng budget. Request tới checkpoint private/gated chưa được hỗ trợ bởi cơ chế đăng nhập ở bước này.

Counter giới hạn **body mà application đọc**, không phải toàn bộ byte trên đường truyền, buffer socket/TLS, object Python sau parse hoặc tổng RAM vật lý. `observed_tensor_payload_bytes` là tổng byte **được mô tả bởi header**, không phải số byte trọng số đã tải. Xem `io.tensor_payload_bytes_requested=0` để phân biệt.

Index audit có policy riêng để kiểm tra cấu trúc lớn. Reader đang chạy payload vẫn là **index 1 MiB / 8.192 tensor**; checker không tự nới hai giới hạn đó. `runtime_index_policy.fits_current_reader_index_policy` cho biết dữ liệu quan sát có vượt reader hiện tại không. Metadata PASS không chứng minh reader/decoder đã sẵn sàng nhận full checkpoint.

## Kết quả và cách đọc

| Status / mã thoát | Ý nghĩa |
|---|---|
| `PASS` / 0 | Đủ header, metadata structure khớp, các kiểm tra baseline/profile đã thực hiện không phát hiện vấn đề. Chỉ PASS metadata. |
| `PARTIAL` / 2 | Chỉ có một phần header; không bật cờ xác minh toàn bộ. Cần xem cả findings đã phát hiện ở phần đọc được. |
| `REVIEW_REQUIRED` / 3 | Đã đọc đủ header nhưng config/profile/known-shape có điểm không khớp hoặc chưa hỗ trợ. Catalogue vẫn được giữ. |
| `ERROR` / 1 | Lỗi mạng, revision, JSON/header/index, budget, file snapshot hoặc worker. Có stage, active shard nếu có và traceback. |
| `INTERRUPTED` / 130 | Người dùng dừng; không được coi là nghiệm thu thành công. |

`metadata_structure_verified` chỉ bật sau khi đủ index/header hợp lệ và `total_size` khớp. Cờ này có thể true trong `REVIEW_REQUIRED`: container file nhất quán nhưng profile xử lý cần sửa. `tensor_review.fp8_adapter_metadata_verified` chỉ xét dtype/rank/grid/tên scale theo profile F32 hiện tại, không xác minh giá trị, thứ tự phép tính hay numerical parity.

Các cờ sau luôn false: `real_checkpoint_compatible`, `architecture_mapping_verified`, `full_model_loaded`, `inference_verified`, `full_model_limits_verified`, `payload_values_verified`. Không gọi hoặc thay đổi `doctor`; mốc full model của nó vẫn BLOCKED.

## File bàn giao sau mỗi lần chạy

```text
reports/metadata/<run-id>/
  request.json
  result.json
  result.md
  tensor-catalogue.jsonl       # Có khi đã tới bước review metadata
  evidence/
    snapshot.json
    model.json
    config.json
    model.safetensors.index.json
    headers/
      <shard-name>.header     # Chỉ prefix 8 byte + JSON header
```

`metadata-latest.json`/Markdown là bản gần nhất, không thay thế toàn bộ thư mục run. Header đã nhận có thể được giữ ngay cả khi validation tiếp theo lỗi, để chẩn đoán chính xác; snapshot không phải chứng nhận dữ liệu hợp lệ. File bị lỗi trước khi nhận xong không được lưu giả như đã hoàn thành.

Snapshot ghi SHA-256/số byte của từng JSON và prefix/header. Replay từ thư mục evidence:

```powershell
$run = (Get-Content .\reports\metadata-latest.json -Raw | ConvertFrom-Json).run_directory
$evidence = Join-Path $run "evidence"
.\glm.bat metadata-check --offline "$evidence"
```

Replay không khởi tạo HTTP source, xác minh lại hash/schema/index/header và ghi một run mới. Snapshot thiếu header vẫn PARTIAL. Replay không tải tiếp phần thiếu và không phải tính năng resume; chạy online mới hiện bắt đầu audit mới. Snapshot có thể bị người có quyền sửa rồi tính lại hash, nên offline chỉ chứng minh tính nhất quán cục bộ, không tái xác thực server/revision hoặc hash payload checkpoint. Từ chối artifact symlink/reparse point, tên thoát thư mục, file quá lớn và prefix/header kèm payload.

Khi gửi report, gửi ZIP cả thư mục `reports/metadata/<run-id>/` cùng log test. Đây là metadata/cấu trúc tensor, không phải trọng số model.

## Kiểm chứng bản vá

87 test mới kiểm tra transport, schema, snapshot, workflow, CLI và hợp đồng launcher Windows bằng test double. Toàn suite tại Linux: **453 phát hiện, 427 đạt, 26 bỏ qua**; [log](verification/unit-tests-linux-v0.7.1.txt). Môi trường hiện tại NumPy 2.3.5, safetensors 0.7.1, Torch 2.10.0+cpu, không có Transformers; không thay lock vì khác biệt đó. Cũng chạy riêng 87 test metadata với `python -S` (không nạp site-packages): **87 đạt, không bỏ qua**; [log standard library](verification/metadata-stdlib-tests-linux-v0.7.1.txt). Test mới metadata không cần các package này.

Parser header mới dùng lại nguyên các kiểm tra cũ; toàn bộ test reader/block/sharded/decoder hiện hữu trong suite được chạy theo khả năng môi trường. Không chạy lại live Windows Job Object, CUDA, MSVC DLL hoặc Transformers đúng revision ở đây. Test launcher dùng policy giả lập chỉ chứng minh cách gọi; không tính là policy Windows mới đã được nghiệm thu.

[Invariant file](verification/metadata-release-invariants-v0.7.1.json) ghi hash các config, lock, kernel, graph, reader index và FP8 adapter không đổi. Project Python/C không có .NET AssemblyVersion; version được tăng đồng bộ ở `pyproject.toml` và `glm_local.__version__` thành 0.7.1. Native ABI không đổi.

## Bước sau khi có metadata thật

Đọc `tensor-catalogue.jsonl`, review các điểm chưa hỗ trợ và chọn cặp projection/scale có bằng chứng thật. Sau đó mới sửa index policy có giới hạn, hỗ trợ dtype/layout cần thiết, rồi mở rộng projection nhiều block và numerical parity. Không đưa một schema giả lập thành checkpoint mapping chính thức để bỏ qua bước này.

Commit:

```text
feat(metadata): add bounded checkpoint header audit and offline replay
```

Nguồn định dạng/giao thức: [Hugging Face metadata parsing](https://huggingface.co/docs/safetensors/metadata_parsing), [safetensors format](https://github.com/safetensors/safetensors#format), [RFC 9110 HTTP Range](https://www.rfc-editor.org/rfc/rfc9110.html#name-range), [Transformers fine-grained FP8](https://huggingface.co/docs/transformers/main/en/quantization/finegrained_fp8). Tài liệu nhánh main không phải bằng chứng tên/shape của checkpoint revision đã ghim; audit cần dữ liệu thực từ revision đó.


## 0.7.1
Total size validation is now diagnostic/report-only; architecture mapping remains pending.
