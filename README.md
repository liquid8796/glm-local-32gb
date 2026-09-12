# ModelDesk — GUI C# và Python core GLM

Mở `ModelDesk.sln` bằng Visual Studio 2026 hoặc chạy `build-studio.bat -Test`. GUI WPF/.NET 10 có bảy tab quản lý model, kiểm chứng, Hugging Face, tải xuống, báo cáo và cài đặt; CLI dùng chung dịch vụ với GUI. GUI/CLI gọi Python/native core qua cùng các entrypoint. [Hướng dẫn ModelDesk](studio/README.md) · [Kiến trúc source](studio/ARCHITECTURE.md).

```powershell
.\modeldesk.bat
.\modeldesk-cli.bat help
.\clean-project.bat          # xem trước cache có thể dọn
.\clean-project.bat -Apply   # chỉ xóa file cache; giữ model, venv, DLL và evidence
```

Trình tải hỗ trợ chia đoạn cho file lớn, tối đa bốn kết nối mỗi file/tám kết nối tổng, resume có checkpoint và xác minh hash. [Benchmark có điều kiện kiểm soát](docs/verification/modeldesk-download-benchmark.md); không phải cam kết tốc độ Internet. Quy trình kiểm tra, commit và tự push `origin/master` được ghi trong `AGENTS.md`.

## Python core — đọc trọng số FP8/NVFP4 theo khối

**Trạng thái 0.10.1: core tối ưu CPU và chẩn đoán timeout, đi kèm ModelDesk 1.0.3.** Phép nhân BF16/F16/F32 theo tile chạy bằng native CPU; NVFP4 CPU gom các tile trong một row band để giảm số lần gọi Python, giữ thứ tự tính FP32. Planner hạch toán bộ đệm đọc tối đa 8 MiB và scratch NVFP4 5 MiB. Lượt chạy có nhật ký tiến độ và lưu giai đoạn cuối khi timeout. Cần chạy `build-native.bat` khi cập nhật source để có các entrypoint native mới.

Profile NVFP4 vẫn là mặc định; profile FP8 cũ được giữ riêng. Kiểm chứng full-checkpoint và tốc độ thực tế là các bước riêng với kiểm thử kernel/metadata.

Model đang làm việc: `dealignai/GLM-5.3-ABLITERATED-NVFP4`, revision `371bdb985d0124e76348c91e4a8fcf3a9d719d09`. Đã xác minh đủ **282 header, 232.385 tensor, 57.600 bộ weight/scale NVFP4**; metadata và architecture PASS. Reader xử lý expert U8 đóng gói E2M1, scale E4M3 theo nhóm16 và scale toàn tensor F32; attention/shared/dense/MTP giữ BF16.

Kiểm chứng bản 0.10.0 trên Windows: **728/728 test đạt, không bỏ qua**. Projection expert thật logical 2048×6144 chạy CPU/GPU có sai số 0 so với tham chiếu FP32 độc lập; chỉ tải 7,08MB tensor và scale. [Bằng chứng NVFP4](docs/verification/nvfp4-v0.10.0.json).

Runtime giải mã trọng số NVFP4 rồi tính FP32 trên CPU/CUDA, phù hợp đường fallback của RTX3070. Không mô phỏng activation W4A4 hoặc cache FP8 của runtime NVIDIA. Kết quả inference toàn checkpoint vẫn chưa nghiệm thu.

```powershell
.\build-native.bat
.\glm.bat --profile nvfp4 runtime-plan --backend hybrid --context 4096
.\glm.bat --profile nvfp4 projection-check --online --backend hybrid --budget-mib 16
.\glm.bat --profile nvfp4 tokenizer-check
.\glm.bat --profile fp8 doctor
```

Cần build lại khi cập nhật native kernels; PTX được driver nạp trực tiếp. [Hướng dẫn NVFP4 và chuyển profile](docs/NVFP4.md). Báo cáo NVFP4 nằm trong `reports/nvfp4/`; dữ liệu và báo cáo FP8 cũ vẫn giữ nguyên. Nghiệm thu bản 0.10.0 chỉ tải metadata và một projection nhỏ; người dùng đã hoàn tất tải checkpoint ở lượt sau.

## Mốc FP8 trước đây — v0.9.0

**Trạng thái 0.9.0: đã triển khai decoder theo config, reader BF16/FP8 cho catalogue đầy đủ, planner CPU/GPU, tokenizer và kiểm chứng projection thật. Chưa nghiệm thu suy luận full checkpoint.**

Giữ đúng `dealignai/GLM-5.3-CYBERSECURITY-FP8`, revision `5915c1b88f998a9c1e1a0c83688e285a08ae3ca5`. Đã kiểm tra **282/282 header, 118.629 tensor, 78 layer backbone + 1 layer MTP**, metadata và architecture PASS. Projection thật `model.layers.0.self_attn.q_a_proj.weight` (2048×6144) chạy CPU/GPU với 384 khối mỗi bên và khớp tham chiếu FP32 độc lập, sai số tối đa 0. Chỉ tải khoảng 12,59 MB cho projection và 20,22 MB tokenizer, không tải bộ trọng số 756 GB.

```powershell
.\glm.bat runtime-plan --context 4096 --generate 32
.\glm.bat projection-check --online --backend hybrid --budget-mib 32
.\glm.bat tokenizer-check --online
```

[Các checkpoint và nghiệm thu](docs/CHECKPOINTS.md) · [Hướng dẫn runtime](docs/RUNTIME.md). `glm.bat` tự dùng `.venv-reference` nếu có, hoặc Python hệ thống. `generate` chạy thử nghiệm bằng token ID hoặc prompt với tokenizer cục bộ; cần đủ shard tại `--model-directory`. Theo lựa chọn của người dùng, lượt này chỉ hoàn thiện mã và kiểm tra nhỏ.

Kiểm chứng Windows bản 0.9.0: **629/629 test đạt, không bỏ qua**, bật native CPU/CUDA và graph Transformers chính thức. [Bằng chứng](docs/verification/runtime-v0.9.0.json) · [Log đầy đủ](docs/verification/runtime-tests-windows-v0.9.0.txt).

Đã kiểm chứng bản sửa trên Windows: **513 test đạt**, không lỗi hoặc skip; official hybrid safetensors tới **120 + 8 token** đều PASS, tối đa 2 shard mở đồng thời. [Báo cáo đóng review](docs/REVIEW-FIXES.md).

`architecture-check` đọc file JSONL được report metadata tham chiếu và xác minh nguồn bằng digest/identity trước khi phân tích. Tên tensor được ánh xạ theo mẫu đầy đủ; projection `gate_proj` và tensor scale có vai trò riêng với MoE router. Profile được hỗ trợ phải khớp config, dtype, shape và đủ tensor từng layer/expert. Tên hoặc profile chưa hỗ trợ trả `REVIEW_REQUIRED`; report hoặc bằng chứng nguồn không hợp lệ trả `ERROR`. Chênh lệch `index.total_size` chưa giải thích được cũng chặn metadata PASS. [Hướng dẫn architecture](docs/ARCHITECTURE-MAPPER.md).

Các kết quả metadata/architecture chỉ kiểm tra cấu trúc. Chúng không bật cờ tương thích checkpoint thật, tính đúng suy luận hoặc giới hạn tài nguyên full model; `doctor` vẫn `BLOCKED`.

Baseline **0.6.1 đã được nghiệm thu trên Windows** qua `reports(2).zip`: 376 test OK và bốn ca official hybrid safetensors PASS, gồm 120 + 8 token; serializer TensorSpec chạy với safetensors 0.8.0 thực. Đây là bằng chứng người dùng gửi trước bản vá này, không phải lần chạy Windows mới. [Biên bản](docs/verification/windows-acceptance-v0.6.1.md).

`metadata-check` (được thêm ở bản 0.7.0) đọc model manifest, `config.json`, index và **chỉ prefix/header** của từng shard; đối chiếu index/header, tensor dtype/shape và cặp scale 128×128 theo profile hiện tại. Tensor chưa được nhận diện vẫn được ghi vào catalogue là `not_reviewed`; không dùng mapping miniature để tuyên bố tương thích GLM thật. [Hướng dẫn metadata](docs/CHECKPOINT-METADATA.md) · [Current Memory Snapshot](docs/CURRENT-MEMORY.md).

```powershell
.\test-reference.bat 2>&1 | Tee-Object -FilePath .\reports\test-reference-v0.8.1.log
.\glm.bat metadata-check
.\glm.bat architecture-check
```

Giữ `.venv-reference` và `build` hiện có khi chép bản mới. **Không cần build lại DLL hoặc cài thêm thư viện** cho lệnh metadata; kernel C/CUDA, graph, dependency/revision lock và quota giữ nguyên. Parser header được tách thành hàm dùng chung cho reader cũ và checker mới, không nới policy reader. Trên Windows, metadata worker cũng được gắn Job Object trước khi chạy (CPU 70%, committed memory 32.000.000.000 byte theo cấu hình); không tạo CUDA context.

Report: `reports/metadata-latest.json`/Markdown và `reports/metadata/<run-id>/`. Mặc định đọc tối đa 512 shard và 64 MiB body metadata; **không dùng tải toàn file khi server bỏ qua Range**. `PASS` chỉ có nghĩa bước metadata đạt, không gỡ `doctor BLOCKED`.

**Bằng chứng lịch sử bản 0.7.0 tại Linux:** 453 test được phát hiện, 427 đạt, 26 bỏ qua, gồm 87 test mới không cần mạng/GPU. [Log](docs/verification/unit-tests-linux-v0.7.0.txt). Lần thử mạng thật của bản đó dừng ở DNS khi lấy model manifest: **0 byte metadata, 0 yêu cầu Range**; chưa xác minh config/index/header từ xa của checkpoint. [Báo cáo](docs/verification/metadata-online-attempt-linux-v0.7.0.json). Đây không phải lần nghiệm thu Windows/RTX 3070/Transformers cho bản 0.8.1.

Project được tạo cho `dealignai/GLM-5.3-CYBERSECURITY-FP8`, giữ đúng checkpoint FP8 theo yêu cầu. Kimi K3 tại commit `ac1584a70205c3a00d5346f736834818f4cc11b4` là nguồn tham khảo cho cách đọc trọng số theo nhu cầu, đặt ngân sách bộ nhớ và kiểm tra tính đúng trước benchmark. Runtime không phụ thuộc source Kimi; bản sao này không còn được theo dõi hoặc đóng gói cùng project. Link upstream được giữ trong phần nguồn kiểm chứng. Kernel C/PTX và core Python trong project là đường chạy hiện tại.

## Dùng ngay trên máy này

Yêu cầu Windows x64, Python 3.10+, driver NVIDIA cho phép thử GPU và Visual Studio C++ tools để build phần CPU. Các thành phần này đã có trên máy; DLL CPU đã được build. Lệnh `mini` cần NumPy cho tham chiếu độc lập; máy hiện có NumPy 2.4.6. Các lệnh kiểm tra phần cứng và `probe` vẫn dùng Python standard library. Không cần CUDA Toolkit cho các phép thử hiện tại.

Mở PowerShell tại thư mục project:

```powershell
.\build-native.bat
.\glm.bat storage-check --backend hybrid
.\test-reference.bat
```

Máy này đã được tạo môi trường `.venv-reference` riêng với Torch CPU và Transformers ghim revision. Double-click `parity.bat` để chạy đối chiếu; báo cáo ở `reports/parity-latest.md`. Trên máy khác chạy `setup-reference.bat` một lần; script chỉ cài thư viện trong môi trường project, không tải checkpoint. [Hướng dẫn đối chiếu chính thức](docs/OFFICIAL-PARITY.md).

Double-click `storage-check.bat` để kiểm tra reader với file safetensors do thư viện chính thức tạo. Ba kích thước thử là 256×384, 257×259 và 1×1. Báo cáo ở `reports/storage-latest.md`; [hướng dẫn đọc trọng số theo khối](docs/SAFETENSORS.md).

Double-click `mini.bat` để thử mô hình thu nhỏ với chuỗi 8, 32 và 64 token, sinh thêm 4 ID mỗi chuỗi. Báo cáo ở `reports/mini-latest.md`. ID token thuộc bộ từ vựng giả lập 32 phần tử, không phải văn bản có nghĩa. [Hướng dẫn và các giới hạn](docs/MINI-DECODER.md).

Trên máy khác chưa có NumPy, cài phần phụ thuộc kiểm chứng bằng `python -m pip install "numpy>=2.0,<3.0"`. Lệnh kiểm thử mặc định bỏ qua tham chiếu NumPy nếu thiếu; `test-native.bat` yêu cầu đủ NumPy, DLL C và GPU để kiểm tra đầy đủ.

Double-click `probe.bat` để chạy thử mặc định và giữ cửa sổ mở. Lần chạy sẽ tự tạo file trọng số giả lập khoảng 144 KiB; không dùng model thật. Báo cáo ở `reports/backend-probe-latest.md`.

Các công cụ kiểm tra vẫn hoạt động:

```powershell
.\glm.bat doctor
.\glm.bat policy-check
.\glm.bat monitor --seconds 10
.\test.bat
```

Hoặc double-click `doctor.bat` để xem báo cáo và giữ cửa sổ mở. `glm.bat` dùng cho terminal và automation, không dừng chờ phím.

- `doctor`: kiểm tra metadata đã lưu và cấu hình phần cứng hiện tại; ghi `reports/latest.md` và `reports/latest.json`. Mã thoát **2** nghĩa là chưa đủ điều kiện chạy model; mã **1** là lỗi công cụ/đầu vào.
- `doctor --refresh`: đọc lại **metadata của revision đã ghim**, không tải trọng số.
- `policy-check`: tạo Job Object, đọc lại hạn mức đã cài và chạy một tiến trình Python nhỏ. Kết quả lưu tại `reports/policy-check.json`.
- `monitor`: chỉ quan sát GPU và hiển thị đề xuất điều tiết; **không điều khiển GPU**. Kết quả lưu tại `reports/gpu-observation.json`.
- `test.bat`: chạy unit test và kiểm thử Job Object thực trên Windows.
- `probe`: đọc từng khối FP8 tối đa 128×128 từ file giả lập, chia các nhóm hàng đầu ra cho CPU/GPU, đối chiếu tham chiếu độc lập và đo tài nguyên trong worker đã được gắn Job Object.
- `test-native.bat`: bật thêm kiểm thử chạy DLL C và CUDA thật. Thiết bị/compiler bị thiếu sẽ báo lỗi ở những test được yêu cầu; không chuyển ngầm sang CPU.
- `mini`: kiểm tra decoder có MLA, RoPE, chọn vị trí chú ý thưa, MoE, cache, residual và đầu ra dự đoán token. Engine xử lý từng token; tham chiếu NumPy tính lại cả chuỗi bằng triển khai riêng.
- `parity`: đối chiếu cùng fixture với graph Transformers nguyên trạng trên CPU FP32, gồm 12 hidden states mỗi token, logits và lựa chọn attention/expert. Phía native hybrid vẫn chạy output head trên GPU.
- `test-reference.bat`: kiểm tra provenance rồi chạy toàn bộ test, bật cả Torch/Transformers, C và CUDA.
- `storage-check`: đối chiếu metadata/byte tensor, đọc FP8 với scale F32 theo khối 128×128 rồi tính trên CPU/GPU. Không mở checkpoint model thật.
- `metadata-check`: audit config/index/header từ revision đã ghim hoặc snapshot offline; lưu catalogue JSONL cùng bằng chứng nguồn.
- `architecture-check`: đọc report metadata và catalogue được tham chiếu, kiểm tra profile cấu trúc có config; không chạy graph hoặc đọc payload.

Lệnh `generate` đã có đường backbone theo config với trọng số cục bộ và cache MLA nén. Đây là runtime thử nghiệm FP32, chưa chứng minh tương đương đường BF16 chính thức và không chạy MTP speculative decoding. `doctor` tiếp tục báo `BLOCKED` cho đến khi có kiểm chứng full checkpoint và tài nguyên. Sửa trường `backend_status` trong JSON không mở khóa trạng thái này.

Kiểm chứng ngày 2026-09-11: **216 test đạt** khi bật native CPU/CUDA và tham chiếu NumPy. Decoder thu nhỏ đã khớp logits, lựa chọn attention/expert và ID sinh độc lập ở cả chuỗi 120 + 8 token. Sai số logits lớn nhất khoảng 1,18e-7; RSS worker đỉnh 138,48 MiB trong lần kiểm tra biên. Policy readback xác nhận CPU 70% / commit 32.000.000.000 byte. [Chi tiết miniature](docs/MINI-DECODER.md) và [phép thử FP8 trước đó](docs/FP8-PROBE.md).

Cập nhật 2026-09-12: **251 test đạt** trong môi trường tham chiếu, gồm graph Transformers chính thức và helper top-k native. Ca official hybrid tới 128 token khớp cả 12 hidden states/token; sai số logits lớn nhất khoảng 2,39e-7. [Kết quả và phạm vi kiểm chứng](docs/OFFICIAL-PARITY.md).

Bản 0.5.0: **301 test đạt**. Reader safetensors đọc đúng byte/dtype/shape của sáu tensor thử, FP8 block matvec đạt trên CPU và GPU, sai số lớn nhất khoảng 1,85e-6. [Chi tiết reader và giới hạn](docs/SAFETENSORS.md).

## Cấu hình đã chốt

Sửa `config/local.json` để thay đổi thư mục model hoặc giảm ngân sách. `model_directory` có thể là đường dẫn tuyệt đối trên ổ khác; đường dẫn tương đối được tính từ thư mục project.

| Yêu cầu | Cấu hình | Mức thực hiện hiện tại |
|---|---|---|
| Giữ model FP8 | Revision `5915c1b88f998a9c1e1a0c83688e285a08ae3ca5` | Chỉ tải metadata; chưa tải trọng số |
| RAM tối đa 32 GB | 32.000.000.000 byte | Job Object giới hạn **committed memory**; chưa chứng minh trần RAM vật lý |
| CPU tối đa 70% | CPU hard cap 70% cho job | Đã đọc lại policy và kiểm thử tiến trình con |
| GPU trung bình khoảng 60% | Cửa sổ 10 giây, cho phép tăng ngắn hạn | Pacing đã áp dụng trước mỗi khối GPU trong phép thử nhỏ; chưa kiểm chứng với GLM |

CPU quota áp dụng cho các tiến trình trong job, tính theo khoảng lập lịch của Windows. Các ứng dụng khác vẫn có thể làm CPU toàn máy vượt 70%. Job cha có thể siết thêm quota. Job memory tính commit, không bao gồm đầy đủ file-backed resident pages, cache hệ điều hành hoặc mọi phần cấp phát trong driver. Không được diễn giải nó thành bảo đảm tổng RAM của máy dưới 32 GB. 32 GB ở đây bằng khoảng 29,80 GiB.

Pacing dừng gửi thêm công việc ở ranh giới giữa các khối tính toán và tiếp tục lấy mẫu. Mất dữ liệu đo, dữ liệu cũ hoặc chưa đủ thời gian quan sát đều chặn gửi thêm công việc. Quá 10 giây không được phép tiếp tục thì phép thử báo lỗi. Không ngắt kernel đang chạy. Cơ chế là heuristic, không bảo đảm mức trung bình khi ứng dụng khác dùng GPU. Phép thử rất ngắn nên số đo GPU không phải benchmark khả năng giữ tải 60%.

## Kết quả kiểm tra ngày 2026-09-11

Máy: i7-11800H, 8 nhân/16 luồng, khoảng 64 GiB RAM lắp đặt, RTX 3070 Laptop 8 GiB VRAM, compute capability 8.6. Dung lượng còn trống thay đổi theo thời điểm; xem `reports/latest.md` để có số đo mới nhất.

Checkpoint có **282 shard**, tổng **755.631.998.192 byte** (~755,63 GB / 703,74 GiB). Ngân sách ổ đĩa kiểm tra cộng thêm 20 GB dự phòng; đây không phải ước lượng dung lượng cho chuyển đổi/đóng gói backend sau này. Tại lúc kiểm tra, không có ổ đơn lẻ đủ trống để chứa checkpoint. Tool chỉ kiểm tra một thư mục model; chưa hỗ trợ phân tán shard qua nhiều ổ.

GLM và Kimi có graph khác nhau. Kimi hiện chỉ chạy CPU. GPU này cần đường tính fallback khi đọc FP8; giữ FP8 trên đĩa không có nghĩa GPU tính FP8 native. Không có số tokens/giây nào đã được đo trên model này và máy này.

## Thành phần trong project

| Đường dẫn | Chức năng |
|---|---|
| `glm_local/checkpoint_check.py` | Điều phối audit metadata, Windows worker, catalogue và report theo từng run |
| `glm_local/checkpoint_http.py` | HTTP Range nghiêm ngặt, ghim revision, giới hạn body/read/redirect/thời gian |
| `glm_local/checkpoint_snapshot.py` | Snapshot header/JSON có hash và replay offline không gọi mạng |
| `glm_local/checkpoint_schema.py` | Đối chiếu index/header/config và phát hiện khoảng trống profile FP8 |
| `glm_local/architecture/` | Xác minh nguồn catalogue, phân loại vai trò chính xác và kiểm tra cấu trúc theo config |
| `glm_local/metadata.py` | Đọc JSON công khai có giới hạn kích thước, xác minh ID/revision và manifest |
| `glm_local/hardware.py` | Đọc CPU/RAM/ổ đĩa qua CIM và GPU qua nvidia-smi |
| `glm_local/audit.py` | Kiểm tra ngân sách, shard còn thiếu và những điều kiện chưa đáp ứng |
| `glm_local/winjob.py` | Tạo child suspended, gắn job trước khi resume, cleanup cả cây tiến trình |
| `glm_local/pacing.py` | Trung bình theo thời gian và đề xuất chờ ở ranh giới tính toán |
| `native/fp8_cpu.c` | Primitive CPU C, giải mã E4M3FN rồi nhân/cộng FP32 |
| `native/fp8_matvec.ptx` | Primitive GPU với cách tính FP32 tương ứng |
| `glm_local/backend_probe.py` | Phối hợp CPU/GPU với dữ liệu giả lập và đối chiếu tham chiếu |
| `glm_local/synthetic_fp8.py` | Fixture riêng có kích thước giới hạn; không đọc safetensors |
| `glm_local/gpu_gate.py` | Đọc telemetry và chặn gửi thêm khối GPU khi cần chờ |
| `glm_local/process_metrics.py` | Đọc RSS/private commit/CPU của worker qua Win32 |
| `glm_local/mini_engine.py` | Decoder thu nhỏ xử lý từng token với cache có kích thước giới hạn |
| `glm_local/mini_reference.py` | Tham chiếu NumPy độc lập, tính cả chuỗi, đọc fixture bằng loader riêng |
| `glm_local/mini_weights.py` | Tạo 9.440 byte trọng số giả lập; kiểm tra hash và chỉ đọc ma trận cần dùng |
| `glm_local/mini_run.py` | Đối chiếu logits, lựa chọn attention/expert, hệ số MoE và ID sinh độc lập |
| `glm_local/official_reference.py` | Ánh xạ toàn bộ fixture sang model chính thức; lấy kết quả bằng hooks |
| `glm_local/parity_run.py` | Đối chiếu hai graph dưới Windows Job Object, lưu báo cáo đầy đủ |
| `native/topk_cpu.cpp` | Bộ chọn FP32 top-k có cách xử lý bằng điểm tương thích runtime đã ghim |
| `config/reference-lock.json` | Revision, phiên bản và hash source cho tham chiếu chính thức |
| `glm_local/safetensor_reader.py` | Reader safetensors với header tối đa 1 MiB, mỗi lần đọc tối đa 64 KiB |
| `glm_local/sharded_safetensors.py` | Index/header validation, lazy payload và LRU giới hạn file mở |
| `glm_local/mini_safetensors.py` | Fixture bốn shard và adapter trọng số cho decoder cố định |
| `glm_local/mini_storage.py` | Chọn storage native chung cho mini/parity; oracle không đổi |
| `glm_local/fp8_blocks.py` | Ánh xạ cặp tensor FP8/scale được chỉ định rõ ràng thành các khối 128×128 |
| `glm_local/safetensor_fixture.py` | Tạo file thử bằng safetensors chính thức và tham chiếu giải mã độc lập |
| `glm_local/safetensor_check.py` | Kiểm chứng byte/scale/tính toán native và ghi báo cáo dưới Job Object |
| `docs/model-metadata.json` | Snapshot manifest đã kiểm tra, để dùng offline |
| `docs/BACKEND.md` | Phần backend còn phải phát triển và các điều kiện nghiệm thu |

Không cần tải Git submodule sau khi clone. Thư mục trọng số `/models/`, báo cáo máy và output build được loại khỏi Git; profile `config/models/` và snapshot `docs/models/` vẫn được theo dõi đầy đủ. Các script không thay đổi power limit, clock hoặc driver.

## Nguồn kiểm chứng

- [Repo Kimi K3](https://github.com/FareedKhan-dev/kimi-k3-in-c/tree/ac1584a70205c3a00d5346f736834818f4cc11b4)
- [Cấu hình checkpoint đã ghim](https://huggingface.co/dealignai/GLM-5.3-CYBERSECURITY-FP8/blob/5915c1b88f998a9c1e1a0c83688e285a08ae3ca5/config.json)
- [Microsoft: CPU rate control](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_cpu_rate_control_information)
- [Microsoft: giới hạn committed memory](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information)
- [NVIDIA: GPU utilization và memory là các chỉ số khác nhau](https://docs.nvidia.com/deploy/nvidia-smi/index.html)

Mã của project dùng MIT. Kimi upstream được tham khảo qua link và giữ license Apache-2.0 riêng; source đó không được đóng gói cùng tool. License checkpoint tách biệt với license công cụ.
