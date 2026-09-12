# GLM Local 32GB — decoder và đọc trọng số theo khối

**Trạng thái 0.7.0: thêm kiểm tra metadata checkpoint đã ghim bằng HTTP Range và snapshot offline. Không tải payload trọng số, chưa chạy checkpoint đầy đủ.**

Baseline **0.6.1 đã được nghiệm thu trên Windows** qua `reports(2).zip`: 376 test OK và bốn ca official hybrid safetensors PASS, gồm 120 + 8 token; serializer TensorSpec chạy với safetensors 0.8.0 thực. Đây là bằng chứng người dùng gửi trước bản vá này, không phải lần chạy Windows mới. [Biên bản](docs/verification/windows-acceptance-v0.6.1.md).

Bản 0.7.0 thêm `metadata-check`: đọc model manifest, `config.json`, index và **chỉ prefix/header** của từng shard; đối chiếu index/header, tensor dtype/shape và cặp scale 128×128 theo profile hiện tại. Tensor chưa được nhận diện vẫn được ghi vào catalogue là `not_reviewed`; không dùng mapping miniature để tuyên bố tương thích GLM thật. [Hướng dẫn metadata](docs/CHECKPOINT-METADATA.md) · [Current Memory Snapshot](docs/CURRENT-MEMORY.md).

```powershell
.\test-reference.bat 2>&1 | Tee-Object -FilePath .\reports\test-reference-v0.7.0.log
.\glm.bat metadata-check
```

Giữ `.venv-reference` và `build` hiện có khi chép bản mới. **Không cần build lại DLL hoặc cài thêm thư viện** cho lệnh metadata; kernel C/CUDA, graph, dependency/revision lock và quota giữ nguyên. Parser header được tách thành hàm dùng chung cho reader cũ và checker mới, không nới policy reader. Trên Windows, metadata worker cũng được gắn Job Object trước khi chạy (CPU 70%, committed memory 32.000.000.000 byte theo cấu hình); không tạo CUDA context.

Report: `reports/metadata-latest.json`/Markdown và `reports/metadata/<run-id>/`. Mặc định đọc tối đa 512 shard và 64 MiB body metadata; **không dùng tải toàn file khi server bỏ qua Range**. `PASS` chỉ có nghĩa bước metadata đạt, không gỡ `doctor BLOCKED`.

Kiểm thử mới tại Linux: **453 test được phát hiện, 427 đạt, 26 bỏ qua**, gồm **87 test mới** không cần mạng/GPU. [Log](docs/verification/unit-tests-linux-v0.7.0.txt). Lần thử mạng thật tại đây dừng ở DNS khi lấy model manifest: **0 byte metadata, 0 yêu cầu Range**; chưa xác minh config/index/header từ xa của checkpoint. [Báo cáo](docs/verification/metadata-online-attempt-linux-v0.7.0.json). Chưa chạy lại Windows/RTX 3070/Transformers đúng revision cho 0.7.0 tại môi trường này.

Project được tạo cho `dealignai/GLM-5.3-CYBERSECURITY-FP8`, giữ đúng checkpoint FP8 theo yêu cầu. Mã Kimi gốc được giữ bằng Git submodule tại `vendor/kimi-k3-in-c`, commit `ac1584a70205c3a00d5346f736834818f4cc11b4`. Các ý tưởng được dùng làm cơ sở là đọc trọng số theo nhu cầu, ngân sách bộ nhớ rõ ràng và kiểm tra tính đúng trước khi benchmark. Phần mới có kernel FP8 C cho CPU, PTX cho GPU và bộ điều phối thử nghiệm bằng Python. Chưa port graph Kimi sang GLM, chưa tải checkpoint thật.

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

Hiện chưa có lệnh chat hoặc suy luận checkpoint thật. `doctor` tiếp tục báo `BLOCKED` cho đến khi có kiểm chứng tương thích model và giới hạn tài nguyên đầy đủ. Sửa trường `backend_status` trong JSON không mở khóa trạng thái này.

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
| `vendor/kimi-k3-in-c/` | Source upstream nguyên trạng, ghim commit, giữ license Apache-2.0 |
| `glm_local/checkpoint_check.py` | Điều phối audit metadata, Windows worker, catalogue và report theo từng run |
| `glm_local/checkpoint_http.py` | HTTP Range nghiêm ngặt, ghim revision, giới hạn body/read/redirect/thời gian |
| `glm_local/checkpoint_snapshot.py` | Snapshot header/JSON có hash và replay offline không gọi mạng |
| `glm_local/checkpoint_schema.py` | Đối chiếu index/header/config và phát hiện khoảng trống profile FP8 |
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

Sau khi clone bản project có commit này ở nơi khác, khôi phục upstream bằng `git submodule update --init --recursive`. Thư mục model, báo cáo máy và trọng số được loại khỏi Git. Các script không thay đổi power limit, clock hoặc driver.

## Nguồn kiểm chứng

- [Repo Kimi K3](https://github.com/FareedKhan-dev/kimi-k3-in-c/tree/ac1584a70205c3a00d5346f736834818f4cc11b4)
- [Cấu hình checkpoint đã ghim](https://huggingface.co/dealignai/GLM-5.3-CYBERSECURITY-FP8/blob/5915c1b88f998a9c1e1a0c83688e285a08ae3ca5/config.json)
- [Microsoft: CPU rate control](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_cpu_rate_control_information)
- [Microsoft: giới hạn committed memory](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information)
- [NVIDIA: GPU utilization và memory là các chỉ số khác nhau](https://docs.nvidia.com/deploy/nvidia-smi/index.html)

Phần mã mới dùng MIT; upstream giữ nguyên Apache-2.0 và NOTICE riêng. License checkpoint tách biệt với license công cụ.
