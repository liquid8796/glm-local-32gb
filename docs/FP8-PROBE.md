# Phép thử FP8 CPU/GPU 0.2.0

Phép thử này xác minh **một primitive nhân ma trận-vector** và đường đọc khối trọng số với dữ liệu tự sinh. Nó không tạo token, không có attention/MoE/KV cache/tokenizer và không đọc checkpoint Hugging Face.

## Chạy lại

```powershell
.\build-native.bat
.\glm.bat probe --backend hybrid
.\test-native.bat
```

DLL CPU dùng MSVC x64 `/O2 /fp:strict`; GPU dùng PTX do NVIDIA driver biên dịch cho thiết bị. Không tải package, CUDA Toolkit hoặc trọng số model. `build-native.bat` chỉ đọc môi trường compiler trong tiến trình con, không thay đổi PATH hay môi trường hệ thống.

Kiểm chứng ngày 2026-09-11: **137 test đạt khi bật cả native CPU và CUDA thật**. Đã kiểm tra thêm phép thử CPU 17×131 và hybrid 384×384 dưới Job Object, đều đạt. `test.bat` thông thường bỏ qua các test native cần bật rõ ràng; dùng `test-native.bat` để kiểm tra đầy đủ trên máy có compiler/DLL và GPU phù hợp.

Có thể chọn `--backend cpu`, `--backend gpu` hoặc `--backend hybrid`. Mặc định là ma trận 384×384, seed 7 và 3 lượt. Các tham số `--rows`, `--cols`, `--iterations`, `--seed` được kiểm tra trước khi tạo worker. Giới hạn mỗi chiều 1024, tối đa 8 lượt và 256 kernel GPU trong một phiên. Hybrid cần hơn 128 hàng để cả hai thiết bị có việc.

`reports/probes/<run-id>/` lưu input cấu hình, fixture và kết quả riêng mỗi lần. `reports/backend-probe-latest.md` và JSON tương ứng chỉ tới kết quả lần gần nhất. Mã thoát 0 là phép thử số học đạt, 3 là sai lệch số học, 1 là lỗi thiết bị/đầu vào/tài nguyên. Không mã nào chứng minh model GLM đã chạy.

## Đường tính toán

1. Tạo file `FP8PROBE` nhỏ, chứa khối byte E4M3FN và scale FP32. Fixture mặc định 147.516 byte. File đã tồn tại không bị ghi đè.
2. Tạo worker ở trạng thái suspended, gắn Job Object với CPU 70% và commit 32.000.000.000 byte, đọc lại policy rồi mới cho worker chạy.
3. Đọc không dùng mmap, mỗi lần đọc không quá 16.384 byte. Scale/shape/độ dài/tính hữu hạn được kiểm tra. File này có thể nằm trong cache hệ điều hành; phép thử chưa đo băng thông SSD lạnh.
4. CPU xử lý các nhóm hàng chẵn, GPU xử lý nhóm hàng lẻ trong chế độ hybrid. Từng khối hoàn thành trước khi đọc khối kế tiếp; chưa tối ưu chạy chồng lấp CPU/GPU.
5. Giải mã FP8 sang số FP32, nhân scale, nhân vector rồi cộng FP32. RTX 3070 không tính tensor-core FP8 native trong bài thử này. C và PTX là hai triển khai riêng.
6. So sánh mọi hàng đầu ra với tham chiếu tạo trực tiếp từ tọa độ/seed, giải mã toán học độc lập và `math.fsum` float64. Tham chiếu không gọi reader, kernel C hoặc CUDA.

Theo [định nghĩa E4M3 của NVIDIA](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp8__e4m3.html), mã NaN là 0x7F và 0xFF. Các bài kiểm tra decoder bao phủ signed zero, subnormal và giá trị hữu hạn lớn nhất; wrapper từ chối NaN trước khi gửi kernel.

## Tài nguyên và ý nghĩa kết quả

- Job CPU cap áp dụng theo khoảng lập lịch và cho cây tiến trình trong job; không kiểm soát ứng dụng khác.
- Commit cap 32 GB không phải trần tổng RAM vật lý. RSS và private commit được đọc từ Win32 của riêng worker. Peak là số đỉnh suốt vòng đời worker, gồm khởi tạo CUDA; telemetry subprocess và file cache hệ điều hành không nằm trong con số RSS của worker.
- Có kiểm tra peak RSS sau mỗi lượt và hủy khi đã quan sát vượt ngân sách. Đây là kiểm tra sau sự kiện, không thể ngăn mọi đỉnh tức thời.
- Bộ đệm CUDA được cấp phát rõ ràng tối đa 17.408 byte mỗi khối. Driver/context/JIT có bộ nhớ bổ sung, nên không được dùng con số này làm tổng VRAM.
- Pacing đọc GPU trước các lần gửi việc, chờ warmup và dừng nếu dữ liệu thiếu/cũ hoặc thiết bị bận quá thời hạn. GPU được đo cho toàn thiết bị, gồm cả ứng dụng khác. Kernel rất ngắn có thể lọt giữa các lần lấy mẫu.
- Cửa sổ mục tiêu là 10 giây, nhưng phép thử mặc định thường chỉ có vài giây quan sát. Kết quả trung bình chỉ bao phủ `observed_seconds`, chưa chứng minh giữ 60% trong một tải dài.

Lần kiểm tra hybrid đầu tiên ngày 2026-09-11 đạt với 18 khối CPU và 9 khối GPU, sai số lớn nhất 3,81469727e-6, RSS worker đỉnh khoảng 126,80 MiB và private commit đỉnh 255,51 MiB. Đây là số đo của dữ liệu giả lập 384×384; không suy ra lượng RAM hoặc tốc độ của checkpoint 755 GB.

`doctor` tiếp tục báo `BLOCKED` cho model đầy đủ. Công việc còn lại được ghi trong [BACKEND.md](BACKEND.md).
