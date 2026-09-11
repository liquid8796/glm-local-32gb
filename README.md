# GLM Local 32GB — nền tảng kiểm tra và quản lý tài nguyên

**Trạng thái 0.1.0: chưa có backend suy luận GLM, chưa chạy được model.**

Project được tạo cho `dealignai/GLM-5.3-CYBERSECURITY-FP8`, giữ đúng checkpoint FP8 theo yêu cầu. Mã Kimi gốc được giữ bằng Git submodule tại `vendor/kimi-k3-in-c`, commit `ac1584a70205c3a00d5346f736834818f4cc11b4`. Các ý tưởng được dùng làm cơ sở là đọc trọng số theo nhu cầu, ngân sách bộ nhớ rõ ràng và kiểm tra tính đúng trước khi benchmark. Project hiện bổ sung công cụ kiểm tra và các thành phần quản lý tài nguyên bằng Python; chưa port graph hoặc kernel Kimi sang GLM.

## Dùng ngay trên máy này

Yêu cầu Python 3.10+; máy hiện có Python 3.13.12. Không cần cài package Python bên ngoài.

Mở PowerShell tại thư mục project:

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

Hiện không có lệnh chat hay lệnh suy luận. `doctor` luôn báo `BLOCKED` cho đến khi backend, kiểm chứng nội dung checkpoint và kiểm chứng tài nguyên được triển khai. Sửa trường `backend_status` trong JSON không mở khóa trạng thái này.

Kiểm chứng ngày 2026-09-11: **55 test đạt**, policy readback xác nhận CPU 70% / commit 32.000.000.000 byte; launcher giữ đúng exit code 2 khi chưa sẵn sàng. Đã quan sát GPU thật trong 5 giây khi không chạy model. Các kiểm tra này chỉ xác minh công cụ hỗ trợ.

## Cấu hình đã chốt

Sửa `config/local.json` để thay đổi thư mục model hoặc giảm ngân sách. `model_directory` có thể là đường dẫn tuyệt đối trên ổ khác; đường dẫn tương đối được tính từ thư mục project.

| Yêu cầu | Cấu hình | Mức thực hiện hiện tại |
|---|---|---|
| Giữ model FP8 | Revision `5915c1b88f998a9c1e1a0c83688e285a08ae3ca5` | Chỉ tải metadata; chưa tải trọng số |
| RAM tối đa 32 GB | 32.000.000.000 byte | Job Object giới hạn **committed memory**; chưa chứng minh trần RAM vật lý |
| CPU tối đa 70% | CPU hard cap 70% cho job | Đã đọc lại policy và kiểm thử tiến trình con |
| GPU trung bình khoảng 60% | Cửa sổ 10 giây, cho phép tăng ngắn hạn | Bộ đề xuất pacing đã unit test; chưa nối với suy luận |

CPU quota áp dụng cho các tiến trình trong job, tính theo khoảng lập lịch của Windows. Các ứng dụng khác vẫn có thể làm CPU toàn máy vượt 70%. Job cha có thể siết thêm quota. Job memory tính commit, không bao gồm đầy đủ file-backed resident pages, cache hệ điều hành hoặc mọi phần cấp phát trong driver. Không được diễn giải nó thành bảo đảm tổng RAM của máy dưới 32 GB. 32 GB ở đây bằng khoảng 29,80 GiB.

Pacing cần backend chủ động dừng gửi thêm công việc ở ranh giới giữa các khối tính toán và tiếp tục lấy mẫu. Mất dữ liệu đo, dữ liệu cũ hoặc chưa đủ thời gian quan sát đều trả về `allow_work=False`. Không ngắt kernel đang chạy. Thời gian chờ chỉ là đề xuất heuristic, không bảo đảm mức trung bình nếu backend không tuân thủ hoặc ứng dụng khác dùng GPU.

## Kết quả kiểm tra ngày 2026-09-11

Máy: i7-11800H, 8 nhân/16 luồng, khoảng 64 GiB RAM lắp đặt, RTX 3070 Laptop 8 GiB VRAM, compute capability 8.6. Dung lượng còn trống thay đổi theo thời điểm; xem `reports/latest.md` để có số đo mới nhất.

Checkpoint có **282 shard**, tổng **755.631.998.192 byte** (~755,63 GB / 703,74 GiB). Ngân sách ổ đĩa kiểm tra cộng thêm 20 GB dự phòng; đây không phải ước lượng dung lượng cho chuyển đổi/đóng gói backend sau này. Tại lúc kiểm tra, không có ổ đơn lẻ đủ trống để chứa checkpoint. Tool chỉ kiểm tra một thư mục model; chưa hỗ trợ phân tán shard qua nhiều ổ.

GLM và Kimi có graph khác nhau. Kimi hiện chỉ chạy CPU. GPU này cần đường tính fallback khi đọc FP8; giữ FP8 trên đĩa không có nghĩa GPU tính FP8 native. Không có số tokens/giây nào đã được đo trên model này và máy này.

## Thành phần trong project

| Đường dẫn | Chức năng |
|---|---|
| `vendor/kimi-k3-in-c/` | Source upstream nguyên trạng, ghim commit, giữ license Apache-2.0 |
| `glm_local/metadata.py` | Đọc JSON công khai có giới hạn kích thước, xác minh ID/revision và manifest |
| `glm_local/hardware.py` | Đọc CPU/RAM/ổ đĩa qua CIM và GPU qua nvidia-smi |
| `glm_local/audit.py` | Kiểm tra ngân sách, shard còn thiếu và những điều kiện chưa đáp ứng |
| `glm_local/winjob.py` | Tạo child suspended, gắn job trước khi resume, cleanup cả cây tiến trình |
| `glm_local/pacing.py` | Trung bình theo thời gian và đề xuất chờ ở ranh giới tính toán |
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
