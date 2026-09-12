# Các checkpoint — 0.9.0

Đây là nghiệm thu FP8 lịch sử. Model mặc định đã chuyển sang NVFP4 ở0.10.0; xem [NVFP4.md](NVFP4.md). FP8 vẫn truy cập bằng `--profile fp8`.

Đích: `dealignai/GLM-5.3-CYBERSECURITY-FP8`, revision `5915c1b88f998a9c1e1a0c83688e285a08ae3ca5`.

| Mốc | Mã triển khai | Nghiệm thu trong lượt này |
|---|---|---|
| Catalogue thật | Hoàn tất | 282/282 header, 118.629 tensor, metadata/architecture PASS |
| Profile backbone/MTP | Hoàn tất cấu trúc | 78 + 1 layer, 59.044 cặp FP8/F32, không findings |
| Descriptor/projection FP8 | Hoàn tất | Projection thật 2048×6144 CPU/GPU PASS, sai số 0 |
| Reader catalogue lớn | Hoàn tất | BF16/F16/F32/FP8, nhiều shard, thay đổi file và giới hạn đọc |
| Planner CPU/GPU | Hoàn tất | Config thật 4096 context: khoảng 2,95 GB theo mô hình ước tính |
| Decoder/generation | Hoàn tất đường backbone thử nghiệm | Mô hình nhỏ qua native CPU/CUDA khớp Transformers |
| Tokenizer/CLI | Hoàn tất | Hash đúng, encode/decode tiếng Việt/Anh, Windows Job và lỗi đầu vào |
| Full-model numerical/resource acceptance | Có công cụ và đường chạy thử nghiệm | **Chưa nghiệm thu: chưa tải đủ trọng số theo lựa chọn người dùng** |

PASS ở các mốc trước không thay thế mốc cuối. `doctor` vẫn BLOCKED. MTP speculative generation, mọi biến thể RoPE/dtype và tối ưu throughput chưa thuộc đường backbone được hỗ trợ.

Bộ hồi quy cuối trên Windows: **629/629 test đạt**, không skip, gồm kernel C/CUDA và graph Transformers ghim revision.

[Bằng chứng bản 0.9.0](verification/runtime-v0.9.0.json) · [Hướng dẫn chạy](RUNTIME.md). Các thư mục run lớn và payload nhỏ giữ trong `reports/` trên máy, không đưa trọng số vào Git. Chạy `test-reference.bat` để kiểm tra đầy đủ.
