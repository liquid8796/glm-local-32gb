# Decoder giả lập 0.3.0

**Mục tiêu:** kiểm tra các cơ chế của decoder trên trọng số tự tạo có kích thước cố định. Lệnh này không tải model, không đọc safetensors thật và không mở cổng mạng.

## Chạy

```powershell
.\glm.bat mini
.\glm.bat mini --backend cpu --lengths 8,32,120 --generate 8 --seed 19
.\test-native.bat
```

Máy hiện có đủ DLL CPU, NVIDIA driver và NumPy. Để build lại DLL dùng `build-native.bat`. Cửa sổ giữ mở khi double-click `mini.bat`; dùng `glm.bat mini` trong terminal để giữ nguyên mã thoát.

Mặc định: hybrid, prompt dài 8/32/64 ID, mỗi prompt sinh thêm 4 ID; seed 7. Cho phép 1–4 độ dài tăng dần, 1–8 ID mới, tổng prompt + ID mới không quá 128 cho mỗi ca, không quá 256 output-head GPU calls cho cả phiên hybrid. Giới hạn không thể nâng bằng config của checkpoint.

Kết quả riêng từng phiên nằm trong `reports/mini/<run-id>/`, bản gần nhất tại `reports/mini-latest.md` và `.json`. Mã thoát 0 nghĩa là miniature khớp đối chiếu, 3 là sai lệch kết quả và 1 là lỗi/thiếu thành phần. `doctor` vẫn báo model thật chưa sẵn sàng.

## Kết quả đã chạy trên máy này

Ngày 2026-09-11, **216 test đạt**, gồm native C/CUDA thật và các test toán học độc lập. Ca mặc định 8/32/64 + 4 token đều đạt. Ca kiểm tra biên hybrid, seed 19, cho kết quả:

| Prompt + ID mới | Sai số logits lớn nhất | ID greedy/attention/expert | Cache payload đang dùng |
|---|---:|---|---:|
| 8 + 8 | 7,10e-8 | Khớp | 7.168 byte |
| 32 + 8 | 7,95e-8 | Khớp | 17.920 byte |
| 120 + 8 | 1,17e-7 | Khớp | 57.344 byte |

Toàn phiên biên có 4.784 phép chiếu CPU và 184 phép chiếu GPU. RSS worker đỉnh 138,48 MiB; private commit đỉnh 299,09 MiB. Đây là số đo của mô hình giả lập rất nhỏ, không suy ra tài nguyên cần cho checkpoint 755 GB. Kiểm thử khi không có site-packages cũng xác nhận bộ test mặc định bỏ qua NumPy tùy chọn đúng cách.

## Những phần được kiểm tra

- Hai layer, hidden 16, từ vựng 32, hai attention heads. Layer đầu có dense FFN và indexer đầy đủ; layer sau có 4 experts chọn 2, một shared expert, tái sử dụng vị trí attention đã chọn.
- MLA với q/kv rank nhỏ; RMSNorm; RoPE; chỉ chọn các vị trí causal; indexer LayerNorm có cả gain/bias và hệ số head có dấu; shared-index layer có K/V riêng.
- MoE chọn bằng sigmoid score cộng correction bias nhưng cộng kết quả expert bằng sigmoid score gốc, chuẩn hóa và nhân scale 2,5; shared expert được cộng riêng.
- Engine thực hiện từng token và giữ cache. Oracle NumPy dùng math/loader riêng để tính cả chuỗi, không gọi engine hoặc kernel native.
- So sánh từng logit, vị trí attention, expert được chọn, hệ số kết hợp expert và chuỗi greedy sinh độc lập. Các bài unit test còn kiểm tra causality, reset, biên context, file hỏng và xử lý lỗi.

Thứ tự phép tính được khảo sát từ [Transformers tại revision đã ghim](https://github.com/huggingface/transformers/blob/3f601734a3580f55484720770850966bba060e4f/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py). Đây là triển khai độc lập với cấu hình thu nhỏ; chưa chạy đối chiếu trực tiếp bằng thư viện Transformers.

## Bộ nhớ, thiết bị và phạm vi số đo

Fixture có 34 ma trận, 9.440 byte FP8. Ma trận lớn nhất 512 byte. Reader kiểm tra SHA256, đọc không buffer tối đa 4.096 byte mỗi lần, không giữ cache payload FP8. Vector gain/bias vẫn thường trú. Oracle riêng đọc và giữ toàn bộ ma trận nhỏ dưới dạng float64; thống kê `storage.reader_stats` không bao gồm oracle.

Cache engine được cấp phát trước **57.344 byte payload** cho 128 vị trí. Mỗi vị trí dùng thêm 448 byte logic. Các số này chưa bao gồm overhead đối tượng Python, activations, trace, trọng số, NumPy, driver hay CUDA context. Báo cáo cũng ghi peak RSS và private commit của cả worker từ Win32 để thấy chi phí thực của tiến trình.

Hybrid hiện dùng CPU cho các phép chiếu bên trong và GPU cho output head. Pacing được áp dụng trước mỗi lần gửi GPU. Đây là kiểm chứng đường tính toán, chưa phải tối ưu chia tải. Job Object giữ CPU quota 70% và commit 32 GB; kiểm tra peak RSS sau token và sau oracle chỉ phát hiện vượt mức sau sự kiện. Không chứng minh trần RAM vật lý tuyệt đối hay mức GPU duy trì dưới tải dài.

Projection native tính FP32. Nonlinear/cache engine và oracle dùng float64. Chọn top-k khi bằng điểm ưu tiên chỉ số nhỏ hơn; quy tắc này không bảo đảm giống `torch.topk`. Chưa kiểm tra toàn bộ dtype, chiều/layer schedule, RoPE dài, tokenizer hoặc scale layout của checkpoint thật.

## Bước tiếp theo

Đối chiếu **cùng bộ trọng số giả lập** với Transformers chính thức, so sánh thêm hidden state trung gian và quy định chính xác dtype. Sau đó mới phát triển reader/tensor mapping cho checkpoint thật và đo tài nguyên ở kích thước lớn. Bước đối chiếu chính thức chưa cần tải checkpoint 755 GB hoặc mua thêm ổ đĩa.
