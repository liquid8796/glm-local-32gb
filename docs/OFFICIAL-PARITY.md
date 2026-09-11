# Đối chiếu Transformers 0.4.0

Lệnh `parity` so sánh **fixture tự tạo 2 layer**, không phải checkpoint `dealignai/GLM-5.3-CYBERSECURITY-FP8`. Toàn bộ 34 ma trận và các vector của fixture được nạp chính xác vào state_dict chính thức; không để lại tham số ngẫu nhiên. Không gọi `from_pretrained`, tokenizer hoặc tải model.

## Chạy lại

```powershell
.\build-native.bat
.\glm.bat parity
.\glm.bat parity --backend hybrid --lengths 8,32,120 --generate 8 --seed 19
.\test-reference.bat
```

Trên máy hiện tại `.venv-reference` đã sẵn sàng. Máy khác cần `setup-reference.bat` trước. Script đọc lock, tạo venv cục bộ, cài Torch CPU từ index chính thức và Transformers từ archive có revision/hash cụ thể. Khi môi trường đã đúng, chạy lại setup sẽ chỉ kiểm tra, không cài hay tải lại. Môi trường cần Python 3.13 x64 để tái lập cấu hình đã kiểm tra trên máy này.

Package chính: Torch **2.14.0+cpu**, Transformers **5.18.0.dev0** từ revision **3f601734a3580f55484720770850966bba060e4f**, NumPy **2.4.6**. `requirements-reference.lock.txt` ghi lại phụ thuộc đã cài. Verifier kiểm tra ba phiên bản, URL/hash archive và hash hai file model/config; không tuyên bố kiểm chứng mọi byte của cả môi trường.

Worker chạy với CPU quota 70% và committed memory 32 GB, Torch/BLAS một luồng, cờ Hugging Face offline. Việc cài thư viện cần mạng; chính phép đối chiếu chạy offline. GPU native vẫn dùng driver/PTX như các bản trước, không phụ thuộc Torch CUDA.

## Đối chiếu những gì

- Cùng token đầu vào, cùng trọng số FP8 được giải mã; official dùng CPU FP32 và eager attention. Graph official không bị thay hoặc monkeypatch.
- 12 trạng thái/token: embedding; input norm, attention output, residual sau attention, post-attention norm, output ở mỗi layer; final norm. Thu bằng hooks, không sửa phép tính.
- Mọi logit, tập vị trí attention, tập expert và hệ số tương ứng theo ID. Không yêu cầu thứ tự trả về của top-k giống nhau, nhưng chọn khác phần tử sẽ thất bại.
- Chuỗi greedy được chạy lại độc lập bằng official graph, không chỉ teacher-force token của native.
- Native projection là FP32; scalar nonlinear/cache ở Python là float64. So sánh từng phần tử với `atol=2e-5, rtol=3e-4`. Đây là kiểm chứng trong sai số, không phải đồng nhất từng bit hay mô phỏng đủ BF16/FP8 runtime thật.

## Lỗi bằng điểm đã phát hiện và xử lý

Lần đầu đối chiếu, ca 8+4 token đạt nhưng ca dài lệch ở lựa chọn attention. Engine cũ ưu tiên chỉ số nhỏ khi bằng điểm; PyTorch dùng partial-sort/nth-element với hành vi bằng điểm khác. ReLU ở indexer có thể tạo nhiều điểm bằng 0, nên khác biệt này làm thay đổi output và cache về sau.

Đã bổ sung `native/topk_cpu.cpp`, dùng thuật toán lựa chọn STL tương ứng cách CPU ATen thực hiện. Đây là code native độc lập, không gọi Torch để lấy đáp án. Test đối chiếu trực tiếp các độ dài 1–128, top-1 đến top-4, điểm duy nhất, bằng điểm và signed zero. Lệnh `parity` sử dụng selector này; `mini` cũ giữ selector ưu tiên chỉ số để tái lập kiểm chứng NumPy.

Hành vi STL bằng điểm phụ thuộc build/compiler. Kết quả chỉ được chứng minh trên MSVC và Torch đã ghim, không hứa giống mọi hệ điều hành/version. Không nới sai số hay bỏ qua mismatch để biến kết quả thất bại thành đạt.

Nguồn thuật toán: [ATen CPU TopKImpl.h](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/TopKImpl.h); graph: [Transformers được ghim](https://github.com/huggingface/transformers/blob/3f601734a3580f55484720770850966bba060e4f/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py).

## Số đo ngày 2026-09-12

**251 test đạt** qua `test-reference.bat`, bao gồm native C/CUDA, graph official, mapping mọi trọng số, kiểm chứng source, capture trạng thái và helper top-k. Setup chạy lại đã được kiểm tra không tải/cài lại khi môi trường đúng.

Ca mặc định 8/32/64 + 4 token, seed 7: tất cả đạt. Ca hybrid seed 19:

| Prompt + ID mới | Sai số logits lớn nhất | Sai số hidden lớn nhất | Attention/expert/greedy |
|---|---:|---:|---|
| 8 + 8 | 1,19e-7 | 3,42e-7 | Khớp |
| 32 + 8 | 1,49e-7 | 4,17e-7 | Khớp |
| 120 + 8 | 2,38e-7 | 3,84e-7 | Khớp |

RSS worker đỉnh khoảng 412,95 MiB, private commit khoảng 570,84 MiB trong ca biên này. Có cả runtime Torch CPU và model tham chiếu cùng worker nên không so trực tiếp với 138 MiB của phép thử NumPy trước đó. Số đo loại trừ tiến trình khác/file cache hệ điều hành; không chứng minh trần RAM vật lý tuyệt đối. GPU chỉ tính output head; chưa đo tải model lớn hay duy trì mục tiêu 60% lâu dài.

Kết quả mới nhất: `reports/parity-latest.md`/JSON; mỗi lần chạy còn có bản riêng tại `reports/parity/<run-id>/`. Sai lệch trả mã 3, lỗi môi trường/worker mã 1, ca kiểm chứng đạt mã 0. Kết quả của checkpoint thật vẫn bị chặn bởi `doctor`.

## Tiếp theo

Triển khai reader safetensors với giới hạn đọc rõ ràng, kiểm tra header/offset/dtype và scale FP8 128×128 trên fixture nhỏ do project tạo. Sau khi reader đúng mới đối chiếu tensor mapping thực và thử từng phần checkpoint. Hiện chưa cần tải 755 GB, giải phóng ổ đĩa hoặc mua phần cứng.
