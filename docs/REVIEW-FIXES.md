# Đóng review — bản 0.8.1

Các thay đổi này xử lý review tại HEAD `dac77d7`, với baseline v0.5.0 `dfb467f`. Không coi kết quả kiểm chứng miniature hoặc metadata là bằng chứng chạy model đầy đủ.

## Đã sửa

- `architecture-check` đọc đúng `tensor-catalogue.jsonl` do metadata workflow xuất ra, kiểm tra kích thước, SHA-256, identity/revision, số record, số shard và nguồn report.
- Bỏ cách xác minh chỉ dựa trên tên có layer. Profile config phải được hỗ trợ, đủ tensor từng layer/expert, shape/dtype/nbytes và cặp scale đúng. Unknown/unsupported/partial cần review.
- Phân biệt đúng dense gate projection, expert projection, router, router bias và scale. Không tính scale thành expert/router đã xác minh.
- Khi đọc đủ header, payload phải bằng chính xác `index.total_size`. Sai lệch trả `ERROR`; chênh lệch và trạng thái accounting được giữ trong JSON/Markdown. Kiểm tra một phần trì hoãn equality nhưng không cho vượt tổng khai báo.
- Chặn offset tensor chồng/hở trong từng shard, catalogue bị sửa và report nguồn bị thay trong lúc phân tích.
- Đồng bộ version code/package/docs ở 0.8.1. Khôi phục link bằng chứng lịch sử v0.7.0 và dependency safetensors 0.7.0 của lần chạy cũ; không đổi lịch sử thành kết quả chạy mới.
- Bổ sung regression test về CLI, producer → consumer, false PASS, provenance, config, inventory, role mapping và release consistency.

## Kiểm chứng thực

`test-reference.bat` chạy **513 test, tất cả đạt**, bật official Transformers, native C/CUDA và storage tests. Hai test hỏng khi review trước đã được sửa nguyên nhân. [Log đầy đủ](verification/review-fixes-tests-windows-v0.8.1.txt).

Kiểm chứng metadata bằng **header giả lập**: catalogue đủ 64 record đi từ `execute_metadata` → `publish_report` → `run_architecture` và đạt metadata PASS; chỉ tăng `total_size` 1 byte thì cả metadata và architecture trả ERROR. Không truy cập checkpoint từ xa trong bài thử này.

Official parity trên CPU/GPU thật với storage safetensors 4 shard, seed 19:

| Prompt + ID mới | Kết quả | Sai số logits tối đa | Sai số hidden tối đa |
|---|---|---:|---:|
| 8 + 8 | PASS | 1,19e-7 | 3,42e-7 |
| 32 + 8 | PASS | 1,49e-7 | 4,17e-7 |
| 120 + 8 | PASS | 2,38e-7 | 3,84e-7 |

Greedy IDs đều khớp, tối đa **2 shard mở đồng thời**. [Bằng chứng tổng hợp](verification/review-fixes-windows-v0.8.1.json).

Các cờ `real_checkpoint_compatible`, `inference_verified` và giới hạn tài nguyên full model vẫn false trong architecture report. Việc đủ cấu trúc metadata không chứng minh payload, tokenizer, dtype runtime, thực thi hoặc mức RAM/GPU của model thật.
