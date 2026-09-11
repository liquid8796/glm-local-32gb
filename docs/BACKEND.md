# Phần suy luận còn thiếu

Đây là ghi nhận khoảng cách kỹ thuật và điều kiện nghiệm thu, không phải backend có thể chạy. Mục tiêu cuối cùng của người dùng **chưa hoàn tất**.

## Khác biệt đã kiểm tra

| Thành phần | Kimi upstream | Checkpoint GLM đích |
|---|---|---|
| Attention | KDA kết hợp gated MLA | MLA/DSA với indexer và chia sẻ index |
| Layers | 93 | 78 |
| MoE | Kiến trúc latent riêng của Kimi | 256 experts, chọn 8, 3 layer đầu dense |
| Trọng số expert | MXFP4 | FP8 E4M3, block scale 128 × 128 |
| Residual/activation | Attention Residuals, SiTU | Residual và SiLU của GLM |
| Thiết bị | CPU | Yêu cầu phối hợp CPU và GPU |

Không thể đổi vài hằng số hoặc tensor name để biến engine Kimi thành engine GLM.

## Những phần cần triển khai trước khi gọi là chạy được

1. **Backend toán học:** graph GLM đúng config, tensor mapping, tokenizer/chat template, positional encoding, attention/indexer, expert routing và đường FP8 phù hợp phần cứng. Đối chiếu tensor và logits bằng dữ liệu thử nhỏ cùng runtime tham chiếu. Mọi thành phần này hiện chưa triển khai.
2. **Đọc trọng số theo ngân sách:** chỉ nạp phần cần tính, quản lý vòng đời buffer và cache; đo riêng commit, resident memory, memory do driver và file cache. Phải tính activation/KV/buffer tạm vào 32 GB. Có thể giữ byte FP8 trên đĩa trong khi dùng buffer tính toán có kiểu khác. Chưa có đường này trong project.
3. **Kết hợp CPU/GPU:** scheduler và backend phải trực tiếp tuân thủ ngân sách, đồng bộ buffer/device và gọi pacing ở ranh giới phù hợp. Hạn mức VRAM và giới hạn công suất không thay thế mục tiêu 60% GPU compute. Chưa có CUDA backend hoặc phép đo dưới tải.
4. **Kiểm chứng full checkpoint:** đủ 282 shard, kiểm tra nội dung/revision, sinh token trên model đầy đủ, so sánh với tham chiếu và đo RAM/CPU/GPU trong prefill/decode nhiều độ dài. Kiểm tra kích thước file chỉ là bước ban đầu, không chứng minh trọng số đúng hay inference đúng.

Trước khi có các bằng chứng trên, không dùng kết quả unit test, policy readback hay quan sát GPU lúc nhàn rỗi làm bằng chứng model đã chạy trong giới hạn yêu cầu.

## Runtime tham khảo đã khảo sát

- [llama.cpp GLM converter](https://github.com/ggml-org/llama.cpp/blob/master/conversion/glm.py) và [GLM DSA graph](https://github.com/ggml-org/llama.cpp/blob/master/src/models/glm-dsa.cpp) có hỗ trợ kiến trúc liên quan, nhưng đường đó cần GGUF; project này giữ yêu cầu checkpoint FP8 gốc.
- [Transformers GLM](https://huggingface.co/docs/transformers/main/en/model_doc/glm_moe_dsa) cùng [FP8 quantizer](https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/quantizer_finegrained_fp8.py) không chứng minh khả năng chạy checkpoint này trong 32 GB. Với GPU capability dưới 8.9, phải xét đường giải mã sang kiểu tính khác.
- [Accelerate disk offload](https://huggingface.co/docs/accelerate/main/en/usage_guides/big_modeling) là cơ chế tổng quát; không tự chứng minh cách chỉ nạp expert được chọn với bộ nhớ có trần.
- [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/) phân biệt CPU weight offload và VRAM budget; `gpu-memory-utilization` không phải GPU compute utilization.

Các đường dẫn nhánh `main`/`master` và tài liệu runtime có thể thay đổi. Ghi nhận khảo sát ngày 2026-09-11. Source Kimi và revision checkpoint được ghim riêng trong project.
