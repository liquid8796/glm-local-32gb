# Phần suy luận còn thiếu

Đây là ghi nhận khoảng cách kỹ thuật và điều kiện nghiệm thu cho GLM. Bản 0.5.0 bổ sung reader safetensors và FP8 block scale đã kiểm chứng bằng file thử, bên cạnh decoder thu nhỏ đối chiếu NumPy/Transformers; mục tiêu suy luận checkpoint đầy đủ **chưa hoàn tất**.

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

1. **Backend toán học:** cấu hình giả lập đã đối chiếu với Transformers nguyên trạng trên CPU FP32, gồm 12 trạng thái trung gian/token và lựa chọn attention/expert. Sai số được kiểm tra theo từng phần tử. Cần xác minh dtype/scale/tensor mapping/tokenizer ở checkpoint thật; kết quả nhỏ chưa bảo đảm tương thích cấu hình 78 layer.
2. **Đọc trọng số theo ngân sách:** đã có reader safetensors với giới hạn header/read, kiểm tra dtype/offset và adapter FP8 2D + scale F32 128×128. Kiểm chứng aligned blocks với bộ giải mã HF, ragged blocks với phép mở rộng scale độc lập. Cần nối reader này vào decoder nhiều tensor/shard, xác minh tên/shape/scale thực và mô hình cấp phát activation/cache ở kích thước thật. Chưa chứng minh trần RAM vật lý toàn model.
3. **Kết hợp CPU/GPU:** phép thử ma trận chia nhóm hàng; decoder thu nhỏ cho CPU tính các projection nội bộ và GPU tính output head. Có pacing trước mỗi lần gửi GPU. Chưa có scheduler tối ưu cho GLM hoặc benchmark đủ dài để kiểm chứng mục tiêu sử dụng GPU dưới tải model.
4. **Kiểm chứng full checkpoint:** đủ 282 shard, kiểm tra nội dung/revision, sinh token trên model đầy đủ, so sánh với tham chiếu và đo RAM/CPU/GPU trong prefill/decode nhiều độ dài. Kiểm tra kích thước file chỉ là bước ban đầu, không chứng minh trọng số đúng hay inference đúng.

Trước khi có các bằng chứng trên, không dùng kết quả unit test, policy readback hay quan sát GPU lúc nhàn rỗi làm bằng chứng model đã chạy trong giới hạn yêu cầu.

## Runtime tham khảo đã khảo sát

- [llama.cpp GLM converter](https://github.com/ggml-org/llama.cpp/blob/master/conversion/glm.py) và [GLM DSA graph](https://github.com/ggml-org/llama.cpp/blob/master/src/models/glm-dsa.cpp) có hỗ trợ kiến trúc liên quan, nhưng đường đó cần GGUF; project này giữ yêu cầu checkpoint FP8 gốc.
- [Transformers GLM](https://huggingface.co/docs/transformers/main/en/model_doc/glm_moe_dsa) cùng [FP8 quantizer](https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/quantizer_finegrained_fp8.py) không chứng minh khả năng chạy checkpoint này trong 32 GB. Với GPU capability dưới 8.9, phải xét đường giải mã sang kiểu tính khác.
- [Accelerate disk offload](https://huggingface.co/docs/accelerate/main/en/usage_guides/big_modeling) là cơ chế tổng quát; không tự chứng minh cách chỉ nạp expert được chọn với bộ nhớ có trần.
- [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/) phân biệt CPU weight offload và VRAM budget; `gpu-memory-utilization` không phải GPU compute utilization.

Các đường dẫn nhánh `main`/`master` và tài liệu runtime có thể thay đổi. Ghi nhận khảo sát ngày 2026-09-11. Source Kimi và revision checkpoint được ghim riêng trong project.
