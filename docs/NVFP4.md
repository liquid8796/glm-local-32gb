# GLM NVFP4 — 0.11.1

Đã thêm `dealignai/GLM-5.3-ABLITERATED-NVFP4` và đặt làm profile mặc định. Revision cố định: `371bdb985d0124e76348c91e4a8fcf3a9d719d09`. Profile FP8 trước đây được giữ tại `config/models/cybersecurity-fp8.json`; NVFP4 tại `config/models/abliterated-nvfp4.json`.

Core 0.11.0 thêm chat template đã kiểm chứng, streaming assistant/reasoning và prefill theo lô. Native CPU xử lý song song theo hàng và SIMD giữa các vector độc lập, giữ thứ tự FP32 của từng kết quả. Reader dùng handle bảo vệ trên Windows để tránh kiểm tra lại filesystem trên mỗi lần đọc khi tệp đang được khóa chống thay đổi. Bộ đệm dải trọng số tối đa 8 MiB, các lần I/O không quá 64 KiB và các cache đều được hạch toán. Xem [hướng dẫn hội thoại](CONVERSATION.md).

NVFP4 trên CPU gom tối đa 128 hàng × 16.384 cột logic vào một lần gọi native; bên trong vẫn tính subtotal 128 cột và cộng FP32 đúng thứ tự cũ. Tối đa 16 vector đầu vào chia sẻ dải trọng số. Scratch NVFP4 5 MiB, dense 33 MiB và workspace prefill 128 MiB được cộng vào planner. Hybrid cũng dùng dải native CPU cho dense; đường expert CUDA vẫn dùng kernel tile. Mặc định chạy CPU; mức tăng tốc một projection không xác nhận tốc độ toàn model.

## Chạy trên máy này

```powershell
.\build-native.bat
.\glm.bat --profile nvfp4 metadata-check --budget-mib 96
.\glm.bat --profile nvfp4 architecture-check
.\glm.bat --profile nvfp4 runtime-plan --backend hybrid --context 4096 --generate 32
.\glm.bat --profile nvfp4 projection-check --online --backend hybrid --budget-mib 16
.\glm.bat --profile nvfp4 tokenizer-check
.\test-reference.bat
```

`--profile` đứng trước tên lệnh; không kết hợp với `--config`. Không truyền profile/config thì dùng `config/local.json`, hiện trỏ NVFP4. Chọn lại FP8 bằng `--profile fp8` cho cùng các lệnh. Metadata, architecture, tokenizer, generation và readiness NVFP4 nằm trong `reports/nvfp4/`; FP8 vẫn dùng `reports/` cũ. Các lệnh fixture synthetic không chứng minh full model.

Build bổ sung `build/nvfp4_cpu.dll`. CUDA nạp `native/nvfp4_matvec.ptx` qua driver; không cần CUDA Toolkit hoặc GPU có FP4 tensor core. Mặc định `projection-check` chọn `model.layers.3.mlp.experts.0.gate_proj.weight`, logical 2048×6144, packed 2048×3072. Chỉ tải bốn tensor liên quan trong budget; không tải toàn shard hoặc full model. Có thể chỉ định projection khác bằng `--tensor`.

Tokenizer gồm hai JSON có hash giống profile FP8. Lần này đã copy sau khi đối chiếu manifest NVFP4 và kiểm tra lại, không tải thêm. Trên máy khác dùng `tokenizer-check --online` để tải riêng hai JSON (~20,22 MB), không remote code.

## Format và phạm vi tính toán

| Tensor expert | Dtype và shape lưu | Vai trò |
|---|---|---|
| `*.weight` | U8[N,K/2] | Hai giá trị E2M1/byte; low nibble là cột chẵn |
| `*.weight_scale` | F8_E4M3[N,K/16] | Scale riêng cho mỗi hàng và nhóm 16 cột |
| `*.weight_scale_2` | F32 scalar[] | Scale chung của weight |
| `*.input_scale` | F32 scalar[] | Calibration activation, không phải scale của weight |

Giải mã weight bằng `E2M1 × (E4M3_block_scale × F32_global_scale)` theo thứ tự làm tròn FP32. Scale lưu trong checkpoint là row-major chưa swizzle. Reader giữ logical shape riêng với physical shape; không áp dụng phép kiểm tra FP8 shape trực tiếp cho U8packed.

Chỉ routed experts thuộc các layer MoE của 78 layer backbone được lượng tử 4-bit. Attention, shared expert, ba layer dense đầu và layer MTP giữ BF16. Metadata bắt buộc đủ inventory, dtype, shape, ignore coverage và producer ModelOpt 0.45.0; config lạ trả REVIEW_REQUIRED.

Runtime hiện là **giải mã weight NVFP4 rồi tính FP32**, với `activation_quantization="none"`. Nó xác minh input_scale hữu hạn/dương nhưng không dùng nó để scale weight; không mô phỏng W4A4 activation hoặc FP8 KV-cache của NVIDIA/vLLM. Không dùng kết quả fallback để tuyên bố nativeW4A4parity.

Mỗi tile tối đa 128×128 phần tử: 8.192 byte weight, 1.024 byte scale; mỗi lần đọc ≤64 KiB và tối đa 2 shard mở, thêm hai handle bảo vệ config/index trên Windows. GPU luôn qua telemetry gate và ngân sách launch. Reader không giữ cả matrix/layer/expert bank đã giải mã. Decoder dùng FP32 latent MLA/DSA và cache K/V mở rộng có giới hạn 256 token mỗi layer. Chạy lại `runtime-plan` để lấy ước lượng bao gồm các cache mới; ước lượng không phải phép đo full model.

## Kiểm chứng và phần chưa nghiệm thu

Đã kiểm tra đủ 282 header, 232.385 tensor và 57.600 bộ NVFP4; metadata/architecture PASS, zero findings. Tổng file shard 464.822.689.680 byte (~464,82 GB), payload 464.795.267.072 byte. Lần đọc mạng timeout tại shard 280; `metadata-check --resume <evidence>` xác minh lại 279 header đã lưu và chỉ lấy tiếp 3 header. Evidence cũ được giữ nguyên.

Kernel được kiểm tra với đủ 256 giá trị byte đóng gói, 127 scale E4M3 hữu hạn không âm, zero scale, cạnh tile, subnormal, overflow và cleanup. Decoder nhỏ qua safetensors thật/CPU/CUDA được so với graph Transformers có trọng số giải mã độc lập; sai số logits tối đa khoảng 1,39e-7.

Projection thật logical 2048×6144 đạt sai số 0; CPU/GPU xử lý 384 tile mỗi bên. Tổng payload mẫu tải về là 7.077.896 byte; không giữ matrix đã giải mã. Bộ hồi quy Windows cuối đạt **728/728 test**, không skip. [Bằng chứng bản này](verification/nvfp4-v0.10.0.json).

`generate` cần config/index và đủ 282 shard cục bộ đúng revision; nó không tải model ngầm. Thư mục người dùng đã tải đủ được kiểm tra tên/dung lượng vào 2026-09-13, và tokenizer tại đó PASS; kiểm tra này không xác minh lại hash toàn bộ payload hoặc chứng minh full-model inference. Ví dụ khi đã có dữ liệu:

```powershell
.\glm.bat --profile nvfp4 generate --model-directory "E:\Models\GLM-5.3-ABLITERATED-NVFP4" --prompt "Hi" --context 128 --generate 1 --backend cpu --timeout 1800
```

E: là đường dẫn ví dụ; không có thao tác tải full model ở đó. Full-model output/BF16-W4A4 parity, resident RAM ≤32 GB, GPU trung bình 60% và throughput chưa nghiệm thu; doctor vẫn BLOCKED. Job Object giới hạn commit và pacing là heuristic. MTP có kiểm tra cấu trúc nhưng không thực hiện speculative decoding.

Đây là lệnh chẩn đoán, chưa phải cấu hình chat: trên i7-11800H, core 0.10.1 xử lý prompt `Hi` và sinh một token EOS trong khoảng **17 phút 14 giây**, không có câu trả lời hiển thị. Mức 300 giây hết hạn ở layer 21/78. Xem [bằng chứng timeout và giới hạn runtime](verification/runtime-timeout-v0.10.1.md) trước khi tăng số token hoặc triển khai API.

Nguồn format: [NVIDIA ModelOpt NVFP4 tại revision đã đối chiếu](https://github.com/NVIDIA/Model-Optimizer/blob/51de53e48ccae8804f8fe1198b7cf89475c5c4f4/modelopt/torch/quantization/qtensor/nvfp4_tensor.py), [vLLM swizzle sau khi load](https://github.com/vllm-project/vllm/blob/d86257e2833e0c3add2c694a380435c451c1b176/vllm/model_executor/kernels/linear/nvfp4/cutlass.py). Các nguồn này mô tả format/toán học, không chứng minh checkpoint đầy đủ đã chạy đúng trên máy này.
