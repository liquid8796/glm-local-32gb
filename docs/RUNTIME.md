# Runtime CYBERSECURITY FP8 — 0.9.0

Tài liệu này giữ mốc FP8 lịch sử. Từ0.10.0 model mặc định là NVFP4; thêm `--profile fp8` trước tên lệnh trong các ví dụ nếu muốn chạy model cũ. Hướng dẫn model hiện tại ở [NVFP4.md](NVFP4.md).

Giữ checkpoint FP8 gốc, config/revision/source tham chiếu đã ghim. `glm.bat` dùng `.venv-reference` nếu có, rồi fallback Python hệ thống. Tạo môi trường bằng `setup-reference.bat` nếu cần; không cần build lại DLL cho bản này.

## Kiểm tra nhỏ

```powershell
.\glm.bat metadata-check
.\glm.bat architecture-check
.\glm.bat runtime-plan --backend hybrid --context 4096 --generate 32
.\glm.bat projection-check --online --backend hybrid --budget-mib 32
.\glm.bat tokenizer-check --online
.\test-reference.bat
```

Metadata chỉ lấy header/JSON. `projection-check --online` lấy riêng weight/scale, mặc định q_a layer 0 (~12,59 MB); budget ví dụ32 MiB, tối đa128 MiB. Nó xác minh lại header từ xa rồi lấy các đoạn payload HTTP206 vào file slice trong thư mục run. Không tạo shard giả hàng GB. CPU/GPU được đối chiếu với tham chiếu độc lập đọc từng hàng, giải mã và tính FP32. Hash slice không thay thế hash cả shard. Bỏ `--online` để dùng shard cục bộ.

`tokenizer-check --online` chỉ lấy tokenizer.json và tokenizer_config.json (~20,22 MB), kiểm tra SHA-256 LFS/Git blob của manifest đã ghim. Không ghi đè file có hash sai, không tải trọng số hay thực thi remote Python. Bỏ `--online` để kiểm tra file cục bộ. Native tokenizer giữ postprocessor trong JSON; prompt văn bản thuần không tự áp dụng chat template.

`runtime-plan` cần architecture PASS, tính cache MLA latent và DSA full/shared, activation/scratch, tile buffers và headroom. Hybrid đọc VRAM còn trống; có thể giảm bằng `--vram-budget-mib`. Context/ngân sách sai bị từ chối trước cache/GPU allocation. `ESTIMATE_FITS` là ước tính, không phải đo full model.

## Generation khi đã có dữ liệu

Thư mục cần config.json, model.safetensors.index.json và đủ282 shard đúng revision. Config/index phải khớp byte đã capture trong metadata evidence; có thể copy hai file đó từ thư mục run đã xác minh. Dùng prompt còn cần hai file tokenizer đã kiểm tra.

```powershell
.\glm.bat generate --model-directory "E:\Models\GLM-5.3-CYBERSECURITY-FP8" --tokens 1,2,3 --context 128 --generate 4 --backend hybrid
.\glm.bat generate --model-directory "E:\Models\GLM-5.3-CYBERSECURITY-FP8" --prompt "Xin chào" --context 128 --generate 4 --backend hybrid
```

E: là ví dụ; công cụ không tự tải full checkpoint ở đó. Lượt này người dùng chọn chỉ hoàn thiện mã và kiểm tra nhỏ. Chưa có downloader toàn model.

Reader xác minh catalogue/index/header và file cục bộ trước payload; ≤64 KiB/read và ≤2 shard mở bằng LRU. FP8/F32 scale tính block128×128; BF16/F16/F32 đọc tile và tính FP32 trên CPU. Hybrid luân phiên hàng block FP8 giữa CPU/GPU; projection chỉ có một hàng block dùng CPU. GPU luôn cần telemetry gate; mỗi projection có context và ngân sách launch hữu hạn. Probe cũ vẫn mặc định256 launch.

Decoder dùng config thực, cache FP32 KV latent/RoPE, key DSA ở layer full và dùng lại index ở layer shared. Expert xử lý tuần tự. Lease được giữ suốt đời cache và trong phép tính tạm; accounting không đo mọi overhead Python/driver/file cache.

## Báo cáo và giới hạn

Mỗi lệnh có `reports/<action>/<run-id>/request.json`, `result.json`, `result.md` và `reports/<action>-latest.json`/Markdown. Action là `plan`, `projection`, `tokenizer`, `generate`. Worker Windows được gắn Job CPU70%/commit32GB trước resume; timeout/cleanup giữ cả cây tiến trình. `generate --timeout` mặc định1800 giây, giới hạn1..86400.

Exit0 tương ứng `PASS`, `ESTIMATE_FITS` hoặc `GENERATED_UNVERIFIED`; exit1 lỗi; projection numerical mismatch exit2; ngắt exit130. Đọc cả status/scope. `GENERATED_UNVERIFIED` có output nhưng không chứng nhận full-model parity/tài nguyên.

Runtime dùng FP32 cache/linear và scalar reductions, chưa mô phỏng mọi dtype/kernel BF16 chính thức. Tie mặc định chọn ID nhỏ nhất. Chỉ hỗ trợ RoPE mặc định interleaved và MoE sigmoid/noaux_tc; config ngoài profile báo lỗi. MTP được kiểm tra inventory nhưng không chạy speculative decoding. Tái dựng K/V theo vị trí có thể rất chậm; chưa có benchmark full checkpoint.

Chưa chứng minh full-model resident RAM≤32GB, GPU trung bình60% hoặc output parity. Job giới hạn committed memory, không phải RAM vật lý toàn máy; GPU gate là heuristic. `doctor` giữ BLOCKED để phản ánh mốc nghiệm thu còn chờ dữ liệu.
