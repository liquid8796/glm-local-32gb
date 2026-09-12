# Current Memory Snapshot — GLM Local 32GB

Cập nhật: 2026-09-12. Baseline: **0.6.1**, đã được người dùng kiểm chứng trên Windows bằng `reports(2).zip`. Bản bàn giao hiện tại: **0.7.0**. Snapshot này ghi lại trạng thái dự án và phạm vi bằng chứng, không phải bản xuất đầy đủ mọi hội thoại.

## Mục tiêu và quyết định giữ nguyên

Checkpoint trong `config/local.json`: `dealignai/GLM-5.3-CYBERSECURITY-FP8`, revision `5915c1b88f998a9c1e1a0c83688e285a08ae3ca5`. Không đổi model, revision, định dạng FP8 hoặc chuyển GGUF để vượt kiểm thử. Không tải payload trọng số checkpoint trong bước metadata.

Máy mục tiêu theo tài liệu và report người dùng: Windows x64, i7-11800H, NVIDIA GeForce RTX 3070 Laptop GPU 8 GiB. CPU hard cap 70% bằng Windows Job Object; committed-memory limit 32.000.000.000 byte; GPU pacing target 0,6 trong cửa sổ 10 giây. Committed memory không phải RAM vật lý toàn máy; pacing target không phải bằng chứng đã giữ trần GPU 60%.

Giữ nguyên `config/local.json`, `config/reference-lock.json`, `requirements-reference.lock.txt`, kernel C/PTX, graph miniature, oracle chính thức, adapter FP8 và reader nhiều shard. `doctor` vẫn BLOCKED; chưa có inference checkpoint đầy đủ hoặc lệnh chat. Version Python package và `glm_local.__version__` cùng **0.7.0**. Project Python/C không có .NET AssemblyVersion; native ABI không đổi.

## Mốc 0.6.1 đã đóng: Windows miniature hybrid parity

Các thay đổi 0.6.0 nối decoder hai layer vào fixture safetensors bốn shard. Bản 0.6.1 sửa exporter để dùng TensorSpec trên safetensors 0.8.0 và giữ buffer tồn tại khi serialize, đồng thời bổ sung chẩn đoán report.

Report Windows `reports(2).zip` đã được review ở lượt trước:

- `test-reference-v0.6.1.log`: **376 tests, OK**, không bỏ qua.
- Bốn ca prompt + sinh thêm **8+4, 32+4, 64+4, 120+8** đều PASS; hidden states/logits trong tolerance, attention/expert selection và greedy token IDs khớp oracle Transformers độc lập.
- Safetensors **0.8.0 thật**, `serializer_api=TensorSpec`; 80 tensor/four shards, 34 cặp weight/scale khác shard; tối đa hai shard mở đồng thời.
- Hybrid thực dùng RTX 3070 Laptop nhưng **chỉ output head lên GPU**, không phải toàn bộ decoder. Policy Windows được xác nhận cài; full-model/inference/GPU-cap flags vẫn false.

Đây là bằng chứng lịch sử từ máy người dùng, **không phải chạy lại Windows cho 0.7.0**. Bản ghi nghiệm thu được giữ trong `docs/verification/windows-acceptance-v0.6.1.md`. Không quay lại sửa serializer/reader từ đầu hoặc bắt chạy lại ba lệnh cũ để đóng mốc này.

## Đã thực hiện trong 0.7.0

Thêm `metadata-check` để kiểm tra config, index và header tensor **đúng revision ghim**, không tải toàn bộ shard hay đọc giá trị trọng số:

- `checkpoint_http.py`: client standard library; chỉ HTTPS Hugging Face/CDN, redirect bị giới hạn và không đọc body redirect; hai Range request cho prefix 8 byte và JSON header mỗi shard. Chặn HTTP 200 thay vì fallback tải cả shard; kiểm tra 206, Content-Range/Length, encoding, kích thước file và ETag khi có. Không đọc token/cache thông tin đăng nhập.
- `checkpoint_schema.py`: kiểm tra manifest/config/index/header, dtype/shape/offset, cặp FP8 `_scale_inv`, grid 128x128; catalogue giữ mọi tensor quan sát được và báo riêng tên chưa nhận diện, scale BF16 chưa được adapter hỗ trợ, tensor ngoài mapping/shape đã biết. Không mặc nhiên gán full architecture mapping hoặc tương thích runtime.
- `checkpoint_snapshot.py`: lưu model/config/index và prefix+header dưới `evidence/`, kèm SHA-256/length; `--offline` kiểm tra lại dữ liệu đã lưu, không truy cập mạng. Snapshot chỉ bảo đảm nhất quán dữ liệu lưu cục bộ, không phải chữ ký xác thực checkpoint.
- `checkpoint_check.py`, `checkpoint_worker.py`, CLI và `metadata-check.bat`: báo cáo per-run + latest, status/exit code, coverage, stage/lỗi, counter, catalogue. Windows launcher cài Job Object trước khi worker chạy; không gọi GPU/Torch/Transformers cho bước metadata.
- Tách `parse_header_bytes()` từ reader đơn file để tái sử dụng đúng bộ kiểm tra; không thay đường đọc payload hoặc nới runtime reader limits.

Giới hạn mặc định: budget đọc body metadata 64 MiB (CLI tối đa 128 MiB), header 1 MiB/shard, 512 shards, 64 KiB/lần đọc; cap JSON/index/request/redirect và thời gian được ghi đầy đủ trong `docs/CHECKPOINT-METADATA.md`. Counter body là byte code ứng dụng đọc, không phải giới hạn traffic/OS/TLS buffering.

Metadata index có cap riêng 32 MiB/262.144 tensor cho việc kiểm tra; **runtime reader vẫn 1 MiB/8.192 tensor**. Report `runtime_index_policy` nêu khoảng cách đó; không tự nới runtime reader chỉ để đổi kết quả metadata.

Status: `PASS` (0) chỉ là các kiểm tra metadata đã triển khai đạt và đủ coverage; `PARTIAL` (2) chưa đủ header; `REVIEW_REQUIRED` (3) cần đối chiếu baseline/tensor profile; `ERROR` (1) lỗi giao thức/dữ liệu/quota; `INTERRUPTED` (130) bị ngắt. `architecture_mapping_verified`, `real_checkpoint_compatible`, `inference_verified`, `full_model_loaded`, `payload_values_verified` vẫn **false**, kể cả metadata PASS.

## Kiểm thử và giới hạn bằng chứng 0.7.0

Linux unit suite: **453 test được phát hiện, 427 đạt, 26 bỏ qua**, gồm **87 test mới** cho HTTP/header/schema/snapshot/orchestration. Log: `docs/verification/unit-tests-linux-v0.7.0.txt`. Safetensors môi trường này là 0.7.0, không có Transformers; không sửa dependency lock theo môi trường này.

Chạy riêng 87 test metadata với `python -S` (không nạp site-packages): **87 đạt, không bỏ qua**; log `docs/verification/metadata-stdlib-tests-linux-v0.7.0.txt`.

Test mới kiểm tra Range bị bỏ qua, response/body/budget sai, revision sai, shard/index mismatch, cross-shard scale, BF16 scale, unknown mapping, snapshot đổi dữ liệu/liên kết, offline replay và report lỗi. End-to-end HTTP sử dụng phản hồi giả lập; kiểm tra launcher Windows dùng test double. **Chưa chạy mới Windows/MSVC/CUDA/Transformers đúng revision cho 0.7.0.** Test bỏ qua không tính là đạt.

Đã thử thật `python -m glm_local metadata-check --max-shards 1` tại đây: **ERROR ở `fetch_model` do không phân giải DNS Hugging Face**, 1 request thử, **0 byte body đã đọc, 0 Range request**. Xem `docs/verification/metadata-online-attempt-linux-v0.7.0.json` và `.txt`. Không thu được config/index/header checkpoint thật; không dùng model card/main branch hoặc fixture giả lập để điền thay bằng chứng.

`docs/verification/metadata-release-invariants-v0.7.0.json` ghi các hash file cấu hình/lock/kernel/graph/oracle/reader/adapter được giữ nguyên so với baseline 0.6.1.

## Việc cần tiếp tục ngay trên máy người dùng

Áp dụng source **0.7.0**, giữ `.venv-reference`, `build` và report cũ. Không cần build lại DLL, cài thêm thư viện hoặc hạ safetensors cho tính năng metadata mới.

```powershell
.\test-reference.bat 2>&1 | Tee-Object -FilePath .\reports\test-reference-v0.7.0.log
.\glm.bat metadata-check
```

Dừng nếu unit test báo lỗi. `metadata-check` mặc định kiểm tra tất cả shard trong giới hạn; `--max-shards 3` chỉ là kiểm tra một phần, không thay cho full coverage. Cần Internet truy cập model công khai đã ghim; không đổi pin hoặc fallback tải payload khi lỗi.

Gửi ZIP log test và **toàn bộ** `reports/metadata/<run-id>/` gồm `result.json`, `result.md`, `tensor-catalogue.jsonl` và `evidence/`. Thư mục evidence cho phép đọc lại offline theo hướng dẫn trong `docs/CHECKPOINT-METADATA.md`; replay không phải tiếp tục tải online sau lỗi.

## Bước sau khi có metadata thật

Đọc coverage/status/findings/catalogue trước. Xác nhận cấu trúc thật, các tên/shape chưa mapping và giới hạn runtime/scale dtype. Chỉ sửa profile/mapping/reader khi có metadata làm căn cứ; không đổi baseline snapshot để ép PASS. Tiếp đó triển khai projection FP8 nhiều block và oracle độc lập, rồi mở rộng theo layer/expert dưới ngân sách. Chưa chuyển sang tải full checkpoint hay làm giao diện chat.

Commit đề xuất:

```text
feat(metadata): add bounded checkpoint header audit and offline replay
```

Tài liệu chính: `docs/CHECKPOINT-METADATA.md`; lịch sử: `docs/SHARDED-DECODER.md`, `docs/SAFETENSORS.md`, `docs/BACKEND.md`, `docs/OFFICIAL-PARITY.md`.
