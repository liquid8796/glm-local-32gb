# Current Memory Snapshot — GLM Local 32GB

Cập nhật: 2026-09-12. Baseline đang sửa: `glm-local-32gb-v0.6.0.zip`. Bản bàn giao hiện tại: **0.6.1**. Bản 0.6.0 trước đó phát triển từ ZIP 0.5.0. Snapshot này tổng hợp từ source, report người dùng và công việc đã kiểm chứng; không phải bản xuất đầy đủ mọi hội thoại trước đó.

## Mục tiêu và các quyết định giữ nguyên

Project thử xây backend đọc/tính theo ngân sách cho checkpoint FP8 được ghi trong `config/local.json`: `dealignai/GLM-5.3-CYBERSECURITY-FP8`, revision `5915c1b88f998a9c1e1a0c83688e285a08ae3ca5`. Không đổi model, revision, định dạng FP8 hay tự chuyển GGUF.

Theo tài liệu baseline, máy mục tiêu chạy Windows x64, i7-11800H, RTX 3070 Laptop 8 GiB; đây là thông tin lịch sử trong README, không phải phép đo lại ở lần này. Giữ CPU hard cap 70% theo Windows Job Object, committed-memory budget 32.000.000.000 byte và GPU pacing target 0,6 trong cửa sổ 10 giây. Không đánh đồng commit cap với trần RAM vật lý toàn máy, hoặc target pacing với bằng chứng GPU thực giữ 60%.

Không sửa `config/local.json`, `config/reference-lock.json`, `requirements-reference.lock.txt`, graph miniature, kernel C/PTX hay source tham chiếu chính thức. Không tải checkpoint; không có inference full model hoặc lệnh chat.

## Trạng thái đã có trước khi tiếp tục

Baseline có decoder giả lập hai layer, fixture FP8 private 9.440 byte, NumPy oracle độc lập, official Transformers parity, primitive C/CUDA, Windows Job Object, GPU pacing, reader safetensors đơn file và adapter FP8 block. Các kết quả Windows lịch sử được ghi trong README/docs; report chạy thực trên máy đó không nằm trong ZIP đầu vào.

Điểm đang dở được ghi rõ cuối `docs/SAFETENSORS.md`: nối decoder thu nhỏ với safetensors chia nhiều shard và tiếp tục đối chiếu với oracle.

## Đã thực hiện trong 0.6.0

Thêm index reader nhiều shard với LRU, lazy payload reads, giới hạn số file mở, kiểm tra lại shard sau eviction và kiểm tra index/header. Thêm fixture bốn shard do serializer safetensors chính thức tạo: 80 tensor, 34 cặp weight/scale khác shard, 10.072 byte payload. Có adapter `MiniSafetensorWeights` cho decoder; oracle vẫn đọc fixture private gốc.

Thêm `--storage private|safetensors` vào `mini` và `parity`; mặc định vẫn `private`. Có `mini-sharded.bat`, `parity-sharded.bat`, báo cáo storage và test mới. Không đổi quota, checkpoint, revision lock hoặc điều kiện `doctor BLOCKED`.

## Bằng chứng lịch sử 0.6.0 và phạm vi

Linux unit suite: 347 test được phát hiện; **321 đạt, 26 bỏ qua**. Log ở `docs/verification/unit-tests-linux.txt`. Đã đối chiếu từng byte/dtype/shape của 80 tensor với `safe_open` chính thức.

Kernel C CPU nguyên trạng đã build riêng bằng GCC và thử năm ca: seed 7 với 8+4, 32+4, 64+4, 120+8; seed 19 với 64+4. Native private/safetensors khớp chính xác logits, 12 hidden states/token và trace; khớp NumPy trong tolerance cùng ID greedy độc lập. Sai số logits lớn nhất khoảng 1,1466e-7. Report: `docs/verification/sharded-linux-cpu.json`.

Chưa chạy lại Windows/MSVC/CUDA/Transformers đúng revision tại đây. Không coi test bỏ qua, test double hoặc C/GCC Linux là bằng chứng Windows/hybrid chính thức đã đạt.

## Report Windows mới và bản sửa 0.6.1

Người dùng gửi `reports(1).zip`: chỉ có `parity-latest.json` và Markdown, không có log unit test hoặc các thư mục report per-run. JSON gốc được lưu nguyên trạng trong `docs/verification/user-parity-error-v0.6.0.json`.

Report: `status=ERROR`, `child_exit_code=1`, lỗi `TypeError: argument 'tensor_dict': 'dict' object is not an instance of 'TensorSpec'`. `job_policy_verified=true`, CPU hard cap 70%, committed-memory limit 32.000.000.000 byte, kill-on-close=true. Các cờ `synthetic_official_parity_verified`, `inference_verified`, `full_model_loaded` đều false. Không có cases hoặc version thư viện trong report; không suy diễn đã đạt CUDA/official parity.

Source 0.6.0 gọi serializer bằng raw dictionaries; requirements lock ghim safetensors 0.8.0, trong khi lần kiểm chứng Linux trước dùng 0.7.0. Đối chiếu source chính thức tag v0.8.0 xác nhận serializer yêu cầu TensorSpec và caller giữ buffer sống. Vị trí exporter nằm trước khởi tạo backend native/hybrid trong execute_parity, nên sửa exporter trước, chưa chuyển bước metadata checkpoint.

Bản 0.6.1:

- Thêm `safetensor_serializer.py`: dùng TensorSpec nếu thư viện cung cấp, raw dictionary nếu không; giữ ctypes buffers tới khi serializer trả bytes, không retry TypeError, không đổi FP8 bytes, giới hạn payload helper 64 KiB mỗi lần.
- `mini_safetensors.py` dùng helper và ghi `serializer_api` vào report fixture. Không đổi số shard/tensor, mapping/graph/kernel/reader, quota hoặc dependency/revision locks.
- Report parity giữ tham số yêu cầu, version tool/worker và traceback lỗi giới hạn 16.000 ký tự, cả JSON và Markdown latest/per-run. Không đổi điều kiện PASS hoặc các cờ xác minh.
- Version project và `glm_local.__version__` cùng **0.6.1**. Project này là Python/C, không có .NET AssemblyVersion; không tăng native ABI vì giao diện kernel không đổi.
- README, BACKEND và SHARDED-DECODER đã cập nhật. Commit đề xuất: `fix(storage): support TensorSpec serialization and retain parity diagnostics`.

## Bằng chứng 0.6.1 và giới hạn còn lại

Linux unit suite: **366 test được phát hiện, 340 đạt, 26 bỏ qua**; 19 test mới. Log `docs/verification/unit-tests-linux-v0.6.1.txt`. Safetensors thực chạy tại đây là **0.7.0**, không có Transformers. Test round-trip và `safe_open` đối chiếu bytes/dtype/shape tiếp tục đạt; test mới kiểm tra buffer lifetime khi GC, empty/scalar, dữ liệu sai và việc giữ traceback/report.

`docs/verification/serializer-api-regression-v0.6.1.json` ghi phép kiểm tra tái hiện lỗi v0.6.0 và bản sửa qua **mô phỏng hợp đồng TensorSpec 0.8**, sử dụng thư viện thật hiện có để tạo/đọc bytes. Kết quả này **không phải** thực thi Rust extension safetensors 0.8.0. Không tải được wheel 0.8.0 vì môi trường không truy cập được máy chủ gói. Chưa chạy lại Windows/MSVC/CUDA hoặc Transformers đúng revision. Không gộp các test bỏ qua/historical report/test double thành nghiệm thu trên máy đích.

## Việc cần tiếp tục ngay

Áp dụng source **0.6.1** lên project hiện có, giữ `.venv-reference`, `build` và report cũ. Không cần downgrade safetensors hoặc build lại DLL cho bản sửa Python này.

Kiểm tra serializer riêng trong môi trường đã ghim, không chạy GPU:

```powershell
.\.venv-reference\Scripts\python.exe -m unittest discover -s tests -p "test_safetensor_serializer.py" -v
```

Nếu đạt, chạy lần lượt; dừng khi lệnh báo lỗi:

```powershell
.\test-reference.bat 2>&1 | Tee-Object -FilePath .\reports\test-reference-v0.6.1.log
.\glm.bat parity --storage safetensors --backend hybrid
.\glm.bat parity --storage safetensors --backend hybrid --lengths 120 --generate 8
```

Xem `tool_version=0.6.1`, `parameters.storage`, `storage.exported_fixture.serializer_api=TensorSpec`, per-case errors/selection/greedy IDs và pacing. Nếu lỗi, sửa trên baseline 0.6.1 này, không bỏ lock hoặc đổi reader để ép đạt. Cần ZIP cả cây `reports/` gồm log test và per-run vì latest chỉ giữ lần cuối.

Chỉ sau khi official hybrid thật trên máy đích đạt mới tiếp tục metadata tensor/dtype/scale checkpoint revision đã ghim rồi projection nhiều block. Mapping hiện tại synthetic-only, chưa kết luận tương thích checkpoint đầy đủ; không tải checkpoint lớn chỉ để kiểm tra metadata.

Tài liệu chính: `docs/SHARDED-DECODER.md`, mục bản sửa 0.6.1.
