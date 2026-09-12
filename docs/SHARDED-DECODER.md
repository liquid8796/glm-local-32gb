# Decoder thu nhỏ đọc safetensors nhiều shard — 0.6.0

Bước tiếp nối của `docs/SAFETENSORS.md`: **decoder thu nhỏ đã có đường đọc safetensors nhiều shard**, thay vì chỉ kiểm tra các ma trận rời. Chỉ dùng trọng số giả lập cố định; không tải checkpoint, không thêm lệnh chat, không tuyên bố chạy GLM đầy đủ.

## Chạy trên Windows

Giữ nguyên cấu hình, compiler và môi trường tham chiếu của bản 0.5.0. Nếu giải nén vào thư mục mới chưa có DLL, chạy `build-native.bat`. Nếu chưa có `.venv-reference`, chạy `setup-reference.bat`. Hai script này vẫn là script cũ; không thay đổi revision hoặc cài đặt lại môi trường đang có.

```powershell
.\build-native.bat
.\test-reference.bat
.\glm.bat parity --storage safetensors --backend hybrid
.\glm.bat parity --storage safetensors --backend hybrid --lengths 120 --generate 8
```

`parity` tự dùng Python trong `.venv-reference`, xác minh provenance trước khi import graph chính thức và chạy worker dưới Windows Job Object. Không thay đổi `reference-lock.json`, không bỏ qua kiểm tra revision. Double-click `parity-sharded.bat` chạy cùng đường kiểm chứng mặc định và giữ cửa sổ mở.

Đối chiếu với NumPy thay vì Transformers:

```powershell
.\mini-sharded.bat --backend cpu
```

`mini-sharded.bat` cũng dùng `.venv-reference` để có đủ thư viện. Lệnh trực tiếp sau dùng Python mặc định, vì vậy chính môi trường đó phải có NumPy và safetensors:

```powershell
.\glm.bat mini --storage safetensors --backend cpu
```

Extra `storage-validation` khai báo NumPy và safetensors cho môi trường khác. Không cần Torch để **tạo** fixture nhiều shard: exporter gọi `safetensors.serialize` trên FP8/F32 raw bytes. Lệnh `parity` và kiểm thử `safe_open` bằng Torch vẫn cần môi trường tương ứng.

Các lệnh cũ `mini`, `parity` không có `--storage` vẫn dùng `private` như trước. Không có chuyển ngầm từ safetensors sang private khi gặp lỗi. `doctor` vẫn `BLOCKED`.

## Nội dung thay đổi

| Thành phần | Trách nhiệm |
|---|---|
| `sharded_safetensors.py` | Đọc index, đối chiếu chính xác index/header, quản lý reader theo LRU, đọc payload theo yêu cầu. |
| `mini_safetensors.py` | Xuất fixture cố định bằng serializer chính thức; ánh xạ matrix/scale/vector và cung cấp giao diện trọng số cho decoder. |
| `mini_storage.py` | Chọn nguồn trọng số native cho cả `mini` và `parity`. |
| `mini_run.py`, `parity_run.py`, CLI | Nhận `--storage`, giữ nguyên graph/kernel/quota/pacing và thêm thông tin storage vào report. |
| Các test mới | Kiểm tra dữ liệu sai, tài nguyên reader, byte/dtype/shape, decoder và đường truyền tham số đến worker. |

Không sửa phép tính trong `mini_engine.py`, kernel C/PTX hay graph Transformers. `FP8BlockMatrix` được dùng lại với cùng giao diện đọc; annotation bổ sung kiểu reader nhiều shard.

## Fixture và mapping

Bốn file `model-00001-of-00004.safetensors` đến `model-00004-of-00004.safetensors`, kèm `model.safetensors.index.json`. Index có `metadata` và `weight_map`; `metadata.total_size` là tổng byte payload tensor, không phải tổng dung lượng file bao gồm header.

Có 34 matrix FP8, 34 scale F32 và 12 vector F32: tổng **80 tensor, 10.072 byte payload**. FP8 matrix payload vẫn đúng **9.440 byte** của fixture cũ; không lượng tử hóa lại. Tất cả 34 cặp weight/scale được cố ý đặt khác shard để thực sự kiểm tra đọc chéo.

Tên như `layer.0.q_a.weight`, `layer.0.q_a.weight_scale_inv` và `layer.0.in_norm.value` là **mapping giả lập cố định**. Chưa xác minh đó là tên, dtype, layout expert hay scale của checkpoint GLM thật. Không suy đoán tensor thiếu hoặc dùng scale mặc định. Mỗi matrix thu nhỏ nằm trong một block 128×128; không diễn giải kết quả này thành việc decoder đã xử lý projection kích thước model thật.

Scale vẫn được **nhân** khi giải mã. Reader từ chối scale không dương/không hữu hạn, mã FP8 NaN và vector không hữu hạn khi phần dữ liệu đó được yêu cầu.

## Ngân sách đọc và file handle

Mở reader chỉ đọc index/header và kiểm tra mapping; chưa đọc payload. Payload được lấy khi decoder gọi matrix/vector/embedding. Embedding đọc đúng một hàng 16 byte và một scale 4 byte.

Native reader không giữ cache FP8 hoặc vector đã giải mã. Tối đa hai shard mở đồng thời mặc định; có eviction và kiểm tra lại fingerprint/header khi mở lại. Bộ nhớ của metadata, object Python, tensor do caller giữ, oracle và cache hệ điều hành **không** được tính là `resident_weight_bytes`.

Reader tổng quát có giới hạn cục bộ: index 1 MiB, 8.192 tensor index, 512 shard, mỗi lần đọc tối đa 64 KiB, tối đa 1–8 reader mở tùy tham số. Adapter miniature siết thêm đúng bốn tên shard và từng file/index không quá 64 KiB. Đây là policy của project, không phải giới hạn của định dạng hay bằng chứng đã thử 512 shard/checkpoint thật.

Index chỉ tham chiếu file phẳng trong thư mục. Từ chối đường dẫn thoát thư mục, tên không phù hợp Windows, tên chỉ khác hoa/thường, symlink/reparse point, duplicate JSON key, shard thiếu, tensor thừa/thiếu và tổng payload không khớp.

Fingerprint phát hiện thay đổi thông thường; **không thay thế hash checkpoint hoặc file lock**. Report exporter ghi SHA-256 của các file thử thực sự đã tạo, không xác thực một model bên ngoài. Reader không âm thầm coi một shard đã đổi sau eviction là baseline mới.

## Oracle độc lập

Native decoder đọc fixture safetensors; NumPy và Transformers vẫn đọc **fixture private gốc** bằng loader tham chiếu cũ. Như vậy lỗi ở loader mới không được chia sẻ sang oracle. Test bổ sung dùng `safe_open` của thư viện chính thức để đối chiếu mọi byte/dtype/shape của 80 tensor với fixture gốc.

Report lưu trong thư mục run riêng, cùng các file `reports/mini-latest.json`/Markdown hoặc `reports/parity-latest.json`/Markdown như trước. Xem `parameters.storage`, `storage.reader_stats` và `storage.exported_fixture` để phân biệt đường kiểm chứng. Hai định dạng cùng dùng tên latest; bản đầy đủ từng lần vẫn nằm trong thư mục run.

## Bằng chứng lịch sử của bản 0.6.0

Môi trường: Linux, Python 3.13.5, NumPy 2.3.5, Torch 2.10.0+cpu, safetensors 0.7.0; không có Transformers. Không dùng môi trường này để thay thế lock Windows của project.

`python -m unittest discover -s tests`: **347 test được phát hiện, 321 đạt, 26 bỏ qua**. Log: `verification/unit-tests-linux.txt`. Các kiểm thử opt-in Windows/native CUDA/Transformers không được tính là đạt tại đây. Số test được phát hiện có thể khác khi bật các nhóm tùy chọn của source cũ.

Ngoài unit test, đã build **nguyên trạng kernel C** bằng GCC, tắt fast-math và fused multiply-add, rồi gọi trực tiếp bằng ctypes trên Linux. Đây là kiểm chứng C CPU riêng, không giả lập Windows Job Object. Native private và native safetensors khớp chính xác logits, 12 hidden states/token và selection trace trên các ca sau; native safetensors còn được đối chiếu độc lập với NumPy:

| Seed | Prompt + sinh thêm | Sai số logits lớn nhất so với NumPy |
|---:|---:|---:|
| 7 | 8 + 4 | 7,3470e-8 |
| 7 | 32 + 4 | 9,1722e-8 |
| 7 | 64 + 4 | 1,0117e-7 |
| 7 | 120 + 8 | 1,1466e-7 |
| 19 | 64 + 4 | 8,9752e-8 |

Mọi ca khớp lựa chọn attention/expert và chuỗi greedy sinh độc lập. Chi tiết cùng hash source C: `verification/sharded-linux-cpu.json`. Đây không phải benchmark tokens/giây hoặc bằng chứng giới hạn RAM/GPU trên máy mục tiêu.

**Chưa chạy lại tại đây:** DLL MSVC, CUDA/RTX 3070, Windows Job Object, graph Transformers đúng revision, checkpoint thật. Đã nối đường parity chính thức và thêm test opt-in; cần chạy các lệnh Windows ở đầu tài liệu để có bằng chứng tương ứng.

## Bước tiếp theo

Chạy official hybrid safetensors trên máy mục tiêu, gồm ca 120 + 8, rồi kiểm tra các report thật. Nếu các ca đó đạt, tiếp tục xác minh mapping/dtype/scale từ metadata checkpoint đã ghim và thiết kế projection nhiều block ở kích thước lớn. Chưa tải 755 GB trọng số chỉ để làm các bước nhỏ này.

Nguồn định dạng: [Hugging Face sharded checkpoints](https://github.com/huggingface/transformers/blob/main/docs/source/en/models.md#sharded-checkpoints), [safetensors format](https://github.com/huggingface/safetensors#format). Các đường dẫn nhánh chính có thể thay đổi; contract miniature và revision oracle được ghim riêng trong source.


## Bản sửa 0.6.1 — serializer TensorSpec

### Report đầu vào

ZIP `reports(1).zip` chỉ có hai file `parity-latest.json` và `parity-latest.md`; không có log `test-reference.bat` hoặc các thư mục run. Bản JSON giữ nguyên tại [verification/user-parity-error-v0.6.0.json](verification/user-parity-error-v0.6.0.json).

- `status = ERROR`, `child_exit_code = 1`.
- Lỗi: `TypeError: argument 'tensor_dict': 'dict' object is not an instance of 'TensorSpec'`.
- `job_policy_verified = true`: policy CPU 70%, committed memory 32.000.000.000 byte đã cài.
- `synthetic_official_parity_verified = false`, `inference_verified = false`, `full_model_loaded = false`.

Report này không ghi version safetensors, traceback, backend/storage/lengths hoặc cases, nên không gán cho nó kết quả test hay loại GPU không có trong file. Căn cứ vị trí gọi trong source, lỗi ở `write_mini_shards` xảy ra trước khi tạo backend C/CUDA; không có bằng chứng đây là lỗi phần cứng.

### Nguyên nhân và phạm vi sửa

Source 0.6.0 dùng `safetensors.serialize` với descriptor dạng dictionary. Lock của project ghim `safetensors==0.8.0`, trong khi bằng chứng Linux cũ dùng 0.7.0. Đối chiếu [source safetensors tại tag v0.8.0](https://github.com/huggingface/safetensors/blob/v0.8.0/bindings/python/src/lib.rs), serializer mới yêu cầu mỗi value là `TensorSpec(dtype=..., shape=..., data_ptr=..., data_len=...)`. Caller phải giữ buffer sống trong suốt lời gọi; `TensorSpec` không sở hữu dữ liệu ở địa chỉ đó.

`safetensor_serializer.py` tạo descriptor đúng theo API được thư viện thực tế cung cấp. Với `TensorSpec`, helper tự sở hữu bản sao buffer bằng ctypes, giữ tất cả buffer tới khi serialize và chuyển output sang bytes hoàn tất. Tensor rỗng có địa chỉ sống khác null với `data_len=0`. Trường hợp không có `TensorSpec` vẫn dùng raw dictionary. Không bắt `TypeError` để thử lại hoặc nuốt lỗi, không đổi version lock, không thêm Torch làm dependency cho exporter. Payload mỗi lần serialize giới hạn 64 KiB; reader/runtime streaming không đổi.

FP8 bytes không được giải mã rồi lượng tử hóa lại. Fixture vẫn bốn shard, 80 tensor, 10.072 byte payload và 34 cặp weight/scale khác shard. Report fixture có thêm `serializer_api` để phân biệt `TensorSpec` với `raw-dict`.

`parity_worker.py` giữ `parameters`, `tool_version`, `worker_environment` và traceback giới hạn 16.000 ký tự khi lỗi. `parity_run.py` chuyển các thông tin này sang JSON/Markdown latest và per-run, đồng thời giữ trạng thái ERROR/exit code và các cờ chưa xác minh. Không thay graph, oracle, kernels C/PTX, ABI native, quota, revision hoặc điều kiện doctor BLOCKED.

### Bằng chứng mới và giới hạn

Linux, Python 3.13.5, NumPy 2.3.5, Torch 2.10.0+cpu, safetensors 0.7.0; không có Transformers. [Log unit suite](verification/unit-tests-linux-v0.6.1.txt): **366 test được phát hiện; 340 đạt, 26 bỏ qua**. Có 19 test mới cho serializer và report; các test opt-in native/Windows/CUDA/official bị bỏ qua không được tính là đạt.

[Regression API](verification/serializer-api-regression-v0.6.1.json) nạp exporter gốc 0.6.0 và tái hiện đúng TypeError với một test double chỉ chấp nhận hợp đồng TensorSpec. Exporter 0.6.1 qua cùng hợp đồng và giữ nguyên toàn bộ matrix/vector/embedding so với fixture private. Đây là **mô phỏng hợp đồng 0.8**, không phải thực thi Rust extension 0.8.0. Thư viện thật 0.7.0 tạo/đọc file và kiểm tra byte round-trip; test `safe_open` hiện có tiếp tục đối chiếu đủ 80 tensor. Bộ test kiểm tra vòng đời mọi buffer khi có garbage collection, giải phóng khi lỗi, tensor rỗng/scalar và việc không thử lại sau TypeError.

Không tải được wheel safetensors 0.8.0 trong môi trường hiện tại do không truy cập được máy chủ gói. Do đó còn phải chạy **thư viện thật 0.8.0 và official hybrid trên máy người dùng**; chưa có kết luận parity đầy đủ hoặc GPU pacing đạt.

### Chạy lại trên Windows

Áp dụng toàn bộ source 0.6.1 lên project hiện tại, giữ `.venv-reference`, `build` và các report cũ. Không cần build lại DLL hoặc chạy setup-reference chỉ vì bản sửa Python này.

Kiểm tra riêng serializer bằng môi trường đã ghim (không cần chạy GPU):

```powershell
.\.venv-reference\Scripts\python.exe -m unittest discover -s tests -p "test_safetensor_serializer.py" -v
```

Sau khi lệnh trên đạt, chạy lần lượt; dừng nếu một lệnh báo lỗi:

```powershell
.\test-reference.bat 2>&1 | Tee-Object -FilePath .\reports\test-reference-v0.6.1.log
.\glm.bat parity --storage safetensors --backend hybrid
.\glm.bat parity --storage safetensors --backend hybrid --lengths 120 --generate 8
```

Kiểm tra `tool_version=0.6.1`, `parameters.storage=safetensors`, fixture `serializer_api=TensorSpec` và các case `PASS`. Lần chạy tiếp ghi đè latest, nhưng vẫn lưu report per-run riêng. Gửi ZIP **cả cây thư mục reports/**, gồm log test và các thư mục `parity/<run-id>/`, thay vì chỉ hai file latest. Chỉ chuyển sang metadata checkpoint/multi-block khi official hybrid trên máy đích đạt.
