# Reader safetensors và FP8 block 0.5.0

Đã có đường đọc file safetensors thật về định dạng, nhưng **chỉ kiểm chứng bằng tensor giả lập**. Không tải hoặc mở checkpoint GLM đầy đủ. Lệnh kiểm chứng không nhận đường dẫn/model ID tùy ý; nó tự tạo file thử nhỏ.

## Cập nhật 0.7.0

Đường decoder nhiều shard đã được nghiệm thu ở 0.6.1; xem [SHARDED-DECODER.md](SHARDED-DECODER.md). Bản 0.7.0 tách `parse_header_bytes` từ parser cũ để dùng chung với [audit metadata từ xa](CHECKPOINT-METADATA.md). Các quy tắc dtype/shape/offset, header 1 MiB và đọc tối đa 64 KiB của reader cũ không đổi. Hàm mới không mở file, không tạo file sparse giả và không đọc payload; nó xác minh header theo kích thước file được caller cung cấp.

## Chạy lại

```powershell
.\glm.bat storage-check
.\glm.bat storage-check --backend cpu --seed 19
.\test-reference.bat
```

Double-click `storage-check.bat` giữ cửa sổ mở để xem kết quả. `.venv-reference` hiện đã có đủ safetensors 0.8.0, Torch CPU và Transformers; trên máy khác dùng `setup-reference.bat` trước. Các kernel C/CUDA dùng chung bản trước, không cần cài thêm thư viện cho máy hiện tại.

Mỗi lần chạy tạo ba file ở `reports/storage/<run-id>/` và báo cáo riêng. Bản gần nhất là `reports/storage-latest.md`/JSON. Mã thoát 0: các phép kiểm chứng đạt; 3: sai lệch số học; 1: lỗi môi trường/file/worker. Trạng thái model đầy đủ của `doctor` vẫn là `BLOCKED`.

## Profile định dạng được hỗ trợ

Reader Python standard library không dùng mmap. Prefix 8 byte chứa độ dài header dạng unsigned little-endian; offset tensor tính từ đầu payload. Header được đọc theo đoạn trước khi parse JSON.

| Giới hạn cục bộ | Giá trị |
|---|---:|
| Kích thước file tối đa | 2 TiB |
| Header tối đa | 1 MiB |
| Số tensor tối đa | 4.096 |
| Số chiều tối đa | 8 |
| Mỗi lần đọc file tối đa | 64 KiB |
| Ma trận con trả về tối đa | 128×128 và 64 KiB |
| Block FP8 để tính toán tối đa | 128×128 = 16 KiB |

Đây là policy của project, không phải tất cả giới hạn của safetensors. Các dtype byte-aligned được nhận gồm F8_E4M3, F16, BF16, F32, F64, các số nguyên 8/16/32/64 bit signed/unsigned và BOOL. Không hỗ trợ packed FP4, FNUZ hoặc mọi dtype mới của thư viện.

Kiểm tra duplicate JSON keys, UTF-8, metadata string-to-string, shape/offset nguyên không âm, số byte đúng dtype, không chồng lấn/khoảng trống/dữ liệu dư. Scalar `shape=[]` và tensor có chiều 0 được hỗ trợ. Không bắt header phải chia hết cho 8: đó là quy ước ghi của thư viện, không phải điều kiện bắt buộc khi đọc. Parser chủ động từ chối một số đầu vào mơ hồ được parser khác chấp nhận.

`SafeTensorReader.read_bytes` trả vùng byte trong một tensor; `read_matrix_tile` gom đúng các hàng của vùng ma trận không liên tục trên đĩa. Thống kê ghi số lần/byte thực đã đọc. Trên Windows, fingerprint của path và file descriptor được lưu riêng vì `ctime` có thể khác cách diễn giải.

## FP8 và scale

Adapter `FP8BlockMatrix(reader, weight_name, scale_name)` yêu cầu tên rõ ràng, không đoán từ tên model. Weight phải là F8_E4M3 2D; scale là F32 2D với shape `ceil(rows/128) × ceil(cols/128)`. Khối cuối có thể nhỏ hơn 128. Mỗi khối chỉ đọc một scale 4 byte và vùng trọng số tương ứng.

`weight_scale_inv` là **hệ số nhân khi giải mã**: `float(weight_fp8) * scale`. Tên chứa `inv` không có nghĩa chia. Scale phải dương/hữu hạn. Reader định dạng cho phép payload NaN theo đặc tả; adapter tính toán từ chối mã FP8 NaN 0x7F/0xFF.

## Đối chiếu và kết quả

**301 kiểm thử đạt** qua `test-reference.bat`, gồm parser, block adapter, thư viện tham chiếu và toàn bộ kiểm thử C/CUDA/decoder trước đó. Đã chạy thêm CPU riêng với seed 19; cả ba kích thước đều đạt.

File thử được tạo bằng `safetensors.torch.save` rồi đọc độc lập bằng `safe_open`. Có sáu tensor, bao gồm FP8, F32 scale/vector, BF16, scalar và tensor rỗng. So sánh metadata và từng byte tensor với reader riêng, sau đó gửi các block qua CPU/GPU với scale khác nhau theo hàng/cột.

Tham chiếu số học mở rộng scale bằng PyTorch CPU float64. Với 256×384, còn gọi bộ giải mã FP8 nguyên trạng của Transformers đã ghim để đối chiếu. Bộ giải mã đó yêu cầu chiều chia hết cho shape lưới scale, nên **không dùng làm oracle cho ragged 257×259**; ca ragged được đối chiếu với phép mở rộng 128×128 độc lập, có cắt biên.

Lần kiểm chứng hybrid ngày 2026-09-12:

| Ma trận | Block CPU / GPU | Khối biên nhỏ | Sai số lớn nhất |
|---|---:|---:|---:|
| 256×384 | 3 / 3 | 0 | 1,85e-6 |
| 257×259 | 6 / 3 | 5 | 1,19e-6 |
| 1×1 | 1 / 0 | 1 | 0 |

Ba file khoảng 100.370, 68.141 và 491 byte. Thư viện có thể thay đổi thứ tự metadata khi serialize, nên cùng seed bảo đảm cùng giá trị tensor, không hứa hash toàn file giống nhau. Mỗi lần vẫn ghi hash của đúng file đã dùng.

Worker RSS đỉnh khoảng 404 MiB trong phép thử đầu, gồm Torch và tham chiếu. Reader chỉ giữ header và vùng được yêu cầu; oracle riêng giữ toàn bộ tensor nhỏ. Thống kê I/O của reader gồm lượt so sánh byte và lượt tính native, không phải benchmark đọc SSD lạnh. File cache của hệ điều hành có thể phục vụ các lượt đọc sau.

## Điều chưa được chứng minh

- Safetensors không chứa checksum nội tại. Fingerprint chỉ phát hiện thay đổi thông thường; kiểm chứng byte trong ca thử không thay thế hash toàn bộ checkpoint sau tải.
- File size policy 2 TiB không phải bằng chứng đã thử file 2 TiB. Chưa đo reader với shard model thật hoặc cả 282 shard.
- Chưa nối safetensors loader vào toàn bộ decoder nhiều tensor/shard; chưa xác minh mapping checkpoint thật hay ngân sách KV/activation ở kích thước model.
- CPU quota/commit cap vẫn được cài trước worker, nhưng trần RAM vật lý và GPU trung bình dưới tải lâu dài chưa được chứng minh.

Bước tiếp theo được ghi nhận ở bản 0.5.0 là nối decoder thu nhỏ với safetensors nhiều shard. Đường đọc này đã triển khai trong **0.6.0**; xem [SHARDED-DECODER.md](SHARDED-DECODER.md) về kiểm chứng và nghiệm thu official Windows 0.6.1 đã nhận. Vẫn chỉ dùng file thử nhỏ; chưa cần tải checkpoint 755 GB.

Nguồn: [đặc tả safetensors v0.8.0](https://github.com/safetensors/safetensors/tree/v0.8.0#format), [bộ giải mã FP8 Transformers đã ghim](https://github.com/huggingface/transformers/blob/3f601734a3580f55484720770850966bba060e4f/src/transformers/integrations/finegrained_fp8.py).
