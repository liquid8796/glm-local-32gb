# Current Memory Snapshot — GLM Local 32GB

Cập nhật ngày 2026-09-12 sau khi đọc `reports(2).zip`. Baseline source giữ nguyên **0.6.1**. Đây là snapshot nghiệm thu riêng, không phải bản vá source hoặc bản phát hành mới. Không sửa source, dependency lock, kernel, configuration hoặc revision trong lần review này.

## Trạng thái hiện tại

**Đã qua mốc Windows official miniature hybrid parity với storage safetensors.** Bản sửa serializer TensorSpec trong 0.6.1 đã có bằng chứng chạy với safetensors 0.8.0 thực trên máy người dùng. Không còn chờ chạy lại ba lệnh kiểm chứng đã yêu cầu cho mốc này.

Kết luận chỉ áp dụng bộ test và fixture synthetic được gửi; không khẳng định full checkpoint đã chạy hoặc đã tương thích.

## Mục tiêu và ràng buộc giữ nguyên từ memory/source trước

Checkpoint được cấu hình: `dealignai/GLM-5.3-CYBERSECURITY-FP8`, revision `5915c1b88f998a9c1e1a0c83688e285a08ae3ca5`. Không đổi model, revision, FP8, không tự chuyển GGUF. Review này không truy cập checkpoint từ xa hoặc tải thêm trọng số.

Máy mục tiêu theo memory: Windows x64, i7-11800H, RTX 3070 Laptop 8 GiB. Report mới trực tiếp ghi nhận NVIDIA GeForce RTX 3070 Laptop GPU; review không đo lại CPU vật lý. CPU hard cap 70%, committed-memory budget 32.000.000.000 byte, GPU pacing target 60% trong cửa sổ 10 giây. Không đồng nhất committed-memory limit với trần RAM vật lý toàn hệ thống; không gọi pacing target là GPU hard cap.

## Bằng chứng nhận từ người dùng

Nguồn: `reports(2).zip`, gồm log `reports/test-reference-v0.6.1.log`, hai thư mục per-run có request/result/fixture, cùng hai file parity-latest. Không chạy lại test Windows tại môi trường assistant; các số dưới đây là kết quả trong report của người dùng.

Log test là UTF-16; khi đọc nguyên nội dung kết thúc bằng:

```text
Ran 376 tests in 23.004s

OK
```

Đếm được 376 kết quả `... ok`, không có test skipped hoặc mục failure/error trong kết quả unittest. Dòng PowerShell `NativeCommandError` gần đầu log đi kèm test đầu tiên đã `ok`; không dùng dòng đó để kết luận test thất bại khi bản tổng kết là OK. Log không ghi riêng exit code của test-reference; hai parity worker đều ghi child_exit_code=0.

Hai report đều ghi:

- tool_version=0.6.1; Python 3.13.12; safetensors 0.8.0; serializer_api=TensorSpec.
- storage.format=safetensors; status=PASS; synthetic_official_parity_verified=true.
- provenance.status=verified: Torch 2.14.0+cpu, NumPy 2.4.6, Transformers 5.18.0.dev0; Transformers revision `3f601734a3580f55484720770850966bba060e4f`.
- Provenance chỉ xác minh version ba package, archive metadata và SHA-256 hai source GLM-MoE-DSA; không phải toàn bộ byte của môi trường.
- job_policy_verified=true; installed CPU/memory policy đúng giá trị đã cấu hình.

| Run ID | Prompt + generated | Max logit error | Max hidden error | Kết quả |
|---|---:|---:|---:|---|
| `20260912T033848Z-fcc4acfe` | 8 + 4 | 1.1920929e-07 | 3.20976746e-07 | PASS; attention/expert/greedy đều khớp |
| `20260912T033848Z-fcc4acfe` | 32 + 4 | 1.78813934e-07 | 3.15322921e-07 | PASS; attention/expert/greedy đều khớp |
| `20260912T033848Z-fcc4acfe` | 64 + 4 | 1.78813934e-07 | 3.16760455e-07 | PASS; attention/expert/greedy đều khớp |
| `20260912T033908Z-8ef23c55` | 120 + 8 | 1.78813934e-07 | 3.51147065e-07 | PASS; attention/expert/greedy đều khớp |

Cả bốn case dùng seed 7; failed_steps=0. Chuỗi generated_ids khớp official_generated_ids, selection attention và routing expert khớp theo quy tắc so sánh của project. Precision được khai báo: per-element atol=2e-5, rtol=3e-4; bit_exact=false. Official reference chạy CPU FP32; native projections dùng C/CUDA FP32, host nonlinear/cache dùng float64.

Run đầu ghi 3.016 CPU projection calls và 116 GPU projection calls; run 120+8 ghi 3.328 CPU calls và 128 GPU calls. Hybrid chỉ đưa output head lên CUDA; không phải cả decoder chạy GPU.

Reader: bốn shard, 80 tensor, 34 cặp weight/scale chéo shard; fixed_mapping_verified=true; peak_open_shards=2; không giữ FP8 payload cache. Reference vẫn đọc fixture private độc lập.

## Kiểm tra chéo thực hiện trong lần review

Với cả hai run, đã đối chiếu request.parameters với result.parameters; tính SHA-256 và kiểm tra kích thước các shard được gửi; đối chiếu hash index và private weights.bin với report. Các giá trị đều khớp. Tổng projection calls khớp tổng các case; GPU admissions khớp GPU calls. parity-latest.json khớp byte-for-byte JSON của run dài cuối cùng.

Kiểm tra này xác nhận tính nhất quán của các file đã gửi; không xác thực checkpoint bên ngoài hoặc thay thế việc tái chạy độc lập.

## Tài nguyên và giới hạn còn lại

Run đầu: peak worker RSS 419,11 MiB, peak worker private commit 569,16 MiB. Run dài: RSS 419,57 MiB, private commit 573,65 MiB. Đây là worker miniature gồm CPU Torch/oracle, không phải RAM/VRAM full model.

Pacing có samples, admitted submissions và telemetry_errors=0. GPU peak trong samples là 25% và 35%; latest weighted average khoảng 10,13% và 18,02%. Telemetry theo whole_device, có thể gồm tiến trình khác và bỏ lỡ workload ngắn; thời gian quan sát khoảng 8,08 và 8,82 giây. gpu_cap_verified=false. Không dùng các số này để cam kết GPU full model luôn dưới 60%.

Các cờ sau vẫn false và phù hợp với phạm vi test:

```text
inference_verified=false
real_checkpoint_compatible=false
full_model_loaded=false
physical_ram_hard_cap_verified=false
full_model_limits_verified=false
```

Chỉ test trọng số/ID giả lập hai layer. Chưa tải tokenizer/full checkpoint, chưa có inference full model hoặc nghiệm thu ngân sách tài nguyên full model. `doctor BLOCKED` không được tự gỡ chỉ vì miniature parity đã PASS.

## Điểm cần theo dõi khi mở rộng

Run dài có shard_open_count=6.107 và shard_evictions=6.101, giới hạn hai handle mở, 74.550 read calls. Đây là bằng chứng nhiều thao tác mở/đóng và đọc nhỏ; là điểm cần đánh giá khi scale, chưa phải chứng minh bottleneck tốc độ hoặc lỗi tính đúng. Không tự tăng cache vô hạn hoặc bỏ kiểm tra shard để tối ưu.

## Bước tiếp theo

Không có lỗi mới cần vá từ report này. Giữ baseline source 0.6.1; không tạo version mới chỉ để đánh dấu PASS.

Chuyển sang bước metadata tensor/dtype/shape/scale của checkpoint revision đã ghim: đối chiếu config/index/header, có giới hạn đọc rõ ràng; chỉ báo verified những gì có bằng chứng, không tự suy diễn mapping synthetic là mapping checkpoint thật. Chưa tải toàn bộ trọng số.

Sau đó mở rộng projection FP8 nhiều block, kiểm chứng scale/edge tiles/cross-shard bằng reference độc lập, giữ ngân sách đọc và bộ nhớ. Khi scale mới xem xét locality/cache có giới hạn để giảm mở lại shard. Chưa triển khai bước metadata hoặc projection mới trong lần review này.

Với bản vá code tiếp theo: cập nhật Markdown sẵn có trong source, tăng project version và glm_local.__version__ đồng bộ, xuất toàn bộ source ZIP và commit message theo type(scope): message. Project Python/C không có .NET AssemblyVersion; không tăng native ABI khi giao diện native không đổi.
