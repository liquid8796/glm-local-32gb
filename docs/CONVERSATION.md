# Hội thoại với ModelDesk 1.1.1 / core 0.11.1

Trong **Chạy model**, chọn chế độ **Hội thoại**, thư mục chứa đúng checkpoint và backend **cpu**. Context 4096, tối đa 256 token mới và reasoning **low** là cấu hình khởi đầu của giao diện. Lập kế hoạch bộ nhớ trước khi chạy. Trọng số NVFP4 vẫn có hàng trăm GB; đọc theo nhu cầu giúp giới hạn RAM nhưng mỗi lượt có thể kéo dài đáng kể.

Tùy chọn **Trả lời trực tiếp** yêu cầu model bắt đầu phần trả lời ngay. Mặc định tùy chọn này tắt; bật khi muốn giảm phần suy luận được sinh trước câu trả lời. Model vẫn có thể mở lại phần suy luận, khi đó ứng dụng tiếp tục hiển thị riêng phần này.

## Một cuộc hội thoại

Gửi tin nhắn đầu tiên và theo dõi câu trả lời trong trang Chạy model. Phần suy luận được thu gọn riêng; token suy luận cũng tính vào số token sinh tối đa. Một phản hồi chỉ được coi là hoàn tất khi model kết thúc câu trả lời đúng giao thức. Hết token, hết timeout hoặc chỉ có suy luận được hiển thị là chưa hoàn tất, kể cả khi tiến trình đã sinh được dữ liệu.

Các cặp tin nhắn đã hoàn tất được gửi lại ở lượt sau. Phần trả lời dở vẫn hiển thị nhưng không đi vào lịch sử tiếp theo; có thể thử lại mà không mất nội dung câu hỏi. Tạo cuộc trò chuyện mới để xóa lịch sử trong giao diện. Đổi profile hoặc thư mục trọng số cần một cuộc trò chuyện mới. Mỗi lượt chạy một worker riêng và xử lý lại lịch sử; phiên bản này chưa duy trì KV cache giữa các worker.

Context phải chứa cả lịch sử, chat template và ngân sách token mới. 4096 là giới hạn của lượt chạy đã chọn, không thay đổi khả năng context được công bố trong config model. Khi lịch sử không vừa, ứng dụng báo lỗi để người dùng rút ngắn hoặc mở cuộc trò chuyện mới; không âm thầm cắt mất tin nhắn. Chế độ **Văn bản thô** và **Token ID** vẫn dành cho kiểm tra completion/decoder.

## CLI

```powershell
.\modeldesk-cli.bat chat --profile nvfp4 --prompt "Xin chào" `
  --backend cpu --context 4096 --max-tokens 256 --timeout 1800 --reasoning low

.\modeldesk-cli.bat chat --profile nvfp4 --messages "D:\Chats\conversation.json" `
  --context 4096 --max-tokens 256 --timeout 7200

.\modeldesk-cli.bat --json chat --profile nvfp4 --prompt "Xin chào"

.\modeldesk-cli.bat chat --profile nvfp4 --prompt "Xin chào" --direct-answer
```

File lịch sử là mảng JSON UTF-8, tối đa 1 MiB và 256 tin nhắn:

```json
[
  {"role": "system", "content": "Trả lời ngắn gọn bằng tiếng Việt."},
  {"role": "user", "content": "Tên tôi là An."},
  {"role": "assistant", "content": "Chào An!"},
  {"role": "user", "content": "Tôi tên gì?"}
]
```

Lệnh thường đưa câu trả lời ra stdout khi có dữ liệu; nhật ký và suy luận (khi được yêu cầu) đi ra stderr. `--json` chỉ đưa kết quả JSON cuối ra stdout. Ctrl+C hủy cây tiến trình. Mã 2 biểu thị phản hồi chưa hoàn tất; xem `Generation` và báo cáo thay vì coi mọi mã khác 1 là thành công. Xem `modeldesk-cli help` để biết tên tùy chọn đầy đủ.

CLI hiển thị ký tự điều khiển terminal trong nội dung model dưới dạng chữ đã escape. Dùng `--json` khi công cụ khác cần lấy chính xác nội dung phản hồi gốc.

Python có thể chạy trực tiếp:

```powershell
.\glm.bat --profile nvfp4 generate --prompt "Xin chào" `
  --prompt-format chat --reasoning-effort low --stream-events `
  --backend cpu --context 4096 --generate 256 --timeout 7200 --prefill-batch-size 16
```

`--messages-file` nhận cùng cấu trúc lịch sử và tự chọn chat. `--keep-thinking` giữ phần suy luận cũ khi có; mặc định bỏ phần đó khỏi những lượt assistant trước. Template GLM đã ghim luôn mở chế độ suy luận. Tùy chọn `--direct-answer` của công cụ nối thêm token đóng `</think>` vào tiền tố assistant đã kiểm chứng; đây là phần mở rộng tiền tố của công cụ, không phải một cờ tắt thinking do template gốc cung cấp. Tùy chọn chỉ hợp lệ với chat, không tạo sẵn nội dung trả lời và không đổi trọng số. Python raw completion giữ hành vi cũ khi không chọn chat.

## Hiệu năng và báo cáo

CPU xử lý dải tối đa 128 hàng bằng pool tối đa tám luồng. Prefill gom tối đa 16 vector, chia sẻ lần đọc trọng số và dùng SIMD trên các vector độc lập. Thứ tự cộng FP32 của từng kết quả giữ nguyên. K/V đã mở rộng được dùng lại trong phạm vi tối đa 256 token mỗi layer; vượt dung lượng thì LRU loại mục cũ và tái tính khi cần.

Một luồng đọc trước dải trọng số tiếp theo trong khi kernel CPU xử lý dải hiện tại. Tối đa hai dải cùng tồn tại; reader chỉ được bàn giao lại sau khi luồng đọc kết thúc. Các kết quả kernel cho từng token được chuyển bằng bộ đệm FP32, tránh chuyển đổi lại từng số ở Python.

Planner tính cả cache K/V, bộ đệm trọng số 8 MiB, scratch dense 33 MiB, scratch NVFP4 5 MiB, workspace prefill 128 MiB và tối đa 16 MiB cho hai dải đọc trước. Windows Job tiếp tục giới hạn bộ nhớ commit/CPU theo cấu hình. Các con số planner là ước lượng dung lượng, không phải cam kết throughput hay độ chính xác của cả checkpoint.

Với config NVFP4 đã ghim, context 4096 và 256 token mới, planner của bản này ước tính 5.449.468.740 byte, gồm khoảng 2,29 GB dung lượng cache K/V mở rộng và 2 GiB dự phòng runtime. Cache mở rộng được cấp phát khi dùng; số đo tiến trình thực tế có thể khác ước lượng.

Nhật ký giai đoạn và `progress.json` cho biết khởi tạo, prefill, decode và layer đang thực hiện. `response-progress.json` giữ phần trả lời đã sinh để phục hồi khi timeout. Timeout tính toàn bộ lượt, bao gồm xác minh metadata và xử lý prompt. Tăng timeout chỉ cấp thêm thời gian; không tăng tốc model. Thay đổi lô prefill về 1 dùng đường scalar để đối chiếu.

Báo cáo hoàn tất có `timings`: thời gian khởi tạo, prefill, token đầu tiên và tốc độ decode đo trong worker. Token đầu tiên có thể là suy luận; tốc độ tính cả token suy luận và EOS. Lượt chỉ sinh một token không có phép đo decode độc lập nên tốc độ decode là `null`.

Phần `reader.cpu_parallelism` ghi số luồng kernel và giới hạn đọc trước; đây là cấu hình tối đa, không phải số nhân luôn bận. Các bộ đếm đọc trước ghi thời gian producer đọc và consumer chờ; hai phần có thể chạy chồng nhau.

Giao thức `MODELDESK_EVENT` là các dòng JSON giữa Python và ModelDesk, có sự kiện bắt đầu, cập nhật assistant/reasoning và kết thúc. Vị trí thay thế được tính theo UTF-16 để GUI/CLI xử lý đúng tiếng Việt và emoji. Đây là cầu nối tiến trình cục bộ; ứng dụng chưa cung cấp HTTP API hoặc provider OpenAI-compatible cho công cụ bên ngoài.
