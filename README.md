Mô hình Phân loại Ngôn ngữ Ký hiệu bằng Ước tính Tư thế
Về dự án
Dự án này sử dụng thư viện MediaPipe của Google để thực hiện ước tính tư thế (pose estimation) trên luồng cấp dữ liệu từ webcam. Sau đó, nó xây dựng một mô hình phân loại đơn giản sử dụng dữ liệu từ quá trình ước tính tư thế để phân loại các tín hiệu ngôn ngữ ký hiệu đơn giản.

Cách sử dụng mã nguồn
Lưu ý: Bạn đã có thể chạy dự án ngay lập tức với dữ liệu đã thu thập và mô hình đã được huấn luyện, nhưng bạn có thể tự thu thập và huấn luyện một mô hình của riêng mình theo các bước sau:

Clone (nhân bản) dự án về.

Cài đặt các gói được chỉ định trong tệp requirements.txt.

Thiết lập đường dẫn chính xác cho dự án của bạn.

(TÙY CHỌN) Chạy lệnh sau để thu thập dữ liệu tư thế cho một ký hiệu ngôn ngữ ký hiệu đơn lẻ:

python scripts/capture_pose_data.py --pose_name="[TÊN CỦA KÝ HIỆU]" --confidence=[ĐỘ TIN CẬY CỦA MÔ HÌNH ƯỚC TÍNH TƯ THẾ (THƯỜNG LÀ 0.5)]
(TÙY CHỌN) Sau khi thu thập dữ liệu cho tất cả các hành động bạn muốn, huấn luyện mô hình bằng lệnh:

python scripts/train.py --model_name=[TÊN CỦA MÔ HÌNH BẠN MUỐN] 
(TÙY CHỌN) Thay thế tên mô hình trong tệp config.py bằng tên mô hình của bạn.

Chạy chương trình Streamlit bằng lệnh:

streamlit run main.py
Tài liệu Chi tiết
Tiếng Việt: docs/PROJECT_OVERVIEW.vi.md

Mới: Chuyển văn bản thành giọng nói (Text-to-Speech - tùy chọn)
Bật tính năng này trong thanh bên (sidebar) của Streamlit ("Bật đọc giọng nói (TTS)").

Yêu cầu gói pyttsx3 (đã được thêm vào requirements). Hoạt động ngoại tuyến trên Windows.

Các ký hiệu đã được huấn luyện
Các ký hiệu đã được huấn luyện trong dự án bao gồm:

Xin chào (Hello) (ASL).

Không (No) (ASL).

Cảm ơn (Thank you) (ASL).

Yêu (Love) (ASL).

Không làm gì cả (Do nothing).