# Tài liệu dự án: Nhận dạng ngôn ngữ ký hiệu bằng MediaPipe + SVM
#Tài liệu hướng dẫn chi tiết bằng tiếng Việt về kiến trúc, luồng xử lý, chức năng từng mô-đun, và các đề xuất cải tiến.
#Tài liệu này giúp bạn hiểu toàn bộ source code, kiến trúc, luồng dữ liệu, cách chạy/huấn luyện và các hướng cải tiến.

## 1) Tổng quan
- Bài toán: Phân loại 5 cử chỉ tĩnh: "xin_chào", "không", "cảm_ơn", "yêu", "bình_thường".
- Công nghệ chính:
  - MediaPipe Face Mesh + Hands để trích xuất landmark (tọa độ x, y).
  - Trích xuất đặc trưng đơn giản: trung bình (x, y) của toàn bộ điểm mặt + toàn bộ (x, y) của 21 điểm mỗi bàn tay.
  - Mô hình phân loại: SVM (RBF kernel) huấn luyện trên các đặc trưng này.
  - Giao diện chạy real-time: Streamlit (trong `main.py`) hoặc OpenCV window (trong `scripts/test_model.py`).

## 2) Kiến trúc và luồng xử lý
```
Webcam → MediaPipe (FaceMesh, Hands)
     ↘  trích xuất đặc trưng (utils/feature_extraction.py)
       → vector đặc trưng (d = 86)
       → model SVM (utils/model.py, models/*.pkl)
       → nhãn thô (ví dụ: "xin_chào")
       → ExpressionHandler (utils/strings.py) → câu hiển thị tiếng Việt
       → Render (Streamlit/OpenCV)
```
- Kích thước đặc trưng mỗi khung hình:
  - Mặt: mean(x, y) → 2 số.
  - Hai tay: 21 điểm × 2 tọa độ × 2 tay = 84 số.
  - Tổng: 2 + 84 = 86 chiều.

## 3) Các mô-đun chính (file và chức năng)
- `config.py`
  - `FEATURES_PER_HAND = 21`, `MODEL_NAME = "simple_5_expression_model.pkl"`, `MODEL_CONFIDENCE = 0.5`.
- `utils/feature_extraction.py`
  - `extract_face_result(face_results)`: trả về vector 2-D (mean x, y) của các landmark khuôn mặt; nếu không có mặt → vector 0.
  - `extract_hand_result(mp_hands, hand_results)`: trả về vector 84-D cho 2 tay (trường hợp không phát hiện tay hoặc chỉ 1 tay đều được xử lý để giữ kích thước đầu ra cố định).
  - `extract_features(mp_hands, face_results, hand_results)`: ghép mặt (2) + tay (84) thành 86 chiều.
- `utils/model.py`
  - Lớp `ASLClassificationModel`: nạp mô hình từ pickle `(model, mapping)` và dự đoán nhãn.
- `utils/strings.py`
  - `ExpressionHandler`: ánh xạ nhãn thô (tên file lớp trong `data/`) sang câu tiếng Việt thân thiện UI.
- `scripts/capture_pose_data.py`
  - Thu thập dữ liệu: ghi 86-D feature theo thời gian và lưu thành `data/<pose_name>.npy`.
- `scripts/train.py`
  - Đọc toàn bộ `.npy` trong `data/`, gán index lớp theo thứ tự file, train SVM, in độ chính xác và lưu `(model, mapping)` vào `models/<model_name>.pkl`.
- `scripts/test_model.py`
  - Mở webcam, inference real-time với OpenCV window (đã sửa import để dùng `utils/feature_extraction`).
- `main.py`
  - Ứng dụng Streamlit hai cột: bên trái video, bên phải câu dự đoán; nạp mô hình từ `models/MODEL_NAME` trong `config.py`.

## 4) Cách chạy nhanh (Windows PowerShell)
- Cài đặt phụ thuộc (khuyến nghị Python 3.10 hoặc 3.11 cho MediaPipe 0.10.x):
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
- Chạy Streamlit UI (không chạy `python main.py`):
```powershell
streamlit run main.py
```
 - Bật TTS (đọc giọng nói): trong Sidebar của Streamlit, bật "Bật đọc giọng nói (TTS)", chọn engine:
   - pyttsx3 (Offline): phụ thuộc giọng SAPI5 cài trong Windows (chất lượng tiếng Việt phụ thuộc voice của hệ thống).
   - gTTS (Vietnamese, Online): chất lượng tiếng Việt tự nhiên hơn, cần internet.
   - Chọn khoảng cách đọc (giây) để tránh đọc liên tục.
- Thu thập dữ liệu cho một cử chỉ mới (ví dụ 60 giây, tin cậy 0.6):
```powershell
python scripts/capture_pose_data.py --pose_name="chao" --confidence=0.6 --duration=60
```
- Huấn luyện lại mô hình từ dữ liệu trong `data/`:
```powershell
python scripts/train.py --model_name=my_model
```
- Cập nhật `config.py` để dùng mô hình mới:
```python
MODEL_NAME = "my_model.pkl"
```
- Kiểm tra nhanh bằng OpenCV window:
```powershell
python scripts/test_model.py --model_path models/my_model.pkl --confidence 0.6
```

## 5) Dòng dữ liệu và quy ước nhãn
- Mỗi file `data/<ten_nhan>.npy` chứa nhiều vector 86-D, một vector cho mỗi khung hình.
- Lúc train, `mapping[class_index] = <ten_nhan>` (tên file không gồm `.npy`).
- Lúc inference, `model.predict(feature)` cho ra `class_index`, sau đó ánh xạ ngược ra `ten_nhan` và cuối cùng `ExpressionHandler` đổi ra câu hiển thị.

## 6) Mẹo chất lượng dữ liệu
- Thu nhiều mẫu cho mỗi nhãn, ở các điều kiện ánh sáng khác nhau.
- Đảm bảo tay trong khung hình, tránh che khuất.
- Giữ camera ổn định; có thể dùng gimbal/giá đỡ.

## 7) Hạn chế hiện tại
- Đặc trưng đơn giản (mean khuôn mặt + toạ độ thô của tay), nhạy với vị trí trong khung hình và khoảng cách camera.
- Không có smoothing thời gian → UI có thể nhấp nháy khi dự đoán đổi liên tục.
- SVM mặc định không `predict_proba` (chưa bật `probability=True`) → khó đặt ngưỡng tự tin.

## 8) Cải tiến đề xuất (mức độ thấp rủi ro)
- Tính ổn định dự đoán:
  - Thêm bộ đệm (deque) 10–20 khung, lấy mode/average để làm mượt.
  - Hiển thị nhãn chỉ khi tần suất ≥ ngưỡng (ví dụ 60%).
- Chuẩn hoá đặc trưng:
  - Chuẩn hoá toạ độ theo kích thước khung hình, hoặc chuẩn hoá theo khoảng cách cổ tay–cổ tay và tâm tay để giảm phụ thuộc vị trí.
  - Thêm vận tốc/độ chênh lệch khung (delta features) nếu muốn bắt cử chỉ động.
- Mô hình:
  - Dùng `SVC(probability=True)` để có `predict_proba`, sau đó đặt ngưỡng tin cậy.
  - Thử `LinearSVC` + chuẩn hoá, hoặc MLPClassifier (sklearn) khi dữ liệu tăng.
- Huấn luyện:
  - Thêm `StratifiedKFold`/cross-validation, lưu confusion matrix.
  - Cố định seed, log tham số và độ đo bằng `logging`.
- Codebase:
  - Thêm type hints, docstring, xử lý lỗi khi thiếu model/data.
  - Viết unit tests nhỏ cho kích thước đặc trưng và mapping.

## 9) Tính năng mới gợi ý (trung bình rủi ro)
- Real-time UI tốt hơn trong Streamlit:
  - Nút Start/Stop thay cho `cv2.waitKey`.
  - Hiển thị thanh confidence và lịch sử dự đoán.
- Thêm cử chỉ mới: chỉ cần thu thập `data/<ten>.npy` rồi huấn luyện lại.
- Export mô hình:
  - Lưu bằng `joblib` cho nhanh.
  - Thử `skl2onnx` để xuất ONNX (phụ thuộc mô hình sử dụng).
- Demo UI web/app:
  - Dùng Streamlit Components hoặc Gradio.

## 10) Lỗi thường gặp và cách xử lý
- Import lỗi `utils.features` trong `scripts/test_model.py` (đã sửa thành `utils/feature_extraction`).
- Lỗi import `mediapipe`, `opencv-python`: đảm bảo đúng phiên bản Python (3.10/3.11) và cài đặt theo `requirements.txt`.
- Model không khớp kích thước đặc trưng: xoá mô hình cũ, huấn luyện lại sau khi thay đổi logic đặc trưng.

## 11) Hợp đồng đầu vào/đầu ra (mini spec)
- Đầu vào inference: một khung hình RGB từ webcam.
- Đầu ra: chuỗi nhãn (`str`) và câu hiển thị tiếng Việt từ `ExpressionHandler`.
- Lỗi: nếu không phát hiện tay/mặt → đặc trưng 0; mô hình vẫn dự đoán nhưng độ tin cậy có thể thấp.

## 12) Bước tiếp theo khuyến nghị
- Thêm smoothing + confidence threshold.
- Chuẩn hoá đặc trưng theo kích thước khung hình.
- Thêm báo cáo huấn luyện (confusion matrix, classification report).
- Tách UI (Streamlit) khỏi xử lý camera bằng thread-safe queue để ổn định FPS.

—
Tài liệu này phản ánh trạng thái code ở thời điểm 2025-10-25. Nếu bạn đổi logic đặc trưng hoặc mô hình, hãy cập nhật lại phần kích thước đặc trưng và hướng dẫn huấn luyện.
 
## Phụ lục: Thêm 2 cử chỉ mới “tôi” và “bạn”

Mục tiêu: thêm 2 lớp mới vào mô hình bằng cách thu thập dữ liệu mới và huấn luyện lại.

### A) Chuẩn bị
- Bạn đã cài dependencies (pip install -r requirements.txt) và có webcam.
- Code đã hỗ trợ hiển thị 2 nhãn mới trong UI (đã thêm vào `utils/strings.py`).

### B) Thu thập dữ liệu (mỗi cử chỉ ~60–120 giây)
- Mở PowerShell tại thư mục dự án, kích hoạt virtualenv nếu có.
- Với cử chỉ “tôi”:
```powershell
python scripts/capture_pose_data.py --pose_name="tôi" --confidence=0.6 --duration=90
```
- Với cử chỉ “bạn”:
```powershell
python scripts/capture_pose_data.py --pose_name="bạn" --confidence=0.6 --duration=90
```
- Lưu ý khi thu:
  - Giữ tay trong khung hình, thử nhiều tư thế/xa-gần để dữ liệu đa dạng.
  - Mỗi cử chỉ nên có tối thiểu 1–2 phút dữ liệu; có thể chạy lặp lại để tăng dữ liệu.
  - Sau khi chạy, bạn sẽ có thêm `data/tôi.npy` và `data/bạn.npy`.

### C) Huấn luyện lại mô hình
- Huấn luyện từ toàn bộ dữ liệu trong thư mục `data/` (bao gồm 5 nhãn cũ + 2 nhãn mới):
```powershell
python scripts/train.py --model_name=simple_7_expression_model
```
- Kết quả: file mô hình mới `models/simple_7_expression_model.pkl` cùng mapping lớp.

### D) Cập nhật cấu hình để dùng mô hình mới
- Sửa file `config.py`:
```python
MODEL_NAME = "simple_7_expression_model.pkl"
```

### E) Chạy kiểm tra real-time
- Streamlit UI:
```powershell
streamlit run main.py
```
- Hoặc OpenCV window:
```powershell
python scripts/test_model.py --model_path models/simple_7_expression_model.pkl --confidence 0.6
```

Ghi chú import khi chạy trực tiếp file trong `scripts/`:
- Dự án đã thêm dòng thiết lập `sys.path` trong các script để import được `utils/…`. Nếu bạn muốn chạy theo cách “chuẩn module”, có thể gọi dạng:
```powershell
python -m scripts.capture_pose_data --pose_name="tôi" --confidence=0.6 --duration=90
```

## 13) Text-to-Speech (TTS)
- Engine hỗ trợ:
  - Offline: `pyttsx3` (SAPI5 Windows). Chất lượng phụ thuộc voice cài đặt. Có thể nhập Voice ID nếu bạn có giọng Việt trên máy.
  - Online: `gTTS` + `playsound` (đã có trong requirements). Tiếng Việt tự nhiên hơn, cần internet.
- Bật trong Sidebar, chọn engine và chỉnh “Khoảng cách đọc (giây)” để debounce.

### F) Gợi ý cải thiện chất lượng
- Nếu dự đoán chưa ổn định, tăng thời lượng thu thập hoặc thu ở nhiều bối cảnh ánh sáng.
- Cân nhắc thêm smoothing thời gian và ngưỡng confidence (xem mục Cải tiến §8).
