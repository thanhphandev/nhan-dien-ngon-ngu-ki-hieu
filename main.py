import cv2
import mediapipe as mp
import sys
import time
import warnings
from collections import deque

# Import các module xử lý
from utils.feature_extraction import extract_features
from utils.strings import ExpressionHandler
from utils.tts import TextToSpeech
from utils.model import ASLClassificationModel
from utils.post_processing import PredictionSmoother
from config import MODEL_NAME, MODEL_CONFIDENCE, SMOOTHING_WINDOW_SIZE, CONFIDENCE_THRESHOLD

import streamlit as st

# Bỏ qua các cảnh báo không cần thiết
warnings.filterwarnings("ignore")

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(page_title="ASL Recognition App", layout="wide")

st.markdown("""
    <style>
        .big-font {
            color: #e76f51 !important;
            font-size: 50px !important;
            font-weight: bold;
            border: 2px solid #fcbf49;
            border-radius: 10px;
            padding: 10px;
            text-align: center;
            background-color: #ffffff;
        }
        /* Căn giữa video */
        div.stImage {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HÀM LOAD MODEL (CACHE ĐỂ TĂNG TỐC)
# ==========================================
@st.cache_resource
def load_ai_model():
    """Load model một lần duy nhất để tránh lag khi reload"""
    print("Loading model...")
    return ASLClassificationModel.load_model(f"models/{MODEL_NAME}")

# Load model ngay khi vào app
try:
    model = load_ai_model()
except Exception as e:
    st.error(f"Lỗi không tìm thấy model: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR & CẤU HÌNH
# ==========================================
st.sidebar.title("🔧 Bảng Điều Khiển")

# NÚT QUAN TRỌNG: BẬT/TẮT CAMERA
# Checkbox này đóng vai trò như công tắc nguồn
run_camera = st.sidebar.checkbox("📷 Bật Camera", value=True)

# Cấu hình độ nhạy AI
st.sidebar.markdown("---")
st.sidebar.subheader("Độ nhạy AI (Threshold)")
detection_confidence = st.sidebar.slider("Min Detection Confidence", 0.0, 1.0, MODEL_CONFIDENCE, 0.05)
tracking_confidence = st.sidebar.slider("Min Tracking Confidence", 0.0, 1.0, MODEL_CONFIDENCE, 0.05)

# Cấu hình TTS (Giọng nói)
st.sidebar.markdown("---")
st.sidebar.subheader("🔊 Cấu hình Giọng nói")
tts_enabled = st.sidebar.checkbox("Bật đọc kết quả", value=False)
tts_engine_choice = st.sidebar.selectbox("Công cụ đọc", ["pyttsx3 (Offline)", "gTTS (Vietnamese, Online)"], index=0)
min_interval = st.sidebar.slider("Khoảng cách giữa các lần đọc (s)", 1.0, 5.0, 2.0, 0.5)

# Xử lý TTS Voice ID (nếu dùng pyttsx3)
tts_voice = None
if "pyttsx3" in tts_engine_choice:
    tts_voice = st.sidebar.text_input("Voice ID (pyttsx3 - Optional)", value="") or None

# Khởi tạo TTS Session
if 'tts' not in st.session_state:
    st.session_state.tts = None
    st.session_state.tts_engine = None

desired_engine = 'pyttsx3' if 'pyttsx3' in tts_engine_choice else 'gtts'

# Logic khởi tạo/huỷ TTS
if tts_enabled:
    # Nếu chưa có TTS hoặc đổi engine thì khởi tạo lại
    if st.session_state.tts is None or st.session_state.tts_engine != desired_engine:
        try:
            with st.spinner("Đang khởi tạo giọng nói..."):
                st.session_state.tts = TextToSpeech(engine=desired_engine, lang='vi', voice=tts_voice)
                st.session_state.tts_engine = desired_engine
        except Exception as e:
            st.sidebar.error(f"Lỗi TTS: {e}")
            tts_enabled = False
elif not tts_enabled and st.session_state.tts is not None:
    # Tắt TTS nếu người dùng bỏ chọn
    try:
        st.session_state.tts.stop()
    except:
        pass
    st.session_state.tts = None

# ==========================================
# 4. GIAO DIỆN CHÍNH
# ==========================================
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 🎥 Camera Feed")
    video_placeholder = st.empty()

with col2:
    st.markdown("### 📝 Kết quả Dự đoán")
    prediction_placeholder = st.empty()
    
    st.markdown("#### Độ tin cậy")
    confidence_bar = st.progress(0)
    confidence_text = st.empty()

    st.markdown("#### Lịch sử")
    history_placeholder = st.empty()

    # Khu vực hiển thị FPS và thông số
    st.markdown("---")
    fps_display = st.empty()

# ==========================================
# 5. LOGIC XỬ LÝ CAMERA (LOOP)
# ==========================================
if run_camera:
    # Khởi tạo Mediapipe
    mp_holistic = mp.solutions.holistic
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Mở Camera
    cap = cv2.VideoCapture(0)
    
    expression_handler = ExpressionHandler()
    smoother = PredictionSmoother(window_size=SMOOTHING_WINDOW_SIZE)
    prev_time = 0 # Dùng để tính FPS
    
    # History buffer
    prediction_history = deque(maxlen=5)

    # Sử dụng 'with' để tự động giải phóng tài nguyên Mediapipe khi tắt loop
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence) as face_mesh, \
         mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence) as hands:

        while cap.isOpened() and run_camera:
            success, image = cap.read()
            if not success:
                st.warning("Không tìm thấy camera hoặc camera đang bận.")
                break

            # Tính toán FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            fps_display.metric("FPS Tốc độ xử lý", f"{int(fps)}")

            # Xử lý hình ảnh
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 1. Detect Faces & Hands
            face_results = face_mesh.process(image)
            hand_results = hands.process(image)

            # 2. Vẽ lên hình
            image.flags.writeable = True
            # Vẽ Face
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )
            # Vẽ Hands
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                    )

            # 3. Dự đoán cử chỉ
            try:
                # Trích xuất đặc trưng (Dùng hàm mới nhất của bạn)
                feature = extract_features(mp_hands, face_results, hand_results)
                
                # Đưa vào model
                # expression = model.predict(feature) # Cũ
                label, confidence = model.predict_with_confidence(feature)
                
                # Thêm vào bộ làm mượt
                smoother.add_prediction(label, confidence)
                smoothed_label, smoothed_confidence = smoother.get_smoothed_prediction()

                # Logic hiển thị
                ui_text = "..."
                if smoothed_confidence >= CONFIDENCE_THRESHOLD:
                    expression_handler.receive(smoothed_label)
                    ui_text = expression_handler.get_message()
                    
                    # Cập nhật history nếu có thay đổi
                    if not prediction_history or prediction_history[-1] != ui_text:
                        prediction_history.append(ui_text)
                else:
                    ui_text = "..." # Không chắc chắn

                # Hiển thị Text
                prediction_placeholder.markdown(f'<div class="big-font">{ui_text}</div>', unsafe_allow_html=True)
                
                # Hiển thị Confidence
                confidence_bar.progress(min(smoothed_confidence, 1.0))
                confidence_text.text(f"Độ tin cậy: {round(smoothed_confidence * 100, 1)}%")
                
                # Hiển thị History
                history_html = "<ul>" + "".join([f"<li>{item}</li>" for item in prediction_history]) + "</ul>"
                history_placeholder.markdown(history_html, unsafe_allow_html=True)

                # Đọc giọng nói
                if tts_enabled and st.session_state.tts and smoothed_confidence >= CONFIDENCE_THRESHOLD:
                    speech_text = expression_handler.get_speech_message()
                    st.session_state.tts.speak_if_allowed(speech_text, min_interval=min_interval)

            except Exception as e:
                print(f"Prediction error: {e}")

            # Hiển thị hình ảnh lên Web
            video_placeholder.image(image, channels="RGB", use_column_width=True)

    # Giải phóng camera khi thoát vòng lặp
    cap.release()
    cv2.destroyAllWindows()

else:
    # Giao diện khi Camera Tắt
    st.info("Camera đang tắt. Tích vào ô '📷 Bật Camera' ở thanh bên trái để bắt đầu.")
    video_placeholder.empty()
    prediction_placeholder.empty()
    fps_display.empty()