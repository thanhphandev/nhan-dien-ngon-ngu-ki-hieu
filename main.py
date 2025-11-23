import cv2
import mediapipe as mp
import sys
import time
import warnings
from collections import deque

# Import các module xử lý của dự án
# Đảm bảo cấu trúc folder đúng như bạn đã upload
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
        /* Làm đẹp thanh progress bar */
        .stProgress > div > div > div > div {
            background-color: #2a9d8f;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HÀM LOAD MODEL (CACHE ĐỂ TĂNG TỐC)
# ==========================================
@st.cache_resource
def load_ai_model():
    """Load model một lần duy nhất để tránh lag khi reload"""
    print(f"Đang tải model từ: models/{MODEL_NAME}...")
    return ASLClassificationModel.load_model(f"models/{MODEL_NAME}")

# Load model ngay khi vào app
try:
    model = load_ai_model()
    st.sidebar.success(f"✅ Đã tải model: {MODEL_NAME}")
except Exception as e:
    st.error(f"❌ Lỗi nghiêm trọng: Không tìm thấy model tại 'models/{MODEL_NAME}'")
    st.error(f"Chi tiết lỗi: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR & CẤU HÌNH
# ==========================================
st.sidebar.title("🔧 Bảng Điều Khiển")

# NÚT QUAN TRỌNG: BẬT/TẮT CAMERA
run_camera = st.sidebar.checkbox("📷 Bật Camera", value=True)

# Cấu hình độ nhạy AI
st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Độ nhạy AI")
detection_confidence = st.sidebar.slider("Độ nhạy phát hiện (Detection)", 0.0, 1.0, 0.7, 0.05, help="Tăng lên nếu máy nhận diện nhầm nhiễu nền là tay")
tracking_confidence = st.sidebar.slider("Độ nhạy theo dõi (Tracking)", 0.0, 1.0, MODEL_CONFIDENCE, 0.05)
current_threshold = st.sidebar.slider("Ngưỡng chốt đáp án (Threshold)", 0.0, 1.0, CONFIDENCE_THRESHOLD, 0.05, help="Chỉ hiển thị kết quả khi độ tin cậy vượt qua mức này")

# Cấu hình TTS (Giọng nói)
st.sidebar.markdown("---")
st.sidebar.subheader("🔊 Cấu hình Giọng nói")
tts_enabled = st.sidebar.checkbox("Bật đọc kết quả", value=True)
tts_engine_choice = st.sidebar.selectbox("Công cụ đọc", ["gTTS (Google - Online, Tiếng Việt hay)", "pyttsx3 (Offline - Nhanh)"], index=0)
min_interval = st.sidebar.slider("Khoảng cách giữa các lần đọc (giây)", 1.0, 5.0, 2.5, 0.5)

# Xử lý TTS Voice ID (nếu dùng pyttsx3)
tts_voice = None
if "pyttsx3" in tts_engine_choice:
    tts_voice = st.sidebar.text_input("Voice ID (pyttsx3 - Tùy chọn)", value="") or None

# --- Logic Khởi tạo/Huỷ TTS Session ---
if 'tts' not in st.session_state:
    st.session_state.tts = None
    st.session_state.tts_engine = None

desired_engine = 'pyttsx3' if 'pyttsx3' in tts_engine_choice else 'gtts'

# Nếu bật TTS nhưng chưa có object hoặc đổi engine -> Tạo mới
if tts_enabled:
    if st.session_state.tts is None or st.session_state.tts_engine != desired_engine:
        try:
            with st.spinner("Đang khởi tạo giọng nói..."):
                # Lưu ý: lang='vi' quan trọng cho gTTS
                st.session_state.tts = TextToSpeech(engine=desired_engine, lang='vi', voice=tts_voice)
                st.session_state.tts_engine = desired_engine
        except Exception as e:
            st.sidebar.error(f"Lỗi khởi tạo TTS: {e}")
            tts_enabled = False
# Nếu tắt TTS mà đang có object -> Hủy
elif not tts_enabled and st.session_state.tts is not None:
    try:
        st.session_state.tts.stop()
    except:
        pass
    st.session_state.tts = None

# ==========================================
# 4. GIAO DIỆN CHÍNH
# ==========================================
st.title("🤟 Nhận Diện Ngôn Ngữ Ký Hiệu Việt Nam")

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

    # Khu vực hiển thị FPS
    st.markdown("---")
    fps_display = st.empty()
    status_text = st.empty()

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

    # Mở Camera (Thử index 0, nếu lỗi thử 1)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Không thể mở Camera. Vui lòng kiểm tra kết nối.")
        st.stop()
    
    expression_handler = ExpressionHandler()
    # Sử dụng window size từ config hoặc hardcode nhỏ hơn nếu muốn nhanh hơn
    smoother = PredictionSmoother(window_size=SMOOTHING_WINDOW_SIZE)
    
    prev_time = 0
    prediction_history = deque(maxlen=5)

    # Context Manager cho Mediapipe giúp quản lý tài nguyên tốt hơn
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
                st.warning("Mất tín hiệu camera.")
                break

            # Tính FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            fps_display.metric("FPS (Tốc độ)", f"{int(fps)}")

            # Chuẩn bị ảnh cho Mediapipe (BGR -> RGB)
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 1. Detect Faces & Hands
            face_results = face_mesh.process(image)
            hand_results = hands.process(image)

            # 2. Vẽ lại lên ảnh (RGB -> BGR để hiển thị opencv nếu cần, nhưng streamlit dùng RGB cũng được)
            # Tuy nhiên Mediapipe vẽ đẹp hơn trên BGR gốc rồi convert lại sau, 
            # ở đây ta vẽ trực tiếp lên ảnh RGB hiện tại để hiển thị luôn
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

            # ============================================================
            # 3. DỰ ĐOÁN CỬ CHỈ (ĐÃ SỬA LỖI TỰ ĐỌC KHI KHÔNG CÓ TAY)
            # ============================================================
            
            # Mặc định là không có kết quả
            ui_text = "..."
            smoothed_confidence = 0.0
            
            # CHỈ XỬ LÝ KHI PHÁT HIỆN CÓ BÀN TAY
            if hand_results.multi_hand_landmarks:
                try:
                    # Trích xuất đặc trưng (Feature Extraction)
                    feature = extract_features(mp_hands, face_results, hand_results)
                    
                    # Đưa vào model AI
                    label, confidence = model.predict_with_confidence(feature)
                    
                    # Làm mượt kết quả (Smoothing)
                    smoother.add_prediction(label, confidence)
                    smoothed_label, smoothed_confidence = smoother.get_smoothed_prediction()

                    # Chỉ hiển thị/đọc nếu độ tin cậy vượt ngưỡng (Threshold)
                    if smoothed_confidence >= current_threshold:
                        expression_handler.receive(smoothed_label)
                        ui_text = expression_handler.get_message()
                        
                        # Cập nhật lịch sử
                        if not prediction_history or prediction_history[-1] != ui_text:
                            prediction_history.append(ui_text)
                        
                        # Đọc giọng nói (TTS)
                        if tts_enabled and st.session_state.tts:
                            status_text.text(f"🔊 Đang đọc: {ui_text}")
                            speech_text = expression_handler.get_speech_message()
                            st.session_state.tts.speak_if_allowed(speech_text, min_interval=min_interval)
                    else:
                        # Có tay nhưng AI chưa chắc chắn
                        ui_text = "..." 
                        status_text.text("🤔 Đang phân tích...")

                except Exception as e:
                    print(f"Lỗi dự đoán: {e}")
                    status_text.text("⚠️ Lỗi xử lý AI")
            else:
                # KHÔNG CÓ TAY: Reset trạng thái
                status_text.text("Sẵn sàng. Hãy đưa tay vào camera.")
                # Có thể chọn reset bộ làm mượt để lần sau đưa tay vào nhận diện nhanh hơn
                # smoother.clear() 

            # ============================================================
            # 4. CẬP NHẬT GIAO DIỆN NGƯỜI DÙNG
            # ============================================================
            
            # Hiển thị kết quả chữ to
            prediction_placeholder.markdown(f'<div class="big-font">{ui_text}</div>', unsafe_allow_html=True)
            
            # Hiển thị thanh độ tin cậy
            confidence_bar.progress(min(smoothed_confidence, 1.0))
            confidence_text.text(f"Độ tin cậy: {round(smoothed_confidence * 100, 1)}%")
            
            # Hiển thị lịch sử
            history_html = "<ul>" + "".join([f"<li>{item}</li>" for item in prediction_history]) + "</ul>"
            history_placeholder.markdown(history_html, unsafe_allow_html=True)

            # Hiển thị hình ảnh camera
            video_placeholder.image(image, channels="RGB", use_column_width=True)

    # Giải phóng camera khi thoát
    cap.release()
    cv2.destroyAllWindows()

else:
    # Giao diện chờ khi chưa bật camera
    st.info("👋 Chào mừng! Hãy tích vào ô '📷 Bật Camera' ở thanh bên trái để bắt đầu sử dụng.")
    video_placeholder.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbXp4Z3Bpbm94Z3Bpbm94Z3Bpbm94Z3Bpbm94Z3Bpbm94Z3Bpbm94Z3Bpbm94Z3Bpbm94ZCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7TKUM3IgJBq2M3QA/giphy.gif", caption="Minh họa ngôn ngữ ký hiệu")
    prediction_placeholder.empty()
    fps_display.empty()