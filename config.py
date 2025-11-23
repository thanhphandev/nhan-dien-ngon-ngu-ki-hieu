# Pose Estimation Configuration
FEATURES_PER_HAND = 21

# Name of the model
MODEL_NAME = "simple_8_expression_model.pkl"
MODEL_CONFIDENCE = 0.5

# Cấu hình xử lý hậu kỳ (Post-processing)
SMOOTHING_WINDOW_SIZE = 10  # Số lượng frame để làm mượt dự đoán
CONFIDENCE_THRESHOLD = 0.6  # Ngưỡng tin cậy tối thiểu để hiển thị kết quả
