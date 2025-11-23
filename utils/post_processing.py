from collections import deque, Counter
import numpy as np

class PredictionSmoother:
    def __init__(self, window_size=10):
        """
        Khởi tạo bộ làm mượt dự đoán.
        :param window_size: Số lượng frame gần nhất để xem xét (cửa sổ trượt).
        """
        self.window_size = window_size
        self.prediction_buffer = deque(maxlen=window_size)
        self.confidence_buffer = deque(maxlen=window_size)

    def add_prediction(self, label, confidence):
        """
        Thêm một dự đoán mới vào buffer.
        :param label: Nhãn dự đoán (str).
        :param confidence: Độ tin cậy (float).
        """
        self.prediction_buffer.append(label)
        self.confidence_buffer.append(confidence)

    def get_smoothed_prediction(self):
        """
        Lấy kết quả dự đoán đã được làm mượt (voting).
        :return: (smoothed_label, avg_confidence)
        """
        if not self.prediction_buffer:
            return None, 0.0

        # Voting: Tìm nhãn xuất hiện nhiều nhất trong buffer
        counter = Counter(self.prediction_buffer)
        most_common_label, count = counter.most_common(1)[0]

        # Tính độ tin cậy trung bình cho nhãn đó (chỉ tính các frame dự đoán ra nhãn đó)
        relevant_confidences = [
            conf for lbl, conf in zip(self.prediction_buffer, self.confidence_buffer)
            if lbl == most_common_label
        ]
        
        avg_confidence = np.mean(relevant_confidences) if relevant_confidences else 0.0
        
        return most_common_label, avg_confidence

    def reset(self):
        """Xóa buffer"""
        self.prediction_buffer.clear()
        self.confidence_buffer.clear()
