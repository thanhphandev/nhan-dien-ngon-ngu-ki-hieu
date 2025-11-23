
class ExpressionHandler:

    MAPPING = {
        "bình_thường": "Ngồi yên 🤐",
        "cảm_ơn": "Cảm ơn 😘",
        "xin_chào": "Xin chào 🙋‍",
        "yêu": "Yêu ❤️",
        "không": "Không 🤚",
        # Mở rộng nhãn mới
        "tôi": "Tôi 👤",
        "bạn": "Bạn 🙂",
        "bánh_mì": "Bánh mì 🍞"
    }

    # Bản đọc cho TTS (không emoji, từ ngữ rõ ràng)
    SPEECH_MAPPING = {
        "bình_thường": "Ngồi yên",
        "cảm_ơn": "Cảm ơn",
        "xin_chào": "Xin chào",
        "yêu": "Yêu",
        "không": "Không",
        "tôi": "Tôi",
        "bạn": "Bạn",
        "bánh_mì": "Bánh mì"
    }

    def __init__(self):
        # Save the current message and the time received the current message
        self.current_message = ""

    def receive(self, message):
        self.current_message = message

    def get_message(self):
        # Trả về nhãn gốc nếu chưa có mapping thân thiện để tránh lỗi
        return ExpressionHandler.MAPPING.get(self.current_message, self.current_message)

    def get_speech_message(self):
        return ExpressionHandler.SPEECH_MAPPING.get(self.current_message, self.current_message)
