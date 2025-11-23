import pickle


class ASLClassificationModel:
    @staticmethod
    def load_model(model_path):
        # Load model and mapping from pickle
        with open(model_path, "rb") as file:
            model, mapping = pickle.load(file)

        if model is not None:
            return ASLClassificationModel(model, mapping)

        raise Exception("Model not loaded correctly!")

    def __init__(self, model, mapping):
        self.model = model
        self.mapping = mapping

    def predict(self, feature):
        """
        Dự đoán nhãn (chỉ trả về text).
        """
        return self.mapping[self.model.predict(feature.reshape(1, -1)).item()]

    def predict_with_confidence(self, feature):
        """
        Dự đoán nhãn và độ tin cậy.
        Trả về: (label, confidence)
        """
        feature = feature.reshape(1, -1)
        
        # Kiểm tra xem model có hỗ trợ probability không
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(feature)[0]
            max_prob_index = probabilities.argmax()
            confidence = probabilities[max_prob_index]
            label_index = self.model.classes_[max_prob_index] # SVC classes_ stores the labels
            
            # Lưu ý: self.mapping map từ index (lúc train) sang string. 
            # Nếu SVC được train với y là số nguyên, classes_ sẽ là các số nguyên đó.
            # Tuy nhiên, để an toàn, ta dùng predict() để lấy label index chuẩn nếu logic trên phức tạp.
            
            # Cách đơn giản hơn: dùng predict để lấy class, rồi lấy max prob
            predicted_class = self.model.predict(feature).item()
            # Tìm index của predicted_class trong model.classes_ để lấy probability tương ứng
            # model.classes_ chứa danh sách các class unique đã train
            class_idx_in_proba = list(self.model.classes_).index(predicted_class)
            confidence = probabilities[class_idx_in_proba]
            
            return self.mapping[predicted_class], confidence
        else:
            # Fallback nếu model cũ không có probability=True
            # Dùng decision_function (khoảng cách đến siêu phẳng) để ước lượng thô (không chính xác là xác suất)
            # Hoặc trả về 1.0 mặc định
            return self.predict(feature), 1.0