import numpy as np  # import thư viện numpy để xử lý mảng
from config import *  # import các biến cấu hình như FEATURES_PER_HAND

#  TRÍCH XUẤT ĐẶC TRƯNG BÀN TAY
def extract_hand_result(mp_hands, hand_results):
    """
    Trích xuất đặc trưng bàn tay từ Mediapipe.
    Trả về vector cố định: Right hand trước, Left hand sau.
    """

    # Nếu không phát hiện bàn tay nào → trả về vector toàn 0 (giữ shape cố định)
    if hand_results is None or hand_results.multi_hand_landmarks is None:
        return np.zeros(FEATURES_PER_HAND * 4)  # 2 tay × (21 điểm × 2 toạ độ)


    # Lấy danh sách landmark và handedness
    hands = hand_results.multi_hand_landmarks  # Danh sách bàn tay
    handed = hand_results.multi_handedness      # Danh sách trái/phải


    # Tạo 2 biến chứa dữ liệu tay phải và tay trái
    right_hand_array = np.zeros((21, 2))  # vector mặc định cho tay phải
    left_hand_array = np.zeros((21, 2))   # vector mặc định cho tay trái


    # Duyệt từng bàn tay cùng với nhãn trái/phải
    for hand_landmark, hand_label in zip(hands, handed):

        label = hand_label.classification[0].label  # lấy text "Right" hoặc "Left"

        hand_array = extract_single_hand(mp_hands, hand_landmark)  # trích từng tay

        if label == "Right":  
            right_hand_array = hand_array  # gán vào tay phải
        else:
            left_hand_array = hand_array   # gán vào tay trái


    # Ghép 2 tay theo thứ tự: Right → Left
    return np.hstack((right_hand_array.flatten(),
                      left_hand_array.flatten()))  # return vector 84 chiều

#  TRÍCH XUẤT 21 LANDMARK CỦA 1 TAY
def extract_single_hand(mp_hands, hand_landmarks):
    # Tạo mảng 21 điểm, mỗi điểm (x, y)
    landmarks_array = np.zeros((21, 2))  

    # Duyệt 21 điểm landmark của Mediapipe
    for i, lm in enumerate(hand_landmarks.landmark):
        landmarks_array[i] = [lm.x, lm.y]  # Lưu toạ độ x, y vào mảng

    return landmarks_array

#  TRÍCH XUẤT ĐẶC TRƯNG KHUÔN MẶT
def extract_face_result(face_results):
    """
    Trích xuất đặc trưng khuôn mặt.
    Ở đây dùng trung bình 468 điểm → vector (x_mean, y_mean)
    """

    # Nếu không có khuôn mặt → return vector 0 (để giữ shape)
    if face_results is None or face_results.multi_face_landmarks is None:
        return np.zeros(2)

    # Lấy mặt đầu tiên
    face = face_results.multi_face_landmarks[0]

    # Chuyển tất cả landmark thành mảng [[x, y], ...]
    face_array = np.array([[lm.x, lm.y] for lm in face.landmark])

    # Tính trung bình theo trục → (mean_x, mean_y)
    return np.mean(face_array, axis=0)

#  GHÉP TẤT CẢ ĐẶC TRƯNG THÀNH 1 VECTOR DUY NHẤT
def extract_features(mp_hands, face_results, hand_results):
    """
    Ghép face_features + right hand + left hand → thành vector duy nhất
    """

    face_features = extract_face_result(face_results)   # vector 2 chiều
    hand_features = extract_hand_result(mp_hands, hand_results)  # vector 84 chiều

    return np.hstack((face_features, hand_features))  # vector tổng cộng 86 chiều
