import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image  # For Chinese text rendering

# Load model and label encoder
model = load_model("lstm_phrase_model.h5")
le = joblib.load("label_encoder_phrase.joblib")

# Phrase mapping (pinyin to full Chinese)
chinese_phrase_map = {
    "zaoan": "早上好",                   # Good morning
    "bukeqi": "不客气",                 # You're welcome
    "henganxiejiandaoni": "很高兴见到你"  # Nice to meet you
}

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Parameters
SEQUENCE_LENGTH = 30
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

# Load Chinese font
font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 32)  # Microsoft YaHei

cap = cv2.VideoCapture(0)
print("Webcam activated. Sign a phrase...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    coords = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
    
    if coords:
        frame_buffer.append(coords)

    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    if len(frame_buffer) == SEQUENCE_LENGTH:
        input_seq = np.expand_dims(frame_buffer, axis=0)
        prediction = model.predict(input_seq, verbose=0)
        predicted_label = le.inverse_transform([np.argmax(prediction)])[0]
        chinese_output = chinese_phrase_map.get(predicted_label, "未知手势")  # Fallback for unknown
        
        draw.text((10, 30), f"预测: {chinese_output}", font=font, fill=(0, 255, 0))
    
    else:
        draw.text((10, 30), '请持续比出一个完整手势...', font=font, fill=(0, 255, 255))

    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow("Real-time Phrase Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
