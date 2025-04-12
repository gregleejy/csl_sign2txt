import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Labels for phrases
label_map = {
    '1': 'zaoan', # 早上好
    '2': 'bukeqi', # 不客气
    '3': 'henganxiejiandaoni' # 很高兴见到你
}

SEQUENCE_LENGTH = 30
DATA_PATH = 'phrase_data'
os.makedirs(DATA_PATH, exist_ok=True)

X, y = [], []

print("Press 1, 2, or 3 to start collecting phrase signs. Press 'q' to quit.")

cap = cv2.VideoCapture(0)
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

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
    else:
        frame_buffer.clear()

    # Display
    cv2.putText(frame, "Press key [1-3] to capture phrase", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    key = cv2.waitKey(1) & 0xFF
    if chr(key) in label_map and len(frame_buffer) == SEQUENCE_LENGTH:
        phrase = label_map[chr(key)]
        X.append(list(frame_buffer))
        y.append(phrase)
        print(f"Captured sequence for: {phrase} — Total samples: {len(X)}")
        frame_buffer.clear()
        time.sleep(1)

    if key == ord('q'):
        break

    cv2.imshow("Phrase Sign Capture", frame)

cap.release()
cv2.destroyAllWindows()

# Save data
np.save(os.path.join(DATA_PATH, 'X.npy'), np.array(X))
np.save(os.path.join(DATA_PATH, 'y.npy'), np.array(y))
print("Data saved.")
