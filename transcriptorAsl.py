import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow.keras.models import load_model

model = load_model('model_asl.h5') 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
           'U', 'V', 'W', 'X', 'Y', 'Z', 'DEL', 'NOTHING', 'SPACE']

def keypoints_to_image(keypoints, size=(64, 64)):
    image = np.zeros(size + (3,), dtype=np.uint8)
    for i in range(0, len(keypoints), 3):
        x = int(keypoints[i] * size[0])
        y = int(keypoints[i + 1] * size[1])
        cv2.circle(image, (x, y), 2, (255, 255, 255), -1)
    return image

cap = cv2.VideoCapture(0)

current_character = None
last_character = None
last_detection_time = time.time()
timeout = 2
reset_timeout = 10
output_text = ""
last_hand_detected_time = time.time()
preview_character = ""

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        last_hand_detected_time = time.time()
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            keypoints_image = keypoints_to_image(keypoints)

            keypoints_image = cv2.resize(keypoints_image, (64, 64))
            keypoints_image = np.expand_dims(keypoints_image, axis=0)

            prediction = model.predict(keypoints_image)
            label = np.argmax(prediction)

            current_character = classes[label]
            preview_character = current_character

            if current_character == last_character:
                if time.time() - last_detection_time >= timeout:
                    if current_character == "DEL":
                        output_text = output_text[:-1]
                    elif current_character == "SPACE":
                        output_text += " "
                    elif current_character == "NOTHING":
                        pass
                    else:
                        output_text += current_character
                    last_detection_time = time.time()
            else:
                last_character = current_character
                last_detection_time = time.time()
    else:
        if time.time() - last_hand_detected_time >= reset_timeout:
            output_text = ""
        preview_character = ""

    cv2.putText(frame, f'Preview: {preview_character}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Output: {output_text}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('ASL Gesture Recognition', frame)
    if cv2.waitKey(5) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
