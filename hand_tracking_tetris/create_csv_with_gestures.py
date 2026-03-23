import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

data = {"fist": [], "thumb_left": [], "thumb_right": [], "open_hand": [], "pause": [], "esc": [], "thumb_down": []}
max_samples = 3000  # maksimalan broj po gestu

print("Instrukcije:")
print("f - fist / stisnuta šaka")
print("l - thumb left / palac levo")
print("r - thumb right / palac desno")

print("s - sačuvaj CSV i izađi")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Prikaz broja preostalih primera za svaki gest
    y0, dy = 30, 30
    for i, (label, feats_list) in enumerate(data.items()):
        text = f"{label}: {max_samples - len(feats_list)} preostalo"
        cv2.putText(frame, text, (10, y0 + i*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Collect Gestures", frame)
    key = cv2.waitKey(1) & 0xFF

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        features = []
        for lm in hand_landmarks.landmark:
            features.append(lm.x)
            features.append(lm.y)
          #  features.append(lm.z)

        if key == ord('f') and len(data["fist"]) < max_samples:
            data["fist"].append(features)
            print(f"Saved fist, total: {len(data['fist'])}")
        elif key == ord('l') and len(data["thumb_left"]) < max_samples:
            data["thumb_left"].append(features)
            print(f"Saved thumb_left, total: {len(data['thumb_left'])}")
        elif key == ord('r') and len(data["thumb_right"]) < max_samples:
            data["thumb_right"].append(features)
            print(f"Saved thumb_right, total: {len(data['thumb_right'])}")
        elif key == ord('n') and len(data["open_hand"]) < max_samples:
            data["open_hand"].append(features)
            print(f"Saved open_hand, total: {len(data['open_hand'])}")
        elif key == ord('p') and len(data["pause"]) < max_samples:
            data["pause"].append(features)
            print(f"Saved pause, total: {len(data['pause'])}")
        elif key == ord('d') and len(data["thumb_down"]) < max_samples:
            data["thumb_down"].append(features)
            print(f"Saved pause, total: {len(data['thumb_down'])}")
        elif key == ord('e') and len(data["esc"]) < max_samples:
            data["esc"].append(features)
            print(f"Saved esc, total: {len(data['esc'])}")
    if key == ord('s'):
        # spojimo sve u jedan CSV
        all_data = []
        for label, feats_list in data.items():
            for feats in feats_list:
                all_data.append(feats + [label])
        df = pd.DataFrame(all_data)
        df.to_csv("gestures_control.csv", index=False, header=False)
        print("CSV saved as gestures_controls.csv")
        break

cap.release()
cv2.destroyAllWindows()
