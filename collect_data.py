import cv2
import mediapipe as mp
import csv
import os

# =====================
# MediaPipe Setup
# =====================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =====================
# Dataset Setup
# =====================
csv_file = "dataset.csv"

if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)

        header = []

        for i in range(21):
            header.extend([
                f"x{i}",
                f"y{i}",
                f"z{i}"
            ])

        header.append("label")

        writer.writerow(header)

# =====================
# Recording Config
# =====================
current_label = "A"

recording = False
record_count = 0
target_samples = 300

# =====================
# Camera
# =====================
cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    landmarks = []

    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        for lm in hand.landmark:
            landmarks.extend([
                lm.x,
                lm.y,
                lm.z
            ])

        # =====================
        # Auto Recording
        # =====================
        if recording:

            row = landmarks.copy()
            row.append(current_label)

            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            record_count += 1

            if record_count >= target_samples:
                recording = False
                print(
                    f"✅ Finished {current_label} "
                    f"({target_samples} samples)"
                )

    # =====================
    # UI Text
    # =====================
    cv2.putText(
        frame,
        f"Label : {current_label}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    if recording:

        cv2.putText(
            frame,
            f"Recording : {record_count}/{target_samples}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.putText(
        frame,
        "A/B/C/D/E = Label | R = Record | ESC = Exit",
        (10, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow("Collect Dataset", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("a"):
        current_label = "A"

    elif key == ord("b"):
        current_label = "B"

    elif key == ord("c"):
        current_label = "C"

    elif key == ord("d"):
        current_label = "D"

    elif key == ord("e"):
        current_label = "E"

    elif key == ord("r"):

        if not recording:
            recording = True
            record_count = 0

            print(
                f"🎥 Recording {current_label} "
                f"({target_samples} samples)"
            )

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()