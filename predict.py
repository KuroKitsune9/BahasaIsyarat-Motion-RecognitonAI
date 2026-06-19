import cv2
import mediapipe as mp
import joblib
import numpy as np
import time

from pythonosc.udp_client import SimpleUDPClient

# ==========================================
# OSC -> TouchDesigner
# ==========================================
osc = SimpleUDPClient(
    "127.0.0.1",
    8000
)

# ==========================================
# Load Model
# ==========================================
model = joblib.load(
    "gesture_model.pkl"
)

# ==========================================
# MediaPipe
# ==========================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ==========================================
# Mode Mapping
# ==========================================
mode_map = {
    "A": 0,  # FIRE
    "B": 1,  # WATER
    "C": 2,  # LOVE
    "D": 3,  # NATURE
    "E": 4   # LIGHTNING
}

mode_names = {
    0: "FIRE",
    1: "WATER",
    2: "LIGHTNING",
    3: "LOVE",
    4: "NATURE"
}

selected_mode = None

last_gesture = None
gesture_start_time = 0

HOLD_TIME = 2

# ==========================================
# Hand Smoothing
# ==========================================
smooth_x = 0.5
smooth_y = 0.5

SMOOTHING = 0.2

# ==========================================
# Camera
# ==========================================
cap = cv2.VideoCapture(0)

last_save_time = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = hands.process(rgb)

    prediction = "-"

    # ======================================
    # Hand Detection
    # ======================================
    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        landmarks = []

        for lm in hand.landmark:

            landmarks.extend([
                lm.x,
                lm.y,
                lm.z
            ])

        X = np.array(
            landmarks
        ).reshape(1, -1)

        prediction = model.predict(X)[0]

        # ==================================
        # Palm Position
        # ==================================
        palm_x = hand.landmark[9].x
        palm_y = hand.landmark[9].y

        # ==================================
        # Smooth Position
        # ==================================
        smooth_x = smooth_x + (
            palm_x - smooth_x
        ) * SMOOTHING

        smooth_y = smooth_y + (
            palm_y - smooth_y
        ) * SMOOTHING

        # ==================================
        # OSC Hand Position
        # ==================================
        osc.send_message(
            "/handx",
            float(smooth_x)
        )

        osc.send_message(
            "/handy",
            float(smooth_y)
        )

        osc.send_message(
            "/gesture",
            prediction
        )

        # ==================================
        # Gesture Hold Detection
        # ==================================
        current_time = time.time()

        if prediction != last_gesture:

            last_gesture = prediction
            gesture_start_time = current_time

        else:

            elapsed = current_time - gesture_start_time

            if elapsed >= HOLD_TIME:

                if prediction in mode_map:

                    new_mode = mode_map[prediction]

                    if new_mode != selected_mode:

                        selected_mode = new_mode

                        print(
                            f"\nMODE SELECTED: {selected_mode}"
                        )

        # ==================================
        # Send Mode Continuously
        # ==================================
        if selected_mode is not None:

            osc.send_message(
                "/mode",
                selected_mode
            )

        print(
            f"Gesture: {prediction}",
            end="\r"
        )

    # ======================================
    # UI
    # ======================================
    cv2.putText(
        frame,
        f"Gesture: {prediction}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    if selected_mode is not None:

        cv2.putText(
            frame,
            f"MODE: {mode_names[selected_mode]}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "SELECT ELEMENT",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

    # ======================================
    # Save Camera Feed for TouchDesigner
    # ======================================
    now = time.time()

    if now - last_save_time > 0.1:

        cv2.imwrite(
            "camera_feed.jpg",
            frame
        )

        last_save_time = now

    # ======================================
    # Show Window
    # ======================================
    cv2.imshow(
        "Bahasa Isyarat AI",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # ESC
    if key == 27:
        break

    # Reset
    if key == ord("r"):

        selected_mode = None
        last_gesture = None

        osc.send_message(
            "/mode",
            -1
        )

        print(
            "\nMODE RESET"
        )

cap.release()
cv2.destroyAllWindows()