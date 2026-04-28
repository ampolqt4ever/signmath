import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import collections

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH      = "signmath_model.h5"
GESTURES        = ["one", "two", "plus", "three", "four", "five",
                   "six", "seven", "eight", "nine", "ten"]
SEQUENCE_LENGTH = 30
LANDMARK_SIZE   = 126
CONFIDENCE_THRESHOLD = 0.85   # only show prediction if confidence >= this
# ──────────────────────────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.4,
)

EMPTY_HAND = np.zeros(63)


def extract_landmarks(results):
    """Same logic as recording — always returns (126,) or None."""
    if not results.multi_hand_landmarks:
        return None

    detected = results.multi_hand_landmarks

    sorted_hands = sorted(detected, key=lambda h: h.landmark[0].x)

    hand0 = np.array([[lm.x, lm.y, lm.z] for lm in sorted_hands[0].landmark]).flatten()

    if len(sorted_hands) >= 2:
        hand1 = np.array([[lm.x, lm.y, lm.z] for lm in sorted_hands[1].landmark]).flatten()
    else:
        hand1 = EMPTY_HAND

    return np.concatenate([hand0, hand1])


def draw_prediction_bar(frame, gesture, confidence):
    """Draw a confidence bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_h = 50
    bar_y = h - bar_h

    # Background
    cv2.rectangle(frame, (0, bar_y), (w, h), (20, 20, 20), -1)

    # Confidence fill
    fill_w = int(w * confidence)
    color = (0, 220, 100) if confidence >= CONFIDENCE_THRESHOLD else (0, 140, 255)
    cv2.rectangle(frame, (0, bar_y), (fill_w, h), color, -1)

    # Text
    label = f"{gesture.upper()}  {confidence * 100:.1f}%"
    cv2.putText(frame, label, (16, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)


def draw_sequence_bar(frame, current, total, y=10):
    """Small bar at the top showing how full the buffer is."""
    h, w = frame.shape[:2]
    bar_w = w - 20
    filled = int(bar_w * current / total)
    cv2.rectangle(frame, (10, y), (10 + bar_w, y + 8), (40, 40, 40), -1)
    cv2.rectangle(frame, (10, y), (10 + filled, y + 8), (100, 180, 255), -1)


# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
print("Loading model...")
model = load_model(MODEL_PATH)
print(f"Model loaded — input shape: {model.input_shape}")

# ─── STATE ────────────────────────────────────────────────────────────────────
sequence   = collections.deque(maxlen=SEQUENCE_LENGTH)  # sliding window
prediction = None
confidence = 0.0

# Smoothing: keep last N predictions and pick the majority
SMOOTH_WINDOW = 5
recent_preds = collections.deque(maxlen=SMOOTH_WINDOW)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("\nRunning — press Q or ESC to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Draw skeleton
    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

    # Extract landmarks and fill buffer
    landmarks = extract_landmarks(results)
    if landmarks is not None:
        sequence.append(landmarks)

    # Buffer fill indicator (top bar)
    draw_sequence_bar(frame, len(sequence), SEQUENCE_LENGTH, y=10)

    # Run inference once buffer is full
    if len(sequence) == SEQUENCE_LENGTH:
        input_data = np.expand_dims(np.array(sequence), axis=0)  # (1, 30, 126)
        probs      = model.predict(input_data, verbose=0)[0]
        pred_idx   = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        recent_preds.append(pred_idx)

        # Majority vote across recent predictions for stability
        smoothed_idx = max(set(recent_preds), key=list(recent_preds).count)
        prediction   = GESTURES[smoothed_idx]

        draw_prediction_bar(frame, prediction, confidence)

        # Console log
        print(f"  {prediction.upper():>6}  {confidence * 100:.1f}%  "
              f"  raw: {[f'{p*100:.0f}%' for p in probs]}")

    elif prediction is not None:
        # Keep showing last prediction while buffer refills
        draw_prediction_bar(frame, prediction, confidence)

    # Hand count indicator
    hand_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
    cv2.putText(frame, f"Hands: {hand_count}", (16, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    cv2.imshow("SignMath — Live", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")