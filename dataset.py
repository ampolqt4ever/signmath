import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GESTURES = ["one", "two", "plus", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
SEQUENCE_LENGTH = 30        # frames per sample
SAMPLES_PER_CLASS = 50      # samples per gesture
DATASET_DIR = "dataset"
LANDMARK_SIZE = 126         # always 126: 2 hands × 21 landmarks × 3 (x,y,z)
                            # if only 1 hand visible, second hand = zeros

# How many hands each gesture needs (used for the warning overlay only)
GESTURE_HANDS = {
    "one":  1,
    "two":  1,
    "plus": 2,
    "three": 1,
    "four":  1,
    "five":  1,
    "six":   1,
    "seven": 1,
    "eight": 1,
    "nine":  1,
    "ten":   1,
}
# ──────────────────────────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.4,
)

EMPTY_HAND = np.zeros(63)  # placeholder when a hand isn't detected


def extract_landmarks(results, required_hands=1):
    """
    Always returns a (126,) array — 2 hands worth of landmarks.
    - If 0 hands detected        → return None (skip frame)
    - If 1 hand but need 1       → hand0 + zeros
    - If 1 hand but need 2       → return None (skip frame, wait for both)
    - If 2 hands detected        → hand0 (left) + hand1 (right), sorted by wrist x
    """
    if not results.multi_hand_landmarks:
        return None

    detected = results.multi_hand_landmarks

    if len(detected) < required_hands:
        return None  # not enough hands yet, keep waiting

    # Sort hands left-to-right by wrist x coordinate for consistency
    sorted_hands = sorted(detected, key=lambda h: h.landmark[0].x)

    hand0 = np.array([[lm.x, lm.y, lm.z] for lm in sorted_hands[0].landmark]).flatten()

    if len(sorted_hands) >= 2:
        hand1 = np.array([[lm.x, lm.y, lm.z] for lm in sorted_hands[1].landmark]).flatten()
    else:
        hand1 = EMPTY_HAND  # one-hand gesture, pad second slot with zeros

    return np.concatenate([hand0, hand1])  # always (126,)


def draw_progress_bar(frame, current, total, y=80):
    h, w = frame.shape[:2]
    bar_w = w - 80
    filled = int(bar_w * current / total)
    cv2.rectangle(frame, (40, y), (40 + bar_w, y + 16), (60, 60, 60), -1)
    cv2.rectangle(frame, (40, y), (40 + filled, y + 16), (0, 220, 100), -1)
    cv2.rectangle(frame, (40, y), (40 + bar_w, y + 16), (120, 120, 120), 1)


def countdown(cap, seconds=3):
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        remaining = seconds - int(time.time() - start)
        if remaining <= 0:
            break
        cv2.putText(frame, f"Starting in {remaining}...", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 200, 255), 2)
        cv2.putText(frame, "Get your hand(s) ready", (40, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
        cv2.imshow("Gesture Recorder", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            return False
    return True


def wait_for_space(cap, gesture, sample, total_samples):
    req = GESTURE_HANDS[gesture]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Gesture: {gesture.upper()}", (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 180), 2)
        cv2.putText(frame, f"Sample {sample + 1} / {total_samples}", (40, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 1)
        cv2.putText(frame, f"Needs {req} hand(s)", (40, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 200, 255), 1)
        cv2.putText(frame, "SPACE to record  |  ESC to quit", (40, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 120, 120), 1)

        cv2.imshow("Gesture Recorder", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            return True
        if key == 27:
            return False
    return False


def record_sample(cap, gesture, sample):
    sequence = []
    req = GESTURE_HANDS[gesture]

    while len(sequence) < SEQUENCE_LENGTH:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        landmarks = extract_landmarks(results, required_hands=req)

        # Draw skeletons for all detected hands
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        if landmarks is not None:
            sequence.append(landmarks)

        # UI
        status = f"Recording: {gesture.upper()}  |  Sample {sample + 1} / {SAMPLES_PER_CLASS}"
        cv2.putText(frame, status, (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
        cv2.putText(frame, f"Frames: {len(sequence)} / {SEQUENCE_LENGTH}", (40, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        draw_progress_bar(frame, len(sequence), SEQUENCE_LENGTH, y=110)

        detected_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        if detected_count < req:
            msg = "Show both hands!" if req == 2 else "No hand detected — show your hand!"
            cv2.putText(frame, msg, (40, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)

        cv2.imshow("Gesture Recorder", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            return None

    return np.array(sequence)  # shape: (30, 126)


def main():
    for gesture in GESTURES:
        os.makedirs(os.path.join(DATASET_DIR, gesture), exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    print(f"\nGestures to record : {GESTURES}")
    print(f"Samples per gesture: {SAMPLES_PER_CLASS}")
    print(f"Frames per sample  : {SEQUENCE_LENGTH}")
    print(f"Landmark size      : {LANDMARK_SIZE} (2 hands, zero-padded)")
    print(f"Total samples      : {len(GESTURES) * SAMPLES_PER_CLASS}\n")
    print("Controls: SPACE = start sample  |  ESC = quit\n")

    try:
        for gesture in GESTURES:
            print(f"\n{'─'*40}")
            print(f"  Gesture : {gesture.upper()}  ({GESTURE_HANDS[gesture]} hand(s) required)")
            print(f"{'─'*40}")

            if not countdown(cap, seconds=3):
                print("Aborted.")
                break

            for sample in range(SAMPLES_PER_CLASS):
                save_path = os.path.join(DATASET_DIR, gesture, f"{sample}.npy")
                if os.path.exists(save_path):
                    print(f"  [{gesture}] sample {sample:02d} already exists — skipping")
                    continue

                if not wait_for_space(cap, gesture, sample, SAMPLES_PER_CLASS):
                    print("Aborted.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                sequence = record_sample(cap, gesture, sample)

                if sequence is None:
                    print("Aborted mid-recording.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                np.save(save_path, sequence)
                print(f"  [{gesture}] sample {sample:02d} saved — shape {sequence.shape}")

            print(f"\n  Done with '{gesture}'!")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # ─── Validation ──────────────────────────────────────────────────────────
    print("\n" + "═" * 40)
    print("  VALIDATION")
    print("═" * 40)
    all_ok = True
    for gesture in GESTURES:
        gesture_dir = os.path.join(DATASET_DIR, gesture)
        files = sorted([f for f in os.listdir(gesture_dir) if f.endswith(".npy")])
        bad = []
        for f in files:
            data = np.load(os.path.join(gesture_dir, f))
            if data.shape != (SEQUENCE_LENGTH, LANDMARK_SIZE):
                bad.append((f, data.shape))
        if bad:
            all_ok = False
            print(f"  [{gesture}] {len(files)} files — BAD shapes: {bad}")
        else:
            print(f"  [{gesture}] {len(files)} files — all ({SEQUENCE_LENGTH}, {LANDMARK_SIZE}) ✓")

    if all_ok:
        print("\n  Dataset looks good. Ready for training!\n")
    else:
        print("\n  Some files have wrong shapes — re-record those samples.\n")


if __name__ == "__main__":
    main()