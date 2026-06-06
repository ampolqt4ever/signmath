import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import collections
import random
import time

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH           = "signmath_model.h5"
GESTURES             = ["one", "two", "plus", "three", "four", "five",
                        "six", "seven", "eight", "nine", "ten"]  # FIXED: matches train.py order
SEQUENCE_LENGTH      = 30
LANDMARK_SIZE        = 126
CONFIDENCE_THRESHOLD = 0.85
SMOOTH_WINDOW        = 5
CORRECT_HOLD_SEC     = 1.5   # pause after correct before moving on
# ──────────────────────────────────────────────────────────────────────────────

GESTURE_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "plus": None,
}

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.4,
)

EMPTY_HAND = np.zeros(63)

# ─── PHASES ───────────────────────────────────────────────────────────────────
# Each question walks through: sign A → sign + → sign B → sign answer
PHASE_LIST = ["number1", "operator", "number2", "answer"]


def extract_landmarks(results):
    if not results.multi_hand_landmarks:
        return None
    sorted_hands = sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x)
    hand0 = np.array([[lm.x, lm.y, lm.z] for lm in sorted_hands[0].landmark]).flatten()
    hand1 = np.array([[lm.x, lm.y, lm.z] for lm in sorted_hands[1].landmark]).flatten() \
            if len(sorted_hands) >= 2 else EMPTY_HAND
    return np.concatenate([hand0, hand1])


def num_to_word(n):
    words = {1:"ONE",2:"TWO",3:"THREE",4:"FOUR",5:"FIVE",
             6:"SIX",7:"SEVEN",8:"EIGHT",9:"NINE",10:"TEN"}
    return words.get(n, str(n))


def generate_question():
    a = random.randint(1, 9)
    b = random.randint(1, 10 - a)
    return a, "+", b, a + b


def phase_target(question, phase):
    """What gesture should the user show for this phase."""
    a, op, b, answer = question
    if phase == "number1":   return num_to_word(a)
    if phase == "operator":  return "PLUS"
    if phase == "number2":   return num_to_word(b)
    if phase == "answer":    return num_to_word(answer)


def phase_matches(prediction, question, phase):
    """Check if the prediction satisfies the current phase."""
    a, op, b, answer = question
    if phase == "number1":
        return GESTURE_TO_NUM.get(prediction) == a
    if phase == "operator":
        return prediction == "plus"
    if phase == "number2":
        return GESTURE_TO_NUM.get(prediction) == b
    if phase == "answer":
        return GESTURE_TO_NUM.get(prediction) == answer


def draw_timer_ring(frame, cx, cy, radius, fraction, color, thickness=7):
    cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, 360, (60,60,60), thickness)
    if fraction > 0:
        cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0,
                    int(360 * fraction), color, thickness)


def draw_start_screen(frame, score, total):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    title = "SignMath Quiz"
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)
    cv2.putText(frame, title, ((w-tw)//2, h//2 - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,220,50), 3)

    rules = [
        "Sign each part of the equation",
        f"10 seconds per gesture",
        f"3 attempts per gesture",
        "",
        "Press  SPACE  to start",
        "Press  Q  to quit",
    ]
    for i, line in enumerate(rules):
        color = (0, 220, 120) if "SPACE" in line else (200, 200, 200)
        size  = 0.9 if "SPACE" in line else 0.7
        (lw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, size, 2)
        cv2.putText(frame, line, ((w-lw)//2, h//2 + i*38),
                    cv2.FONT_HERSHEY_SIMPLEX, size, color, 2)

    if total > 0:
        prev = f"Last session: {score}/{total}"
        (pw, _), _ = cv2.getTextSize(prev, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
        cv2.putText(frame, prev, ((w-pw)//2, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (140,140,140), 1)


def draw_game_ui(frame, question, phase, prediction, confidence,
                 score, total, status, status_color,
                 correct_time, seq_len, pred_time):

    h, w = frame.shape[:2]
    a, op, b, answer = question

    # ── Header ──────────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 72), (20, 20, 20), -1)
    cv2.putText(frame, f"Score: {score} / {total}", (16, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 220, 50), 2)

    # ── Question ─────────────────────────────────────────────────────────────
    q_text = f"{a}  +  {b}  =  ?"
    fs, th = 2.6, 4
    (tw, _), _ = cv2.getTextSize(q_text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    cv2.putText(frame, q_text, ((w-tw)//2 + 2, 162),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0,0,0), th+2)
    cv2.putText(frame, q_text, ((w-tw)//2, 160),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), th)

    # ── Phase instruction ────────────────────────────────────────────────────
    target = phase_target(question, phase)
    if phase == "answer":
        inst = f"Now sign the answer:  {target}"
        ic   = (100, 255, 140)
    elif phase == "operator":
        inst = f"Sign the operator:  {target}"
        ic   = (255, 180, 50)
    else:
        inst = f"Sign:  {target}"
        ic   = (100, 220, 255)

    (iw, _), _ = cv2.getTextSize(inst, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(frame, inst, ((w-iw)//2, 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, ic, 2)

    # ── Buffer bar ───────────────────────────────────────────────────────────
    bw = w - 20
    cv2.rectangle(frame, (10, 110), (10+bw, 118), (40,40,40), -1)
    cv2.rectangle(frame, (10, 110),
                  (10 + int(bw * min(seq_len, SEQUENCE_LENGTH) / SEQUENCE_LENGTH), 118),
                  (100, 180, 255), -1)
    cv2.putText(frame, f"Frames: {seq_len} / {SEQUENCE_LENGTH}", (16, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)

    # ── Status ───────────────────────────────────────────────────────────────
    if status:
        (sw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        sy = h - 110
        cv2.putText(frame, status, ((w-sw)//2 + 2, sy+2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0), 5)
        cv2.putText(frame, status, ((w-sw)//2, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, status_color, 3)

        if correct_time is not None:
            elapsed  = time.time() - correct_time
            fraction = min(elapsed / CORRECT_HOLD_SEC, 1.0)
            bw2      = 260
            bx       = (w - bw2) // 2
            by       = sy + 14
            cv2.rectangle(frame, (bx, by), (bx+bw2, by+8), (60,60,60), -1)
            cv2.rectangle(frame, (bx, by),
                          (bx+int(bw2*fraction), by+8), (0,220,100), -1)

    # ── Bottom prediction ─────────────────────────────────────────────────────
    bar_y = h - 52
    cv2.rectangle(frame, (0, bar_y), (w, h), (20,20,20), -1)
    if prediction and confidence >= CONFIDENCE_THRESHOLD:
        cv2.rectangle(frame, (0, bar_y),
                      (int(w * confidence), h), (40,100,200), -1)
        label = f"Detected: {prediction.upper()}   {confidence*100:.0f}%   ({pred_time:.2f}s)"
    else:
        label = "Show your hand..."
    cv2.putText(frame, label, (16, h-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)

    # Hand count
    hc = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
    cv2.putText(frame, f"Hands: {hc}", (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160,160,160), 1)


# ─── LOAD MODEL & NORMALIZATION STATS ─────────────────────────────────────────
print("Loading model...")
model = load_model(MODEL_PATH)
print(f"Model loaded.\nGestures: {GESTURES}\n")

# Load normalization statistics
NORM_MEAN = np.load("norm_mean.npy")
NORM_STD = np.load("norm_std.npy")

# ─── GLOBAL STATE ─────────────────────────────────────────────────────────────
sequence     = collections.deque(maxlen=SEQUENCE_LENGTH)
recent_preds = collections.deque(maxlen=SMOOTH_WINDOW)
prediction   = None
confidence   = 0.0

score        = 0
total        = 0
question     = generate_question()

phase_idx    = 0
phase        = PHASE_LIST[phase_idx]

status       = ""
status_color = (255,255,255)
correct_time = None
waiting_next = False
pred_start_time = None

# ─── SCREENS ──────────────────────────────────────────────────────────────────
SCREEN_START = "start"
SCREEN_GAME  = "game"
screen       = SCREEN_START


def begin_gesture():
    """Reset buffer for a fresh attempt."""
    global pred_start_time, sequence, recent_preds, prediction, status
    pred_start_time = time.time()
    sequence.clear()
    recent_preds.clear()
    prediction = None
    status     = ""


def next_phase():
    """Advance to the next phase of the current question."""
    global phase_idx, phase, status, status_color
    phase_idx += 1
    phase      = PHASE_LIST[phase_idx]
    status     = ""
    status_color = (255,255,255)
    begin_gesture()


def next_question():
    """Load a fresh question and reset all state."""
    global question, phase_idx, phase
    global status, status_color, correct_time, waiting_next
    question     = generate_question()
    phase_idx    = 0
    phase        = PHASE_LIST[phase_idx]
    status       = ""
    status_color = (255,255,255)
    correct_time = None
    waiting_next = False
    begin_gesture()


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Press SPACE on the start screen to begin. Q/ESC to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

    # ════════════════════════════════════════════════════════════════════════
    if screen == SCREEN_START:
        draw_start_screen(frame, score, total)

    # ════════════════════════════════════════════════════════════════════════
    elif screen == SCREEN_GAME:

        # ── Auto-advance after correct gesture ──────────────────────────────
        if waiting_next and correct_time is not None:
            if time.time() - correct_time >= CORRECT_HOLD_SEC:
                next_question()

        # ── Landmarks & inference ─────────────────────────────────────────
        landmarks = extract_landmarks(results)
        if landmarks is not None:
            sequence.append(landmarks)

        pred_time = time.time() - pred_start_time if pred_start_time else 0

        if len(sequence) == SEQUENCE_LENGTH and not waiting_next:
            # Normalize sequence using training statistics
            seq_array = np.array(sequence)
            seq_normalized = (seq_array - NORM_MEAN) / (NORM_STD + 1e-8)
            input_data = np.expand_dims(seq_normalized, axis=0)
            probs      = model.predict(input_data, verbose=0)[0]
            pred_idx   = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

            recent_preds.append(pred_idx)
            smoothed_idx = max(set(recent_preds), key=list(recent_preds).count)
            prediction   = GESTURES[smoothed_idx]

            if confidence >= CONFIDENCE_THRESHOLD:
                if phase_matches(prediction, question, phase):
                    # ── Correct gesture ──────────────────────────────────
                    if phase == "answer":
                        score       += 1
                        total       += 1
                        _, _, _, ans = question
                        status       = f"CORRECT!   Answer = {ans}"
                        status_color = (0, 220, 80)
                        correct_time = time.time()
                        waiting_next = True
                        print(f"  ✓ CORRECT [{prediction}]  {score}/{total}  ({pred_time:.2f}s)")
                    else:
                        tgt          = phase_target(question,
                                                    PHASE_LIST[phase_idx + 1])
                        status       = f"✓ {prediction.upper()}! Now sign: {tgt}"
                        status_color = (0, 200, 80)
                        print(f"  ✓ phase={phase} [{prediction}]  ({pred_time:.2f}s)")
                        next_phase()
                else:
                    # ── Wrong gesture ────────────────────────────────────
                    signed_val = GESTURE_TO_NUM.get(prediction, "?")
                    status       = f"Wrong: got {prediction.upper()} — try again!"
                    status_color = (0, 60, 255)

        # ── Draw game UI ─────────────────────────────────────────────────
        draw_game_ui(frame, question, phase, prediction, confidence,
                     score, total, status, status_color,
                     correct_time, len(sequence), pred_time)

    cv2.imshow("SignMath — Quiz", frame)

    key = cv2.waitKey(1) & 0xFF

    if key in (27, ord('q'), ord('Q')):
        break

    if screen == SCREEN_START:
        if key == ord(' '):
            screen = SCREEN_GAME
            score  = 0
            total  = 0
            next_question()
            print("Game started!\n")

    elif screen == SCREEN_GAME:
        if key == ord('n'):          # manual skip
            total += 1
            next_question()
            print(f"  → Skipped  {score}/{total}")
        if key == ord('m'):          # back to menu
            screen = SCREEN_START

cap.release()
cv2.destroyAllWindows()

print(f"\n{'═'*35}")
print(f"  Final Score : {score} / {total}")
if total > 0:
    print(f"  Accuracy    : {score/total*100:.1f}%")
print(f"{'═'*35}\n")