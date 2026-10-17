# %% ==========================================================
# Real-Time Dynamic ISL Recognition — Record-then-Predict Mode
# =============================================================
import cv2
import mediapipe as mp
import numpy as np
import joblib
import tensorflow as tf
import time
from config import *
from preprocess_utils import (
    # preprocess_frame,
    extract_features_from_rgb,
)

# %% =======================================================
# STEP 1 : Load model + encoder
# ==========================================================
model = tf.keras.models.load_model(LSTM_MODEL)
encoder = joblib.load(DYNAMIC_LABEL_ENCODER)

print("=" * 60)
print("DYNAMIC ISL RECOGNITION — Record-then-Predict")
print("=" * 60)
print(f"Model: {LSTM_MODEL}")
print(f"Classes ({len(encoder.classes_)}): {list(encoder.classes_)}")
print(f"Sequence length: {SEQUENCE_LENGTH} frames")
print("=" * 60)

# %% =======================================================
# STEP 2 : MediaPipe
# ==========================================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=0.5
)

# %% ==========================================================
# STEP 4 : Post-processing helpers
# ==========================================================
def smooth_sequence(seq, window=3):
    """Moving-average smoothing along time axis."""
    if window <= 1:
        return seq
    pad = window // 2
    seq_pad = np.pad(seq, ((pad, pad), (0, 0)), mode='edge')
    kernel = np.ones(window) / window
    out = np.stack([
        np.convolve(seq_pad[:, f], kernel, mode='valid')
        for f in range(seq.shape[1])
    ], axis=1)
    return out.astype(np.float32)

def trim_idle_frames(seq):
    """Remove leading/trailing frames where no hands are present."""
    left_empty  = np.all(seq[:, 1:64] == 0, axis=1)
    right_empty = np.all(seq[:, 65:128] == 0, axis=1)
    both_empty  = left_empty & right_empty

    non_empty = np.where(~both_empty)[0]
    if len(non_empty) == 0:
        return seq

    return seq[non_empty[0]:non_empty[-1] + 1]

def resample_to_length(seq, target_len):
    """Uniformly resample to fixed length."""
    if len(seq) == target_len:
        return seq
    idxs = np.linspace(0, len(seq) - 1, target_len).astype(int)
    return seq[idxs]

# TESTING: Test-time augmentation (TTA) for dynamic sequences
def predict_with_tta(model, seq, encoder, n_augments=5):
    """Average prediction across small perturbations."""
    all_probas = []
    all_probas.append(model.predict(
        np.expand_dims(seq, 0), verbose=0
    )[0])

    for _ in range(n_augments - 1):
        aug = seq.copy()
        shift = np.random.randint(-2, 3)
        aug = np.roll(aug, shift=shift, axis=0)

        mask = np.ones(aug.shape[1], dtype=bool)
        mask[0] = False
        mask[64] = False
        noise = np.random.normal(0, 0.005, aug.shape).astype(np.float32)
        aug[:, mask] += noise[:, mask]

        all_probas.append(model.predict(
            np.expand_dims(aug.astype(np.float32), 0), verbose=0
        )[0])

    avg = np.mean(all_probas, axis=0)
    pred_class = int(np.argmax(avg))
    confidence = float(np.max(avg))
    word = encoder.inverse_transform([pred_class])[0]
    return word, confidence, avg

# %% ==========================================================
# STEP 4 : Settings
# %% ==========================================================
RECORD_SECONDS = 5
PREDICT_DISPLAY_SECONDS = 3
TARGET_FPS = 30
USE_TTA = True

# %% ==========================================================
# STEP 5 : Webcam setup
# %% ==========================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

WINDOW = "Dynamic ISL Recognition"
cv2.namedWindow(WINDOW)

state = "IDLE"          # IDLE | RECORDING | SHOWING
recorded_frames = []    # preprocessed RGB frames captured so far
record_start_time = 0.0
prediction_text = ""
prediction_confidence = 0.0
prediction_display_until = 0.0

print("\nPress SPACE to start recording.")
print("Press Q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.flip(frame, 1)
    display = frame.copy()
    now = time.time()

    # ----------------------------------------------------------
    # State: RECORDING — collect frames for RECORD_SECONDS
    # ----------------------------------------------------------
    if state == "RECORDING":
        
        # frame_rgb = preprocess_frame(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        recorded_frames.append(frame_rgb)

        # Draw landmarks for user feedback
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(display, hl, mp_hands.HAND_CONNECTIONS)

        elapsed = now - record_start_time

        # Recording countdown
        cv2.putText(display, "RECORDING...",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3)
        cv2.putText(display,
                    f"Time left: {max(0, RECORD_SECONDS - elapsed):.1f}s",
                    (30, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
        cv2.putText(display,
                    f"Frames captured: {len(recorded_frames)}",
                    (30, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (200, 200, 200), 2)

        # End recording → process
        if elapsed >= RECORD_SECONDS:
            state = "SHOWING"
            prediction_display_until = now + PREDICT_DISPLAY_SECONDS

            # Sample SEQUENCE_LENGTH frames evenly
            total = len(recorded_frames)
            if total == 0:
                prediction_text = "No frames recorded!"
                prediction_confidence = 0.0
            else:
                idxs = np.linspace(
                    0, total - 1, SEQUENCE_LENGTH
                ).astype(int)
                sampled = [recorded_frames[i] for i in idxs]

                # Extract features for each sampled frame
                print(f"\nProcessing {len(sampled)} frames...")
                t0 = time.time()
                seq = np.stack([
                    extract_features_from_rgb(f) for f in sampled
                ]).astype(np.float32)     # (90, 128)
                extract_time = time.time() - t0
                print(f"Feature extraction: {extract_time:.2f}s")

                # 2) Smooth
                seq = smooth_sequence(seq, window=3)

                # 3) Trim idle (may shorten sequence)
                seq = trim_idle_frames(seq)

                # 4) Resample back to SEQUENCE_LENGTH
                seq = resample_to_length(seq, SEQUENCE_LENGTH)

                # 5) Predict
                t0 = time.time()
                if USE_TTA:
                    word, confidence, proba = predict_with_tta(
                        model, seq, encoder, n_augments=5
                    )
                else:
                    proba = model.predict(
                        np.expand_dims(seq, 0), verbose=0)[0]
                    pred_class = int(np.argmax(proba))
                    confidence = float(np.max(proba))
                    word = encoder.inverse_transform([pred_class])[0]
                
                print(f"Inference: {time.time()-t0:.2f}s")

                prediction_text = word
                prediction_confidence = confidence

                print(f"\n🎯 {word} ({confidence:.3f})")
                top3 = np.argsort(proba)[-3:][::-1]
                for i in top3:
                    cls = encoder.inverse_transform([i])[0]
                    print(f"   {cls:20s} {proba[i]:.3f}")

            recorded_frames = []

    # ----------------------------------------------------------
    # State: SHOWING — display prediction for a few seconds
    # ----------------------------------------------------------
    elif state == "SHOWING":
        if now >= prediction_display_until:
            state = "IDLE"
        else:
            # Color by confidence
            if prediction_confidence >= DYNAMIC_CONFIDENCE_THRESHOLD:
                color = (0, 255, 0)
            elif prediction_confidence >= 0.4:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)

            cv2.putText(display,
                        f"Word: {prediction_text}",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.4, color, 3)
            cv2.putText(display,
                        f"Confidence: {prediction_confidence:.2f}",
                        (30, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)

    # ----------------------------------------------------------
    # State: IDLE — wait for SPACE
    # ----------------------------------------------------------
    else:  # IDLE
        cv2.putText(display, "Press SPACE to record gesture",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255), 3)
        cv2.putText(display,
                    f"Will record for {RECORD_SECONDS} seconds",
                    (30, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (200, 200, 200), 2)

    # ----------------------------------------------------------
    # Footer
    # ----------------------------------------------------------
    cv2.putText(display, "Q: quit | SPACE: record",
                (30, display.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 2)

    cv2.imshow(WINDOW, display)

    # ----------------------------------------------------------
    # Key handling
    # ----------------------------------------------------------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' ') and state == "IDLE":
        state = "RECORDING"
        recorded_frames = []
        record_start_time = time.time()
        print(f"\n▶ Recording {RECORD_SECONDS}s... please sign now.")

cap.release()
cv2.destroyAllWindows()
hands.close()
print("\nStopped.")
