# %% ==========================================================
# STEP 1 : Imports
# ==========================================================
import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

from config import *
from preprocess_utils import (
    preprocess_frame, 
    extract_features_from_rgb
)

# %% ================================
# STEP 2 : Initialize MediaPipe 
# ===================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=0.5
)

# ===============================
# STEP 3 : Frame Preprocessing
# ===============================
# from preprocess_utils

# %% =======================================================
# STEP 4 : Hand ordering 
# ==========================================================
# from preprocess_utils

# %% ==========================================================
# STEP 5 : Uniform frame sampling
# ==========================================================
def sample_frames_from_video(cap, target_frames=SEQUENCE_LENGTH):
    """Read `target_frames` evenly spaced frames from the video.
    Returns list of preprocessed RGB frames (each 224x224x3).
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return []

    target_frames = min(target_frames, total_frames)
    frame_indices = np.linspace(0, total_frames - 1, target_frames).astype(int)
    
    frames = []
    last_valid = None

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            # Fallback: replicate last valid, or black frame
            if last_valid is not None:
                frames.append(last_valid)
            else:
                frames.append(np.zeros((*TARGET_SIZE, 3), dtype=np.uint8))
            continue

        frame_rgb = preprocess_frame(frame)
        # frame = cv2.resize(frame, TARGET_SIZE)
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame_rgb)
        last_valid = frame_rgb

    return frames


# %% ==========================================================
# STEP 6 : Process one video → (SEQUENCE_LENGTH, 128) array
# ==========================================================
def process_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frames = sample_frames_from_video(cap, SEQUENCE_LENGTH)
    cap.release()

    if len(frames) == 0:
        return None

    sequence = [extract_features_from_rgb(f) for f in frames]

    # Ensure exactly SEQUENCE_LENGTH
    while len(sequence) < SEQUENCE_LENGTH:
        sequence.append(sequence[-1])   # replicate last frame
    sequence = sequence[:SEQUENCE_LENGTH]

    return np.array(sequence, dtype=np.float32)   # (90, 128)


# %% ==========================================================
# STEP 7 : Walk through dataset
# ==========================================================
word_classes = sorted([
    d for d in os.listdir(DYNAMIC_DATASET_PATH)
    if os.path.isdir(DYNAMIC_DATASET_PATH / d)
])
print(f"Word Classes: {len(word_classes)}")
print(word_classes)

valid_ext = (".mp4")
sequences = []   
labels = []
skipped = 0
failed = 0

for word in tqdm(word_classes, desc="Words"):
    folder = DYNAMIC_DATASET_PATH / word
    videos = [v for v in os.listdir(folder) if v.lower().endswith(valid_ext)]

    for video_name in tqdm(videos, desc=word, leave=False):
        video_path = folder / video_name
        try:
            seq = process_video(video_path)
            if seq is None:
                skipped += 1
                continue

            sequences.append(seq)
            labels.append(word)

        except Exception as e:
            failed += 1
            print(f"Error: {video_path} → {e}")

print("\n" + "=" * 60)
print(f"Processed: {len(sequences)}")
print(f"Skipped:   {skipped}")
print(f"Failed:    {failed}")
print("=" * 60)


# %% ==========================================================
# STEP 8 : Save all sequences + labels
# ==========================================================
X = np.stack(sequences).astype(np.float32)                   
y = np.array(labels, dtype=object)

np.save(DYNAMIC_SEQUENCE_FOLDER / "X_sequences.npy", X)          # shape (2369, 90, 128)
np.save(DYNAMIC_SEQUENCE_FOLDER / "y_labels.npy", y)             # shape (2369,)

 # (total_sequences, sequence_length, features_per_frame) 
   
print(f"\nSaved X shape: {X.shape}")
print(f"Saved y shape: {y.shape}")
print(f"Location: {DYNAMIC_SEQUENCE_FOLDER}")

hands.close()

print("\nVideo landmark extraction complete...")
