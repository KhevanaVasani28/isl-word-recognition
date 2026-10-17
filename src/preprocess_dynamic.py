# %% ==========================================================
# STEP 1 : Imports
# ==========================================================
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config import *
from sklearn.utils.class_weight import compute_class_weight

# %% ==========================================================
# STEP 2 : Load sequences
# ==========================================================
X = np.load(DYNAMIC_SEQUENCE_FOLDER / "X_sequences.npy")
y_labels = np.load(DYNAMIC_SEQUENCE_FOLDER / "y_labels.npy", allow_pickle=True)

print(f"Loaded X: {X.shape}")
print(f"Loaded y: {y_labels.shape}")
print(f"Classes: {len(np.unique(y_labels))}")

# %% ==========================================================
# STEP 3 : Encode labels
# ==========================================================
encoder = LabelEncoder()
y = encoder.fit_transform(y_labels)
joblib.dump(encoder, DYNAMIC_LABEL_ENCODER)
print(f"Label encoder saved → {DYNAMIC_LABEL_ENCODER}")
print(f"Classes: {list(encoder.classes_)}")

# %% ==========================================================
# STEP 4 : Split FIRST (before augmentation) — no data leakage
# ==========================================================
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nOriginal Train: {X_train_orig.shape}")
print(f"Original Test:  {X_test_orig.shape}")

# %% ==========================================================
# STEP 5 : Augmentation (temporal + spatial jitter)
# ==========================================================
def augment_sequence(seq, factor=4):
    """
    Generate augmented versions of a single sequence.
    seq shape: (60, 128)
    Return list of augmented sequences. seq shape: (60, 128)."""

    augmented = []
    T, F = seq.shape

    for _ in range(factor):
        aug = seq.copy()

        # 1) Temporal jitter: random small shift of frame indices
        shift = np.random.randint(-5, 6)
        aug = np.roll(aug, shift=shift, axis=0)

        # 2) Time warping — speed up or slow down
        if np.random.rand() < 0.5:
            # Randomly drop 5–10% of frames
            drop_ratio = np.random.uniform(0.05, 0.15)
            n_drop = int(T * drop_ratio)
            keep = np.sort(np.random.choice(T, T - n_drop, replace=False))
            aug = aug[keep]
            # Resample back to T
            idxs = np.linspace(0, len(aug) - 1, T).astype(int)
            aug = aug[idxs]

        # 2) Identify columns to jitter (exclude hand-type indicators at 0 and 64)
        mask = np.ones(F, dtype=bool)
        mask[0] = False
        mask[64] = False

        # 3) Gaussian noise on landmark coordinates
        noise = np.random.normal(0, 0.02, aug.shape).astype(np.float32)
        aug[:, mask] += noise[:, mask]

        # 4) Random scale (±5%)
        scale = np.random.uniform(0.9, 1.1)
        aug[:, mask] *= scale

        # 5) Random small rotation in xy-plane (frame-wise)
        angle = np.deg2rad(np.random.uniform(-15, 15))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for hand_offset in (1, 65):
            hand = aug[:, hand_offset:hand_offset + 63]
            xs = hand[:, 0::3]
            ys = hand[:, 1::3]
            new_x = cos_a * xs - sin_a * ys
            new_y = sin_a * xs + cos_a * ys
            hand[:, 0::3] = new_x
            hand[:, 1::3] = new_y

        augmented.append(aug.astype(np.float32))

    return augmented

print("\nAugmenting training set only...")
X_train_aug, y_train_aug = [], []

for i in tqdm(range(len(X_train_orig))):
    X_train_aug.append(X_train_orig[i])
    y_train_aug.append(y_train_orig[i])

    for aug_seq in augment_sequence(X_train_orig[i], factor=DYNAMIC_AUGMENTATION_FACTOR):
        X_train_aug.append(aug_seq)
        y_train_aug.append(y_train_orig[i])

X_train_aug = np.stack(X_train_aug).astype(np.float32)
y_train_aug = np.array(y_train_aug)

# Test set stays ORIGINAL — no augmentation
X_test_final = X_test_orig.astype(np.float32)
y_test_final = y_test_orig

print(f"\nFinal Train: {X_train_aug.shape}")
print(f"Final Test:  {X_test_final.shape}")
print(f"Augmentation multiplier: {len(X_train_aug)/len(X_train_orig):.2f}x")

# # Verify hand-type flags are intact (should be 0 or 1 only)
# left_types = X_train_aug[:, :, 0]
# right_types = X_train_aug[:, :, 64]

# assert np.all(np.isin(np.unique(left_types), [0.0, 1.0])), "Left hand type corrupted!"
# assert np.all(np.isin(np.unique(right_types), [0.0, 1.0])), "Right hand type corrupted!"

# print(f"✅ Hand-type flags intact")
# print(f"   Left  types: {np.unique(left_types)}")
# print(f"   Right types: {np.unique(right_types)}")

# %% ==========================================================
# STEP 6 : Save
# ==========================================================
np.save(DYNAMIC_X_TRAIN, X_train_aug)
np.save(DYNAMIC_X_TEST,  X_test_final)
np.save(DYNAMIC_Y_TRAIN, y_train_aug)
np.save(DYNAMIC_Y_TEST,  y_test_final)

print("\nSaved:")
print(f"  {DYNAMIC_X_TRAIN}")
print(f"  {DYNAMIC_X_TEST}")
print(f"  {DYNAMIC_Y_TRAIN}")
print(f"  {DYNAMIC_Y_TEST}")

# %% ==========================================================
# STEP 7 : Compute class weights for imbalanced training
# ==========================================================

classes_present = np.unique(y_train_aug)
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes_present,
    y=y_train_aug
)
class_weights = {int(c): float(w) for c, w in zip(classes_present, class_weights_array)}

joblib.dump(class_weights, MODEL_FOLDER / "dynamic_class_weights.pkl")
print("\nClass weights computed and saved.")
print(f"  Min weight: {min(class_weights.values()):.3f}")
print(f"  Max weight: {max(class_weights.values()):.3f}")