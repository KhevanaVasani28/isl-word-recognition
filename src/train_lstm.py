# %% ==========================================================
# STEP 1 : Imports
# ==========================================================
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    LSTM, Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.callbacks import (  # type: ignore
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score, precision_score, recall_score
)
from config import *

# %% ==========================================================
# STEP 2 : Load preprocessed sequences
# ==========================================================
X_train = np.load(DYNAMIC_X_TRAIN)
X_test  = np.load(DYNAMIC_X_TEST)
y_train = np.load(DYNAMIC_Y_TRAIN)
y_test  = np.load(DYNAMIC_Y_TEST)

encoder = joblib.load(DYNAMIC_LABEL_ENCODER)
class_weights = joblib.load(MODEL_FOLDER / "dynamic_class_weights.pkl")
num_classes = len(encoder.classes_)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Classes: {num_classes}")
print(encoder.classes_)

# %% ==========================================================
# STEP 3 : One-hot encode labels
# ==========================================================
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat  = to_categorical(y_test,  num_classes)

# %% ==========================================================
# STEP 4 : Build LSTM model
# ==========================================================
def build_model(seq_len, n_feat, n_classes):
    model = Sequential([
        Input(shape=(seq_len, n_feat)),

        LSTM(64, return_sequences=True, activation='tanh'),
        BatchNormalization(),
        Dropout(LSTM_DROPOUT),

        LSTM(128, return_sequences=True, activation='tanh'),
        BatchNormalization(),
        Dropout(LSTM_DROPOUT),

        LSTM(64, return_sequences=False, activation='tanh'),
        BatchNormalization(),
        Dropout(LSTM_DROPOUT),

        Dense(64, activation='relu'),
        Dropout(0.3),

        Dense(32, activation='relu'),

        Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=LSTM_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


model = build_model(SEQUENCE_LENGTH, FEATURES_PER_FRAME,num_classes)
model.summary()

# %% ==========================================================
# STEP 5 : Callbacks
# ==========================================================
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=LSTM_EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=str(LSTM_MODEL),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# %% ==========================================================
# STEP 6 : Train
# ==========================================================
history = model.fit(
    X_train, y_train_cat,
    validation_split=LSTM_VALIDATION_SPLIT,
    epochs=LSTM_EPOCHS,
    batch_size=LSTM_BATCH_SIZE,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# %% ==========================================================
# STEP 7 : Evaluate
# ==========================================================
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
print(f"\nTest Accuracy: {acc:.4f}")
print(f"Macro Precision: {precision:.4f}")
print(f"Macro F1:      {f1:.4f}")
print(f"Macro Recall:  {recall:.4f}")

report = classification_report(
    y_test, y_pred,
    target_names=encoder.classes_,
    zero_division=0
)
print(report)

with open(LSTM_CLASSIFICATION_REPORT, "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n ")
    f.write(f"Macro Precision: {precision:.4f}\n") 
    f.write(f"Macro Recall: {recall:.4f}\n")
    f.write(f"Macro F1: {f1:.4f}\n\n")
    f.write(report)

# %% ==========================================================
# STEP 8 : Plots
# ==========================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Val')
axes[0].set_title('Accuracy'); axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Val')
axes[1].set_title('Loss'); axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(LSTM_TRAIN_HISTORY, dpi=150)
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(16, 14))
ConfusionMatrixDisplay(cm, display_labels=encoder.classes_).plot(
    ax=ax, xticks_rotation=90, cmap='Blues'
)
plt.title(f'LSTM Confusion Matrix — Acc: {acc:.4f}')
plt.tight_layout()
plt.savefig(LSTM_CONFUSION_MATRIX, dpi=150)
plt.show()

print(f"\nModel saved → {LSTM_MODEL}")