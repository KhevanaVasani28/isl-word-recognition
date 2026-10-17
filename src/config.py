from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset
DATASET_PATH = BASE_DIR / "datasets" / "static_gestures"

# # CSV Folder
CSV_FOLDER = BASE_DIR / "csv"

# # Models Folder
MODEL_FOLDER = BASE_DIR / "models"

# Results Folder
RESULT_FOLDER = BASE_DIR / "results"

# Output CSV
LANDMARK_CSV = CSV_FOLDER / "landmarks_extracted.csv"

# Image Size (after letterboxing and resizing)
TARGET_SIZE = (224, 224)

# MediaPipe Settings
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5

# Preprocessed Files

X_TRAIN = CSV_FOLDER / "X_train.npy"
X_TEST = CSV_FOLDER / "X_test.npy"

Y_TRAIN = CSV_FOLDER / "y_train.npy"
Y_TEST = CSV_FOLDER / "y_test.npy"

# ==========================
# Trained Models and Label Encoder
# ==========================
LABEL_ENCODER = MODEL_FOLDER / "label_encoder.pkl"

RF_MODEL = MODEL_FOLDER / "random_forest_model.pkl"

# ==========================
# Results
# ==========================

CLASSIFICATION_REPORT_RF = RESULT_FOLDER / "rf_classification_report.txt"

RF_CONFUSION_MATRIX = RESULT_FOLDER / "rf_confusion_matrix.png"

#++++++++++++++++++++++++++++++++++++++++++++++++++

# Add these to your existing config.py after the existing configurations

# ===========================
# Data Augmentation Settings
# ===========================
AUGMENTATION_SETTINGS = {
    'rotation_range': 15,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'zoom_range': 0.1,
    'horizontal_flip': False,  # For hand gestures, flip might change meaning
    'brightness_range': [0.8, 1.2],
}

# ==========================
# Cross-validation Settings
# ==========================
CV_FOLDS = 3
CV_SCORING = 'accuracy'

# ==========================
# Hyperparameter Tuning
# ==========================
HYPERPARAMETER_GRID = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
}

# ==========================
# Confidence Threshold
# ==========================
CONFIDENCE_THRESHOLD = 0.6

# ==========================
# Augmented Data Paths
# ==========================
AUGMENTED_CSV = CSV_FOLDER / "landmarks_augmented.csv"




# ==========================
# Dynamic Word Gesture Settings
# ==========================
DYNAMIC_DATASET_PATH = BASE_DIR / "datasets" / "dynamic_gestures"

# Folder to save extracted video sequences
DYNAMIC_SEQUENCE_FOLDER = CSV_FOLDER / "dynamic_sequences"
DYNAMIC_SEQUENCE_FOLDER.mkdir(parents=True, exist_ok=True)

# Preprocessed files (sequences)
DYNAMIC_X_TRAIN = CSV_FOLDER / "dynamic_X_train.npy"
DYNAMIC_X_TEST  = CSV_FOLDER / "dynamic_X_test.npy"
DYNAMIC_Y_TRAIN = CSV_FOLDER / "dynamic_y_train.npy"
DYNAMIC_Y_TEST  = CSV_FOLDER / "dynamic_y_test.npy"

# Label encoder for words
DYNAMIC_LABEL_ENCODER = MODEL_FOLDER / "dynamic_label_encoder.pkl"

# Trained LSTM model
LSTM_MODEL = MODEL_FOLDER / "lstm_model.keras"
LSTM_TRAIN_HISTORY = RESULT_FOLDER / "lstm_training_history.png"
LSTM_CONFUSION_MATRIX = RESULT_FOLDER / "lstm_confusion_matrix.png"
LSTM_CLASSIFICATION_REPORT = RESULT_FOLDER / "lstm_classification_report.txt"

# Sequence settings
SEQUENCE_LENGTH = 90             # frames per video
FEATURES_PER_FRAME = 128          # 2 hands × (1 type + 63 landmarks)
DYNAMIC_CONFIDENCE_THRESHOLD = 0.6


# Frame sampling strategy
SAMPLE_MODE = "uniform"           # 'uniform' or 'fps'

# LSTM Training Hyperparameters
LSTM_EPOCHS = 80
LSTM_BATCH_SIZE = 32
LSTM_LEARNING_RATE = 0.001
LSTM_VALIDATION_SPLIT = 0.2
LSTM_EARLY_STOPPING_PATIENCE = 8
LSTM_DROPOUT = 0.3

# Data Augmentation
DYNAMIC_AUGMENTATION_FACTOR = 4   # produce 4 extra sequences per video

DYNAMIC_CONFIDENCE_THRESHOLD = 0.6
 