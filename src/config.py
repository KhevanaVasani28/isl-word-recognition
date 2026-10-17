from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent

# Add these below code to the existing config.py made during isl-character-recognition
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
 
