# 🤟 Indian Sign Language (ISL) Gesture Recognition & English Translation System PART-2

An AI-powered system that recognizes Indian Sign Language **dynamic gestures** in real time using computer vision and machine learning, with the goal of translating signs into English text.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green.svg)
![TensorFlow](https://img.shields.io/badge/Tensorflow-2.15+-orange.svg)

---

## 📌 Overview

This project focuses on **dynamic gestures (words)**.The system uses **MediaPipe** for hand landmark detection and a **LSTM** for gesture recognition. It supports both **single-hand** and **two-hand** gestures, correctly distinguishing between **left** and **right** hands. The dataset used here is a video dataset consisting of .mp4 files.

---

## 🎯 Features

- ✅ Real-time gesture recognition via webcam
- ✅ MediaPipe-based hand landmark extraction (21 landmarks per hand, 3D)
- ✅ Left-hand and right-hand detection with proper ordering
- ✅ Data augmentation for improved model generalization
- ✅ Confidence scores for predictions
- ✅ Prediction smoothing for stable output
- ✅ Supports up to 2 hands simultaneously

---

## 🏗️ Project Structure

```
isl-word-recognition/
│
├── src/
│   ├── config.py                        # Central configuration (paths, constants)
|   ├── preprocess_utils.py              # General functions 
│   ├── extract_landmarks_video.py       # Extract landmarks from dataset images
│   ├── preprocess_dynamic.py            # Preprocess + augment + split data
│   ├── train_lstm.py                    # Trained LSTM 
│   └── realtime_exp_dynamic.py          # Real-time webcam prediction
│
├── datasets/                      
│   ├── dynamic_gestures            # Raw gesture video dataset (not uploaded)
|       ├── Hello/      
|       ├── Good Morning/
|       ├── Good Evening/
|       ├── Thank You/
|       └──  ...
|     
├── csv/                           # Extracted landmarks + npy files
├── models/                        # Trained model + label encoder
├── results/                       # Confusion matrix, reports
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ How It Works

### **Pipeline**

```
Dataset Videos Gestures (Eg. Hello, Good Morning, Good Evening)
        ↓
extract_landmarks_video.py   → Detect hands via MediaPipe, extract 21 landmarks/hand for each frame
        ↓
preprocess_dynamic.py       → Clean, encode, augment, split (train/test)
        ↓
train_lstm.py               → Train LSTM, confusion matix and classification report
        ↓
realtime_prediction.py     → Real-time webcam prediction
```

### **Landmark Representation**

Each frame is converted into a **128-dimensional feature vector**:

| Hand      | Feature Type | Size |
|-----------|--------------|------|
| Left      | type (0) + 21 landmarks × 3 (x,y,z) | 64 |
| Right     | type (1) + 21 landmarks × 3 (x,y,z) | 64 |
| **Total** | | **128** |

- Coordinates are **normalized** relative to the wrist (landmark 0) for translation invariance.
- If a hand is not detected, its slots are filled with **zeros** (with type preserved so the model knows which hand is missing).

---

## 🚀 Getting Started

### **1. Clone the Repository**

```bash
git clone https://github.com/KhevanaVasani28/isl-word-recognition.git
cd isl-word-recognition
```

### **2. Create a Virtual Environment (Recommended)**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Prepare Dataset**

Place your gesture image folders inside:

```
datasets/dynamic_gestures/
├── Hello/      
├── Good Morning/
├── Good Evening/
├── Thank You/
└──  ...
```

### **5. Run the Pipeline (in order)**

```bash
# Step 1: Run the general configurations (path, constants)
python src/config.py

# Step 2: Run the general function utilities
python src/preprocess_utils.py

# Step 3: Extract landmarks from video for each frame
python src/extract_landmarks_video.py

# Step 4: Preprocess, augment, and split data
python src/preprocess_dynamic.py

# Step 5: Train the LSTM model
python src/train_lstm.py

# Step 6: Run real-time prediction
python src/realtime_exp_dynamic.py
```
---

## 🧠 Model Details

| Parameter | Value |
|-----------|-------|
| Model | LSTM |
| Data Augmentation | Rotation, scaling, translation, noise |
| Confidence Threshold | 0.6 (configurable) |
| Classes | 40 (dynamic gestures) |

---

## 📊 Results

After training, the following files are generated inside `results/`:

- `lstm_classification_report.txt` — Precision, recall, F1-score per class
- `lstm_confusion_matrix.png` — Visual confusion matrix

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **OpenCV** — frames and video processing
- **MediaPipe** — hand landmark detection
- **NumPy / Pandas** — data manipulation
- **tensorflow** — deep learning LSTM model
- **Matplotlib / Seaborn** — visualization
- **joblib** — model persistence
- **tqdm** — progress bars

---

## 🔮 Roadmap

- [x] Static gestures (A–Z, 0–9)
- [x] Dynamic gestures (words)
- [ ] Sentence-level translation
- [ ] Text-to-speech output
- [ ] Mobile app deployment

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 👤 Author

**Khevana Vasani**
- GitHub: https://github.com/KhevanaVasani28
- Email: khevanavasani@gmail.com

---

## 🙏 Acknowledgements

- [MediaPipe](https://mediapipe.dev/) by Google for hand tracking
- [Kaggle](https://www.kaggle.com/) for the ISL dataset
- Open-source community for inspiration and tools

---

## ⭐ If you find this project useful, please give it a star!
