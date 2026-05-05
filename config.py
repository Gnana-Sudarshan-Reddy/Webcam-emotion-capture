"""
Configuration file for Real-Time Face Emotion Recognition System.
Contains all hyperparameters, paths, and settings.
"""

import os

# ─────────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Create directories if they don't exist
for d in [MODEL_DIR, DATA_DIR, RESULTS_DIR, ASSETS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────
# Emotion Labels
# ─────────────────────────────────────────────
EMOTION_LABELS = [
    "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"
]

NUM_CLASSES = len(EMOTION_LABELS)

# Color mapping for each emotion (BGR format for OpenCV)
EMOTION_COLORS = {
    "Angry":    (0, 0, 255),       # Red
    "Disgust":  (0, 140, 255),     # Dark Orange
    "Fear":     (0, 255, 255),     # Yellow
    "Happy":    (0, 255, 0),       # Green
    "Sad":      (255, 0, 0),       # Blue
    "Surprise": (255, 0, 255),     # Magenta
    "Neutral":  (200, 200, 200),   # Gray
}

# Emoji mapping for emotions
EMOTION_EMOJIS = {
    "Angry":    "😠",
    "Disgust":  "🤢",
    "Fear":     "😨",
    "Happy":    "😄",
    "Sad":      "😢",
    "Surprise": "😲",
    "Neutral":  "😐",
}

# ─────────────────────────────────────────────
# Model Hyperparameters
# ─────────────────────────────────────────────
IMG_SIZE = 48                       # FER-2013 image size
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 1)  # Grayscale
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.25
DENSE_DROPOUT_RATE = 0.5

# Model architecture
CONV_FILTERS = [32, 64, 128, 256]  # Filters per conv block
DENSE_UNITS = [512, 256]           # Dense layer units

# ─────────────────────────────────────────────
# Data Augmentation Parameters
# ─────────────────────────────────────────────
AUGMENTATION_CONFIG = {
    "rotation_range": 15,
    "width_shift_range": 0.15,
    "height_shift_range": 0.15,
    "shear_range": 0.15,
    "zoom_range": 0.15,
    "horizontal_flip": True,
    "fill_mode": "nearest",
}

# ─────────────────────────────────────────────
# Face Detection Settings
# ─────────────────────────────────────────────
HAAR_CASCADE_PATH = os.path.join(
    ASSETS_DIR, "haarcascade_frontalface_default.xml"
)
# Haar Cascade parameters
HAAR_SCALE_FACTOR = 1.3
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_SIZE = (30, 30)

# DNN Face Detector (Caffe model)
DNN_PROTOTXT = os.path.join(ASSETS_DIR, "deploy.prototxt")
DNN_CAFFEMODEL = os.path.join(ASSETS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
DNN_CONFIDENCE_THRESHOLD = 0.5

# ─────────────────────────────────────────────
# Webcam Settings
# ─────────────────────────────────────────────
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS_SMOOTHING = 0.9  # Exponential moving average factor

# ─────────────────────────────────────────────
# Model Paths
# ─────────────────────────────────────────────
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model_best.keras")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model_final.keras")
TRAINING_HISTORY_PATH = os.path.join(RESULTS_DIR, "training_history.json")
