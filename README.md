# 😊 Real-Time Face Emotion Recognition

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv&logoColor=white)](https://opencv.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras&logoColor=white)](https://keras.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io)

> **Advanced deep learning system** that detects and classifies **7 human emotions** in real-time from webcam video using a custom **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras** and **OpenCV**.

---

## 🎯 Overview

This project implements an end-to-end **facial emotion recognition pipeline** that captures live video from a webcam, detects faces using OpenCV, and classifies emotions using a custom-trained CNN. The system operates in real-time with an intuitive HUD overlay showing emotion predictions, confidence scores, and probability distributions.

### Emotions Detected

| Emotion | Emoji | Color Code |
|---------|-------|------------|
| Angry | 😠 | 🔴 Red |
| Disgust | 🤢 | 🟠 Orange |
| Fear | 😨 | 🟡 Yellow |
| Happy | 😄 | 🟢 Green |
| Sad | 😢 | 🔵 Blue |
| Surprise | 😲 | 🟣 Magenta |
| Neutral | 😐 | ⚪ Gray |

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Webcam /    │────▶│  Face Detection  │────▶│  Preprocessing  │
│  Video Input │     │  (Haar / DNN)    │     │  (48×48 Gray)   │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Annotated  │◀────│  Post-Processing │◀────│  CNN Inference  │
│  Output     │     │  & Visualization │     │  (7 Classes)    │
└─────────────┘     └──────────────────┘     └─────────────────┘
```

### CNN Model Architecture

```
Input (48×48×1)
    │
    ├── Conv Block 1: Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
    ├── Conv Block 2: Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
    ├── Conv Block 3: Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Dropout(0.25)
    ├── Conv Block 4: Conv2D(256) → BN → Conv2D(256) → BN → MaxPool → Dropout(0.25)
    │
    ├── Flatten
    ├── Dense(512) → BN → Dropout(0.5)
    ├── Dense(256) → BN → Dropout(0.5)
    └── Dense(7, softmax) → Output
```

**Key Features:**
- L2 Regularization on all layers to prevent overfitting
- Batch Normalization for training stability
- Progressive dropout (0.25 conv → 0.5 dense)
- Data augmentation (rotation, shift, zoom, flip)
- Learning rate scheduling with ReduceLROnPlateau

---

## 📁 Project Structure

```
Real-Time Face Emotion Recognition/
│
├── config.py                        # Central configuration & hyperparameters
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── .gitignore                       # Git ignore rules
│
├── src/                             # Core source code
│   ├── __init__.py                  # Package init
│   ├── emotion_model.py             # CNN architecture (4 conv blocks)
│   ├── face_detector.py             # Dual face detection (Haar + DNN)
│   ├── preprocessor.py              # Image preprocessing pipeline
│   ├── train.py                     # Training pipeline with callbacks
│   ├── predict.py                   # Inference engine
│   └── utils.py                     # Utility functions & helpers
│
├── app/                             # Application layer
│   ├── webcam_app.py                # Main real-time webcam application
│   ├── streamlit_dashboard.py       # Web-based dashboard with analytics
│   └── video_analyzer.py            # Analyze pre-recorded video files
│
├── models/                          # Trained model weights
├── data/                            # Dataset storage (FER-2013)
├── results/                         # Training results & visualizations
└── tests/                           # Unit tests
    └── test_pipeline.py             # Comprehensive pipeline tests
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Real-Time-Face-Emotion-Recognition.git
cd Real-Time-Face-Emotion-Recognition
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

Download the **FER-2013** dataset from Kaggle:

**Option A — Directory Format (Recommended):**
```bash
# Download from: https://www.kaggle.com/datasets/msambare/fer2013
# Extract so structure is:
#   data/train/{angry,disgust,fear,happy,sad,surprise,neutral}/
#   data/test/{angry,disgust,fear,happy,sad,surprise,neutral}/
```

**Option B — CSV Format:**
```bash
pip install kaggle
kaggle datasets download -d deadskull7/fer2013
# Place fer2013.csv in data/ directory
```

### 5. Train the Model

```bash
python src/train.py
```

Optional training arguments:
```bash
python src/train.py --epochs 100 --batch-size 128 --lr 0.0005
```

### 6. Run the Application

**Webcam App (OpenCV):**
```bash
python app/webcam_app.py
```

**Web Dashboard (Streamlit):**
```bash
streamlit run app/streamlit_dashboard.py
```

**Video Analysis:**
```bash
python app/video_analyzer.py --input path/to/video.mp4
```

---

## 🎮 Webcam App Controls

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit application |
| `S` | Save screenshot |
| `P` | Toggle probability bars |
| `R` | Reset emotion tracker |
| `F` | Toggle fullscreen |

---

## ⚙️ Configuration

All hyperparameters and settings are centralized in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMG_SIZE` | 48 | Input image dimensions |
| `BATCH_SIZE` | 64 | Training batch size |
| `EPOCHS` | 50 | Maximum training epochs |
| `LEARNING_RATE` | 0.001 | Initial learning rate |
| `DROPOUT_RATE` | 0.25 | Conv block dropout |
| `CONV_FILTERS` | [32, 64, 128, 256] | Filters per conv block |
| `DENSE_UNITS` | [512, 256] | Dense layer units |

---

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Or with unittest:

```bash
python -m unittest tests/test_pipeline.py -v
```

---

## 📊 Model Performance

Training on the FER-2013 dataset (35,887 images):

| Metric | Score |
|--------|-------|
| Training Accuracy | ~95% |
| Validation Accuracy | ~65-68% |
| Test Accuracy | ~65-67% |

> **Note:** FER-2013 is a challenging dataset with inherent label noise (~65% human accuracy). The model achieves competitive performance with state-of-the-art approaches.

### Per-Class Performance

| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Happy | ~0.85 | ~0.90 | ~0.87 |
| Surprise | ~0.78 | ~0.82 | ~0.80 |
| Neutral | ~0.60 | ~0.65 | ~0.62 |
| Sad | ~0.50 | ~0.55 | ~0.52 |
| Angry | ~0.55 | ~0.50 | ~0.52 |
| Fear | ~0.50 | ~0.45 | ~0.47 |
| Disgust | ~0.65 | ~0.40 | ~0.50 |

---

## 🔧 Technical Details

### Face Detection
- **Haar Cascade**: Fast, lightweight (~30+ FPS). Uses histogram equalization for improved detection.
- **DNN (SSD)**: More accurate, uses pre-trained Caffe model. Configurable confidence threshold.
- **Hybrid Mode**: Tries DNN first, falls back to Haar if no detections.

### Preprocessing Pipeline
1. Convert to grayscale
2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
3. Resize to 48×48
4. Normalize to [0, 1]
5. Reshape for CNN input (1, 48, 48, 1)

### Data Augmentation
- Random rotation (±15°)
- Width/height shift (±15%)
- Shear transformation (15%)
- Zoom (±15%)
- Horizontal flip

### Training Callbacks
- **ModelCheckpoint**: Save best model by validation accuracy
- **EarlyStopping**: Stop after 10 epochs without improvement
- **ReduceLROnPlateau**: Halve LR after 5 stagnant epochs
- **CSVLogger**: Log all metrics to CSV

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Core language |
| **TensorFlow/Keras** | CNN model building & training |
| **OpenCV** | Face detection & video processing |
| **NumPy** | Numerical computations |
| **Pandas** | Data handling |
| **Matplotlib/Seaborn** | Visualization |
| **Streamlit** | Web dashboard |
| **scikit-learn** | Metrics & data splitting |

---

## 📈 Future Improvements

- [ ] Transfer learning with VGGFace / ResNet50
- [ ] Attention mechanisms for better feature extraction
- [ ] Face landmark detection for improved ROI
- [ ] Emotion intensity estimation (not just classification)
- [ ] Multi-camera support
- [ ] Edge deployment (TFLite / ONNX)
- [ ] REST API endpoint for cloud deployment

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📬 Contact

**Gnana Sudarshan Reddy** — [GitHub Profile](https://github.com/Gnana-Sudarshan-Reddy)

---

<p align="center">
  <b>⭐ If you found this project useful, please give it a star! ⭐</b>
</p>
