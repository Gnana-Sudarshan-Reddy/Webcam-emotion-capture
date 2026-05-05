"""
Emotion Recognition CNN Model
==============================
Custom Convolutional Neural Network for facial emotion classification.

Architecture:
  - 4 Convolutional blocks (Conv2D → BatchNorm → Conv2D → BatchNorm → MaxPool → Dropout)
  - 2 Dense layers with BatchNorm and Dropout
  - Softmax output for 7 emotion classes

Designed for 48x48 grayscale input images (FER-2013 format).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    Flatten,
    BatchNormalization,
    Input,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from config import (
    INPUT_SHAPE,
    NUM_CLASSES,
    LEARNING_RATE,
    DROPOUT_RATE,
    DENSE_DROPOUT_RATE,
    CONV_FILTERS,
    DENSE_UNITS,
    BEST_MODEL_PATH,
)


def build_emotion_cnn(
    input_shape=None,
    num_classes=None,
    learning_rate=None,
    summary=True,
):
    """
    Build the Emotion Recognition CNN model.

    Architecture per convolutional block:
        Conv2D(filters, 3x3) → BatchNorm → ReLU →
        Conv2D(filters, 3x3) → BatchNorm → ReLU →
        MaxPooling2D(2x2) → Dropout

    Args:
        input_shape (tuple): Input image shape (H, W, C).
        num_classes (int): Number of output classes.
        learning_rate (float): Adam optimizer learning rate.
        summary (bool): Whether to print model summary.

    Returns:
        keras.Model: Compiled CNN model.
    """
    input_shape = input_shape or INPUT_SHAPE
    num_classes = num_classes or NUM_CLASSES
    learning_rate = learning_rate or LEARNING_RATE

    model = Sequential(name="EmotionRecognitionCNN")

    # Input layer
    model.add(Input(shape=input_shape))

    # ──────────────────────────────────────────
    # Convolutional Block 1 — 32 filters
    # ──────────────────────────────────────────
    model.add(Conv2D(CONV_FILTERS[0], (3, 3), padding="same", activation="relu",
                     kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Conv2D(CONV_FILTERS[0], (3, 3), padding="same", activation="relu",
                     kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(DROPOUT_RATE))

    # ──────────────────────────────────────────
    # Convolutional Block 2 — 64 filters
    # ──────────────────────────────────────────
    model.add(Conv2D(CONV_FILTERS[1], (3, 3), padding="same", activation="relu",
                     kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Conv2D(CONV_FILTERS[1], (3, 3), padding="same", activation="relu",
                     kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(DROPOUT_RATE))

    # ──────────────────────────────────────────
    # Convolutional Block 3 — 128 filters
    # ──────────────────────────────────────────
    model.add(Conv2D(CONV_FILTERS[2], (3, 3), padding="same", activation="relu",
                     kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Conv2D(CONV_FILTERS[2], (3, 3), padding="same", activation="relu",
                     kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(DROPOUT_RATE))

    # ──────────────────────────────────────────
    # Convolutional Block 4 — 256 filters
    # ──────────────────────────────────────────
    model.add(Conv2D(CONV_FILTERS[3], (3, 3), padding="same", activation="relu",
                     kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Conv2D(CONV_FILTERS[3], (3, 3), padding="same", activation="relu",
                     kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(DROPOUT_RATE))

    # ──────────────────────────────────────────
    # Fully Connected Layers
    # ──────────────────────────────────────────
    model.add(Flatten())

    model.add(Dense(DENSE_UNITS[0], activation="relu", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(DENSE_DROPOUT_RATE))

    model.add(Dense(DENSE_UNITS[1], activation="relu", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(DENSE_DROPOUT_RATE))

    # ──────────────────────────────────────────
    # Output Layer
    # ──────────────────────────────────────────
    model.add(Dense(num_classes, activation="softmax"))

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    if summary:
        model.summary()
        total_params = model.count_params()
        print(f"\n[INFO] Total parameters: {total_params:,}")
        print(f"[INFO] Model compiled with Adam (lr={learning_rate})")

    return model


def load_trained_model(model_path=None):
    """
    Load a pre-trained emotion recognition model.

    Args:
        model_path (str): Path to saved model file.

    Returns:
        keras.Model: Loaded model ready for inference.
    """
    model_path = model_path or BEST_MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            "Please train the model first using: python src/train.py"
        )

    model = load_model(model_path)
    print(f"[INFO] Model loaded from: {model_path}")
    return model


def get_model_info(model):
    """
    Get model architecture information.

    Args:
        model: Keras model.

    Returns:
        dict: Model information.
    """
    return {
        "name": model.name,
        "total_params": model.count_params(),
        "trainable_params": sum(
            [np.prod(w.shape) for w in model.trainable_weights]
        ),
        "layers": len(model.layers),
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
    }


if __name__ == "__main__":
    import numpy as np
    # Build and display model
    model = build_emotion_cnn(summary=True)
    print("\n[TEST] Running inference on random input...")
    dummy_input = np.random.rand(1, 48, 48, 1).astype("float32")
    prediction = model.predict(dummy_input, verbose=0)
    print(f"[TEST] Output shape: {prediction.shape}")
    print(f"[TEST] Predictions: {prediction[0]}")
    print(f"[TEST] Predicted class: {np.argmax(prediction[0])}")
    print("[TEST] Model build successful! ✓")
