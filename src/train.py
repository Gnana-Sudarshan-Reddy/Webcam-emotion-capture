"""
Training Pipeline
==================
Complete training pipeline for the Emotion Recognition CNN model.

Features:
  - FER-2013 dataset loading and preprocessing
  - Data augmentation with ImageDataGenerator
  - Learning rate scheduling with ReduceLROnPlateau
  - Early stopping and model checkpointing
  - Training history visualization
  - Confusion matrix and classification report
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger,
)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    DATA_DIR,
    MODEL_DIR,
    RESULTS_DIR,
    EMOTION_LABELS,
    NUM_CLASSES,
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
    AUGMENTATION_CONFIG,
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    TRAINING_HISTORY_PATH,
)
from src.emotion_model import build_emotion_cnn
from src.preprocessor import FacePreprocessor


def load_fer2013(data_path=None):
    """
    Load FER-2013 dataset from CSV file.

    The CSV should have columns: 'emotion', 'pixels', 'Usage'
    - emotion: integer label (0-6)
    - pixels: space-separated pixel values (48*48 = 2304 values)
    - Usage: 'Training', 'PublicTest', or 'PrivateTest'

    Args:
        data_path (str): Path to fer2013.csv file.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    data_path = data_path or os.path.join(DATA_DIR, "fer2013.csv")

    if not os.path.exists(data_path):
        print("=" * 60)
        print("  FER-2013 Dataset Not Found!")
        print("=" * 60)
        print(f"\n  Expected path: {data_path}")
        print("\n  To download the dataset:")
        print("  1. Visit: https://www.kaggle.com/datasets/msambare/fer2013")
        print("  2. Download fer2013.csv")
        print(f"  3. Place it in: {DATA_DIR}/")
        print("\n  Or use Kaggle CLI:")
        print("  $ pip install kaggle")
        print("  $ kaggle datasets download -d msambare/fer2013")
        print(f"  $ unzip fer2013.zip -d {DATA_DIR}/")
        print("\n  Alternatively, use the directory-based format:")
        print(f"  Place images in {DATA_DIR}/train/ and {DATA_DIR}/test/")
        print("  with subdirectories for each emotion class.")
        print("=" * 60)

        # Try directory-based format
        return load_fer2013_directory()

    print(f"[INFO] Loading FER-2013 from: {data_path}")
    df = pd.read_csv(data_path)

    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")

    # Parse pixel strings into arrays
    pixels = df["pixels"].values
    labels = df["emotion"].values

    # Prepare data
    X, y = FacePreprocessor.prepare_training_data(pixels, labels)

    if "Usage" in df.columns:
        # Split using the Usage column
        train_mask = df["Usage"] == "Training"
        val_mask = df["Usage"] == "PublicTest"
        test_mask = df["Usage"] == "PrivateTest"

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
    else:
        # Manual split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42,
            stratify=np.argmax(y_temp, axis=1)
        )

    print(f"[INFO] Training set:   {X_train.shape[0]} samples")
    print(f"[INFO] Validation set: {X_val.shape[0]} samples")
    print(f"[INFO] Test set:       {X_test.shape[0]} samples")

    # Print class distribution
    train_labels = np.argmax(y_train, axis=1)
    print("\n[INFO] Training set class distribution:")
    for i, label in enumerate(EMOTION_LABELS):
        count = np.sum(train_labels == i)
        print(f"  {label}: {count} ({count/len(train_labels)*100:.1f}%)")

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_fer2013_directory(data_dir=None):
    """
    Load FER-2013 from directory structure:
    data/train/{emotion_name}/image.png
    data/test/{emotion_name}/image.png

    Args:
        data_dir (str): Base data directory.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    import cv2

    data_dir = data_dir or DATA_DIR
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"Neither fer2013.csv nor directory structure found.\n"
            f"Expected: {train_dir}/ with emotion subdirectories.\n"
            f"Please download the FER-2013 dataset first."
        )

    def load_from_dir(directory):
        images = []
        labels = []
        emotion_map = {name.lower(): i for i, name in enumerate(EMOTION_LABELS)}

        for emotion_name in sorted(os.listdir(directory)):
            emotion_dir = os.path.join(directory, emotion_name)
            if not os.path.isdir(emotion_dir):
                continue

            label = emotion_map.get(emotion_name.lower())
            if label is None:
                print(f"[WARN] Skipping unknown emotion directory: {emotion_name}")
                continue

            for img_file in os.listdir(emotion_dir):
                img_path = os.path.join(emotion_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    images.append(img)
                    labels.append(label)

        return np.array(images), np.array(labels)

    print(f"[INFO] Loading training data from: {train_dir}")
    X_train, y_train_labels = load_from_dir(train_dir)

    if os.path.exists(test_dir):
        print(f"[INFO] Loading test data from: {test_dir}")
        X_test, y_test_labels = load_from_dir(test_dir)
    else:
        # Split from training
        X_train, X_test, y_train_labels, y_test_labels = train_test_split(
            X_train, y_train_labels, test_size=0.2, random_state=42,
            stratify=y_train_labels
        )

    # Normalize and reshape
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # One-hot encode
    y_train = to_categorical(y_train_labels, NUM_CLASSES)
    y_test_enc = to_categorical(y_test_labels, NUM_CLASSES)

    # Split validation from training
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42,
        stratify=np.argmax(y_train, axis=1)
    )

    print(f"[INFO] Training set:   {X_train.shape[0]} samples")
    print(f"[INFO] Validation set: {X_val.shape[0]} samples")
    print(f"[INFO] Test set:       {X_test.shape[0]} samples")

    return X_train, y_train, X_val, y_val, X_test, y_test_enc


def create_data_generators(X_train, y_train, X_val, y_val):
    """
    Create data generators with augmentation for training.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.

    Returns:
        tuple: (train_generator, val_generator, steps_per_epoch, validation_steps)
    """
    # Training data augmentation
    train_datagen = ImageDataGenerator(**AUGMENTATION_CONFIG)

    # Validation — no augmentation, only rescaling
    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(
        X_train, y_train, batch_size=BATCH_SIZE, shuffle=True
    )
    val_generator = val_datagen.flow(
        X_val, y_val, batch_size=BATCH_SIZE, shuffle=False
    )

    steps_per_epoch = len(X_train) // BATCH_SIZE
    validation_steps = len(X_val) // BATCH_SIZE

    return train_generator, val_generator, steps_per_epoch, validation_steps


def get_callbacks():
    """
    Create training callbacks.

    Returns:
        list: List of Keras callbacks.
    """
    callbacks = [
        # Save best model
        ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        # Early stopping
        EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        # CSV logger
        CSVLogger(
            os.path.join(RESULTS_DIR, "training_log.csv"),
            append=False,
        ),
    ]

    return callbacks


def plot_training_history(history, save_path=None):
    """
    Plot training and validation accuracy/loss curves.

    Args:
        history: Keras training history object or dict.
        save_path (str): Path to save the plot.
    """
    if isinstance(history, dict):
        hist = history
    else:
        hist = history.history

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    axes[0].plot(hist["accuracy"], label="Train Accuracy", linewidth=2)
    axes[0].plot(hist["val_accuracy"], label="Val Accuracy", linewidth=2)
    axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Loss plot
    axes[1].plot(hist["loss"], label="Train Loss", linewidth=2)
    axes[1].plot(hist["val_loss"], label="Val Loss", linewidth=2)
    axes[1].set_title("Model Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = save_path or os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Training curves saved to: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix with emotion labels.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        save_path (str): Path to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
        ax=axes[0]
    )
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Normalized
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="Oranges",
        xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
        ax=axes[1]
    )
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()

    save_path = save_path or os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Confusion matrix saved to: {save_path}")
    plt.close()


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data.

    Args:
        model: Trained Keras model.
        X_test (np.ndarray): Test images.
        y_test (np.ndarray): Test labels (one-hot).

    Returns:
        dict: Evaluation metrics.
    """
    # Get predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Test accuracy
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'='*50}")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"{'='*50}")

    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=EMOTION_LABELS,
        output_dict=True,
    )
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=EMOTION_LABELS))

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred)

    return {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "classification_report": report,
    }


def train(args=None):
    """
    Main training function.

    Args:
        args: Command-line arguments (optional).
    """
    print("=" * 60)
    print("  Real-Time Face Emotion Recognition — Training Pipeline")
    print("=" * 60)
    print(f"  TensorFlow version: {tf.__version__}")
    print(f"  GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print("=" * 60)

    # Step 1: Load data
    print("\n[STEP 1/5] Loading dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_fer2013()

    # Step 2: Build model
    print("\n[STEP 2/5] Building CNN model...")
    model = build_emotion_cnn(summary=True)

    # Step 3: Create data generators
    print("\n[STEP 3/5] Setting up data augmentation...")
    train_gen, val_gen, steps, val_steps = create_data_generators(
        X_train, y_train, X_val, y_val
    )

    # Step 4: Train
    print("\n[STEP 4/5] Training model...")
    start_time = datetime.now()

    history = model.fit(
        train_gen,
        steps_per_epoch=steps,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=get_callbacks(),
        verbose=1,
    )

    training_time = datetime.now() - start_time
    print(f"\n[INFO] Training completed in: {training_time}")

    # Save final model
    model.save(FINAL_MODEL_PATH)
    print(f"[INFO] Final model saved to: {FINAL_MODEL_PATH}")

    # Save training history
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist_dict["training_time"] = str(training_time)
    with open(TRAINING_HISTORY_PATH, "w") as f:
        json.dump(hist_dict, f, indent=2)
    print(f"[INFO] Training history saved to: {TRAINING_HISTORY_PATH}")

    # Plot training curves
    plot_training_history(history)

    # Step 5: Evaluate
    print("\n[STEP 5/5] Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("  Training Complete!")
    print(f"  Best model saved to: {BEST_MODEL_PATH}")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print(f"{'='*60}")

    return model, history, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Emotion Recognition CNN")
    parser.add_argument("--data", type=str, default=None, help="Path to FER-2013 CSV")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    args = parser.parse_args()

    # Override config if args provided
    if args.epochs:
        EPOCHS = args.epochs
    if args.batch_size:
        BATCH_SIZE = args.batch_size
    if args.lr:
        LEARNING_RATE = args.lr

    train(args)
