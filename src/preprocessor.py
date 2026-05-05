"""
Image Preprocessor Module
==========================
Handles all image preprocessing for the emotion recognition pipeline:
  - Grayscale conversion
  - Resizing to model input dimensions
  - Histogram equalization
  - Normalization
  - Data augmentation (for training)
"""

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMG_SIZE, INPUT_SHAPE


class FacePreprocessor:
    """
    Preprocesses face images for the emotion classification CNN.
    Converts faces to 48x48 grayscale normalized arrays.
    """

    def __init__(self, target_size=None, apply_clahe=True):
        """
        Initialize preprocessor.

        Args:
            target_size (int): Target dimension for resizing (square).
            apply_clahe (bool): Whether to apply CLAHE for contrast enhancement.
        """
        self.target_size = target_size or IMG_SIZE
        self.apply_clahe = apply_clahe

        if self.apply_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        print(f"[INFO] Preprocessor initialized (target: {self.target_size}x{self.target_size})")

    def preprocess(self, face_img):
        """
        Full preprocessing pipeline for a single face image.

        Args:
            face_img (np.ndarray): Input face image (BGR or grayscale).

        Returns:
            np.ndarray: Preprocessed image ready for CNN (1, 48, 48, 1).
        """
        # Convert to grayscale if needed
        if len(face_img.shape) == 3 and face_img.shape[2] == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        elif len(face_img.shape) == 3 and face_img.shape[2] == 1:
            gray = face_img.squeeze()
        else:
            gray = face_img.copy()

        # Apply CLAHE for contrast enhancement
        if self.apply_clahe:
            gray = self.clahe.apply(gray)

        # Resize to target dimensions
        resized = cv2.resize(
            gray,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_AREA,
        )

        # Normalize pixel values to [0, 1]
        normalized = resized.astype("float32") / 255.0

        # Reshape for CNN input: (1, height, width, channels)
        processed = normalized.reshape(1, self.target_size, self.target_size, 1)

        return processed

    def preprocess_batch(self, face_images):
        """
        Preprocess a batch of face images.

        Args:
            face_images (list): List of face images (BGR or grayscale).

        Returns:
            np.ndarray: Batch of preprocessed images (N, 48, 48, 1).
        """
        batch = []
        for face in face_images:
            processed = self.preprocess(face)
            batch.append(processed[0])  # Remove batch dimension

        return np.array(batch)

    @staticmethod
    def prepare_training_data(pixels_array, labels_array):
        """
        Prepare FER-2013 data for training.

        Args:
            pixels_array (np.ndarray): Array of pixel strings or flat arrays.
            labels_array (np.ndarray): Array of emotion labels (integers).

        Returns:
            tuple: (X, y) where X is (N, 48, 48, 1) and y is one-hot encoded.
        """
        from tensorflow.keras.utils import to_categorical

        images = []
        for pixels in pixels_array:
            if isinstance(pixels, str):
                pixel_values = np.array(pixels.split(), dtype="float32")
            else:
                pixel_values = np.array(pixels, dtype="float32")

            img = pixel_values.reshape(48, 48)
            images.append(img)

        X = np.array(images)
        X = X.astype("float32") / 255.0
        X = X.reshape(-1, 48, 48, 1)

        y = to_categorical(labels_array, num_classes=7)

        return X, y
