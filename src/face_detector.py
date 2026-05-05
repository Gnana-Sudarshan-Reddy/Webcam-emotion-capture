"""
Face Detector Module
====================
Provides dual face detection using:
  1. Haar Cascade Classifier (fast, lightweight)
  2. DNN-based SSD detector (more accurate)

Supports multi-face detection with configurable confidence thresholds.
"""

import cv2
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    HAAR_CASCADE_PATH,
    HAAR_SCALE_FACTOR,
    HAAR_MIN_NEIGHBORS,
    HAAR_MIN_SIZE,
    DNN_PROTOTXT,
    DNN_CAFFEMODEL,
    DNN_CONFIDENCE_THRESHOLD,
)


class HaarFaceDetector:
    """
    Face detector using Haar Cascade Classifier.
    Fast and lightweight — ideal for real-time applications.
    """

    def __init__(self, cascade_path=None):
        """
        Initialize Haar Cascade face detector.

        Args:
            cascade_path (str): Path to Haar cascade XML file.
                                Uses OpenCV's built-in if not specified.
        """
        if cascade_path and os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            # Fall back to OpenCV's built-in cascade
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

        if self.face_cascade.empty():
            raise IOError("Failed to load Haar Cascade classifier.")

        print("[INFO] Haar Cascade Face Detector initialized.")

    def detect_faces(self, frame, scale_factor=None, min_neighbors=None, min_size=None):
        """
        Detect faces in a frame using Haar Cascade.

        Args:
            frame (np.ndarray): Input BGR image.
            scale_factor (float): Image scaling factor for multi-scale detection.
            min_neighbors (int): Minimum neighbors for detection quality.
            min_size (tuple): Minimum face size (w, h).

        Returns:
            list: List of (x, y, w, h) bounding boxes.
        """
        scale_factor = scale_factor or HAAR_SCALE_FACTOR
        min_neighbors = min_neighbors or HAAR_MIN_NEIGHBORS
        min_size = min_size or HAAR_MIN_SIZE

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Improve contrast

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Convert to list of tuples
        if len(faces) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


class DNNFaceDetector:
    """
    Face detector using OpenCV's DNN module with a pre-trained SSD model.
    More accurate than Haar Cascade, slightly slower.
    """

    def __init__(self, prototxt_path=None, model_path=None, confidence_threshold=None):
        """
        Initialize DNN-based face detector.

        Args:
            prototxt_path (str): Path to deploy.prototxt.
            model_path (str): Path to .caffemodel weights.
            confidence_threshold (float): Minimum confidence for detections.
        """
        self.confidence_threshold = confidence_threshold or DNN_CONFIDENCE_THRESHOLD
        prototxt = prototxt_path or DNN_PROTOTXT
        model = model_path or DNN_CAFFEMODEL

        if os.path.exists(prototxt) and os.path.exists(model):
            self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
            self.available = True
            print("[INFO] DNN Face Detector initialized.")
        else:
            self.net = None
            self.available = False
            print("[WARN] DNN model files not found. DNN detector unavailable.")
            print(f"       Expected: {prototxt}")
            print(f"       Expected: {model}")

    def detect_faces(self, frame):
        """
        Detect faces using DNN SSD model.

        Args:
            frame (np.ndarray): Input BGR image.

        Returns:
            list: List of (x, y, w, h) bounding boxes.
        """
        if not self.available:
            return []

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                # Ensure coordinates are within frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                face_w = x2 - x1
                face_h = y2 - y1

                if face_w > 0 and face_h > 0:
                    faces.append((x1, y1, face_w, face_h))

        return faces


class FaceDetector:
    """
    Unified face detector that combines Haar and DNN detectors.
    Automatically falls back to Haar if DNN is unavailable.
    """

    def __init__(self, method="haar"):
        """
        Initialize face detector.

        Args:
            method (str): Detection method — 'haar', 'dnn', or 'hybrid'.
        """
        self.method = method.lower()

        self.haar_detector = HaarFaceDetector()

        if self.method in ("dnn", "hybrid"):
            self.dnn_detector = DNNFaceDetector()
            if not self.dnn_detector.available:
                print("[WARN] Falling back to Haar Cascade detector.")
                self.method = "haar"
        else:
            self.dnn_detector = None

        print(f"[INFO] Using face detection method: {self.method}")

    def detect(self, frame):
        """
        Detect faces in a frame.

        Args:
            frame (np.ndarray): Input BGR image.

        Returns:
            list: List of (x, y, w, h) bounding boxes.
        """
        if self.method == "dnn":
            faces = self.dnn_detector.detect_faces(frame)
        elif self.method == "hybrid":
            # Use DNN first, fall back to Haar if no detections
            faces = self.dnn_detector.detect_faces(frame)
            if len(faces) == 0:
                faces = self.haar_detector.detect_faces(frame)
        else:
            faces = self.haar_detector.detect_faces(frame)

        return faces

    def detect_and_extract(self, frame):
        """
        Detect faces and extract face ROIs.

        Args:
            frame (np.ndarray): Input BGR image.

        Returns:
            list: List of (bbox, face_roi) tuples.
                  bbox = (x, y, w, h), face_roi = cropped face image.
        """
        faces = self.detect(frame)
        results = []

        for (x, y, w, h) in faces:
            # Add padding around face for better detection
            pad_x = int(w * 0.1)
            pad_y = int(h * 0.1)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame.shape[1], x + w + pad_x)
            y2 = min(frame.shape[0], y + h + pad_y)

            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size > 0:
                results.append(((x, y, w, h), face_roi))

        return results
