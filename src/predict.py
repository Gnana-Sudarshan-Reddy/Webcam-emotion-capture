"""
Emotion Prediction Engine
==========================
Combines face detection, preprocessing, and CNN inference.
"""

import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EMOTION_LABELS, EMOTION_COLORS, EMOTION_EMOJIS, BEST_MODEL_PATH
from src.face_detector import FaceDetector
from src.preprocessor import FacePreprocessor
from src.emotion_model import load_trained_model


class EmotionPredictor:
    """Complete emotion prediction pipeline."""

    def __init__(self, model_path=None, detection_method="haar"):
        model_path = model_path or BEST_MODEL_PATH
        print("[INFO] Initializing Emotion Predictor...")
        self.face_detector = FaceDetector(method=detection_method)
        self.preprocessor = FacePreprocessor()
        self.model = load_trained_model(model_path)
        print("[INFO] Emotion Predictor ready!")

    def predict_emotion(self, face_img):
        """Predict emotion from a single face image."""
        processed = self.preprocessor.preprocess(face_img)
        predictions = self.model.predict(processed, verbose=0)[0]
        emotion_idx = np.argmax(predictions)
        emotion = EMOTION_LABELS[emotion_idx]
        confidence = float(predictions[emotion_idx])
        probabilities = {
            EMOTION_LABELS[i]: float(predictions[i])
            for i in range(len(EMOTION_LABELS))
        }
        return {
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": probabilities,
            "emoji": EMOTION_EMOJIS.get(emotion, ""),
            "color": EMOTION_COLORS.get(emotion, (255, 255, 255)),
        }

    def predict_frame(self, frame):
        """Detect faces and predict emotions for all faces in a frame."""
        results = []
        face_data = self.face_detector.detect_and_extract(frame)
        for bbox, face_roi in face_data:
            try:
                prediction = self.predict_emotion(face_roi)
                prediction["bbox"] = bbox
                results.append(prediction)
            except Exception as e:
                print(f"[WARN] Prediction failed for face at {bbox}: {e}")
        return results

    def annotate_frame(self, frame, predictions=None, show_probabilities=True):
        """Annotate a frame with emotion predictions."""
        annotated = frame.copy()
        if predictions is None:
            predictions = self.predict_frame(frame)

        for pred in predictions:
            x, y, w, h = pred["bbox"]
            emotion = pred["emotion"]
            confidence = pred["confidence"]
            color = pred["color"]

            # Bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Label
            label = f"{emotion} ({confidence:.0%})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_y = max(y - 10, label_size[1] + 10)
            cv2.rectangle(annotated,
                (x, label_y - label_size[1] - 5),
                (x + label_size[0] + 5, label_y + 5), color, cv2.FILLED)
            text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
            cv2.putText(annotated, label, (x + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)

            # Probability bars
            if show_probabilities:
                bar_x = x + w + 10
                bar_y_start = y
                bar_width = 120
                bar_height = 15
                gap = 3
                for i, (emo, prob) in enumerate(pred["probabilities"].items()):
                    by = bar_y_start + i * (bar_height + gap)
                    cv2.rectangle(annotated, (bar_x, by),
                        (bar_x + bar_width, by + bar_height), (50, 50, 50), cv2.FILLED)
                    fill_w = int(bar_width * prob)
                    emo_color = EMOTION_COLORS.get(emo, (200, 200, 200))
                    cv2.rectangle(annotated, (bar_x, by),
                        (bar_x + fill_w, by + bar_height), emo_color, cv2.FILLED)
                    cv2.putText(annotated, f"{emo[:3]} {prob:.0%}",
                        (bar_x + 2, by + bar_height - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated
