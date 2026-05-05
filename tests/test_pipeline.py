"""
Unit Tests for Emotion Recognition Pipeline
=============================================
Tests face detection, preprocessing, model architecture, and utilities.
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class TestConfig(unittest.TestCase):
    """Test configuration module."""

    def test_emotion_labels(self):
        from config import EMOTION_LABELS, NUM_CLASSES
        self.assertEqual(len(EMOTION_LABELS), 7)
        self.assertEqual(NUM_CLASSES, 7)
        self.assertIn("Happy", EMOTION_LABELS)
        self.assertIn("Sad", EMOTION_LABELS)
        self.assertIn("Angry", EMOTION_LABELS)

    def test_emotion_colors(self):
        from config import EMOTION_COLORS, EMOTION_LABELS
        for label in EMOTION_LABELS:
            self.assertIn(label, EMOTION_COLORS)
            color = EMOTION_COLORS[label]
            self.assertEqual(len(color), 3)

    def test_directories_exist(self):
        from config import MODEL_DIR, DATA_DIR, RESULTS_DIR
        self.assertTrue(os.path.isdir(MODEL_DIR))
        self.assertTrue(os.path.isdir(DATA_DIR))
        self.assertTrue(os.path.isdir(RESULTS_DIR))


class TestPreprocessor(unittest.TestCase):
    """Test image preprocessor."""

    def setUp(self):
        from src.preprocessor import FacePreprocessor
        self.preprocessor = FacePreprocessor()

    def test_preprocess_bgr(self):
        """Test preprocessing a BGR image."""
        fake_face = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        result = self.preprocessor.preprocess(fake_face)
        self.assertEqual(result.shape, (1, 48, 48, 1))
        self.assertTrue(result.max() <= 1.0)
        self.assertTrue(result.min() >= 0.0)

    def test_preprocess_gray(self):
        """Test preprocessing a grayscale image."""
        fake_face = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        result = self.preprocessor.preprocess(fake_face)
        self.assertEqual(result.shape, (1, 48, 48, 1))

    def test_preprocess_batch(self):
        """Test batch preprocessing."""
        faces = [np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(5)]
        result = self.preprocessor.preprocess_batch(faces)
        self.assertEqual(result.shape, (5, 48, 48, 1))


class TestFaceDetector(unittest.TestCase):
    """Test face detector module."""

    def setUp(self):
        from src.face_detector import HaarFaceDetector
        self.detector = HaarFaceDetector()

    def test_detector_loads(self):
        """Test that the Haar cascade loads correctly."""
        self.assertFalse(self.detector.face_cascade.empty())

    def test_detect_empty_image(self):
        """Test detection on an empty/blank image."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = self.detector.detect_faces(blank)
        self.assertIsInstance(faces, list)

    def test_detect_returns_list(self):
        """Test that detection returns a list."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        faces = self.detector.detect_faces(img)
        self.assertIsInstance(faces, list)


class TestEmotionModel(unittest.TestCase):
    """Test CNN model architecture."""

    def test_build_model(self):
        """Test model builds correctly."""
        from src.emotion_model import build_emotion_cnn
        model = build_emotion_cnn(summary=False)
        self.assertIsNotNone(model)
        self.assertEqual(model.output_shape, (None, 7))
        self.assertEqual(model.input_shape, (None, 48, 48, 1))

    def test_model_prediction_shape(self):
        """Test model prediction output shape."""
        from src.emotion_model import build_emotion_cnn
        model = build_emotion_cnn(summary=False)
        dummy = np.random.rand(1, 48, 48, 1).astype("float32")
        pred = model.predict(dummy, verbose=0)
        self.assertEqual(pred.shape, (1, 7))

    def test_predictions_sum_to_one(self):
        """Test softmax outputs sum to ~1."""
        from src.emotion_model import build_emotion_cnn
        model = build_emotion_cnn(summary=False)
        dummy = np.random.rand(3, 48, 48, 1).astype("float32")
        preds = model.predict(dummy, verbose=0)
        for p in preds:
            self.assertAlmostEqual(np.sum(p), 1.0, places=5)

    def test_batch_prediction(self):
        """Test batch prediction."""
        from src.emotion_model import build_emotion_cnn
        model = build_emotion_cnn(summary=False)
        batch = np.random.rand(8, 48, 48, 1).astype("float32")
        preds = model.predict(batch, verbose=0)
        self.assertEqual(preds.shape, (8, 7))


class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def test_fps_counter(self):
        from src.utils import FPSCounter
        counter = FPSCounter()
        import time
        time.sleep(0.05)
        fps = counter.update()
        self.assertIsInstance(fps, float)
        self.assertGreater(fps, 0)

    def test_emotion_tracker(self):
        from src.utils import EmotionTracker
        tracker = EmotionTracker()
        tracker.add("Happy", 0.95)
        tracker.add("Happy", 0.85)
        tracker.add("Sad", 0.70)
        dist = tracker.get_distribution()
        self.assertEqual(dist["Happy"], 2)
        self.assertEqual(dist["Sad"], 1)
        self.assertEqual(tracker.get_dominant_emotion(), "Happy")

    def test_tracker_max_history(self):
        from src.utils import EmotionTracker
        tracker = EmotionTracker(max_history=10)
        for i in range(20):
            tracker.add("Happy", 0.9)
        self.assertEqual(len(tracker.history), 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
