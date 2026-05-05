"""
Real-Time Webcam Emotion Recognition App
==========================================
Main application for live emotion detection from webcam feed.

Features:
  - Real-time face detection and emotion prediction
  - Color-coded bounding boxes per emotion
  - Probability bar chart overlay
  - FPS counter and face count HUD
  - Screenshot capture (S key)
  - Toggle probability bars (P key)
  - Emotion tracking and session analytics

Controls:
  Q / ESC  — Quit
  S        — Save screenshot
  P        — Toggle probability bars
  R        — Reset emotion tracker
  F        — Toggle fullscreen
"""

import os
import sys
import cv2
import numpy as np
import time
from collections import deque

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    EMOTION_LABELS, EMOTION_COLORS, EMOTION_EMOJIS,
    BEST_MODEL_PATH, RESULTS_DIR,
)
from src.face_detector import FaceDetector
from src.preprocessor import FacePreprocessor
from src.emotion_model import load_trained_model
from src.utils import FPSCounter, EmotionTracker, save_frame, draw_fancy_bbox, create_overlay


class WebcamEmotionApp:
    """Real-time webcam emotion recognition application."""

    def __init__(self, model_path=None, camera_index=None, detection_method="haar"):
        self.camera_index = camera_index if camera_index is not None else CAMERA_INDEX
        self.model_path = model_path or BEST_MODEL_PATH
        self.show_probabilities = True
        self.is_fullscreen = False

        # Initialize components
        print("=" * 55)
        print("  Real-Time Face Emotion Recognition System")
        print("=" * 55)

        self.face_detector = FaceDetector(method=detection_method)
        self.preprocessor = FacePreprocessor()
        self.model = load_trained_model(self.model_path)
        self.fps_counter = FPSCounter()
        self.tracker = EmotionTracker()

        # Prediction smoothing per face (by approximate position)
        self.smooth_predictions = {}
        self.smooth_alpha = 0.6

        print("[INFO] All components initialized successfully!")
        print("=" * 55)

    def predict_emotion(self, face_img):
        """Predict emotion from a face ROI."""
        processed = self.preprocessor.preprocess(face_img)
        predictions = self.model.predict(processed, verbose=0)[0]
        idx = np.argmax(predictions)
        return {
            "emotion": EMOTION_LABELS[idx],
            "confidence": float(predictions[idx]),
            "probabilities": {EMOTION_LABELS[i]: float(predictions[i]) for i in range(len(EMOTION_LABELS))},
            "color": EMOTION_COLORS.get(EMOTION_LABELS[idx], (255, 255, 255)),
        }

    def process_frame(self, frame):
        """Detect faces and predict emotions in a single frame."""
        results = []
        face_data = self.face_detector.detect_and_extract(frame)

        for bbox, face_roi in face_data:
            try:
                pred = self.predict_emotion(face_roi)
                pred["bbox"] = bbox
                results.append(pred)
                self.tracker.add(pred["emotion"], pred["confidence"])
            except Exception as e:
                print(f"[WARN] Error processing face: {e}")

        return results

    def draw_results(self, frame, predictions):
        """Draw all annotations on the frame."""
        output = frame.copy()

        for pred in predictions:
            x, y, w, h = pred["bbox"]
            color = pred["color"]
            emotion = pred["emotion"]
            confidence = pred["confidence"]

            # Fancy bounding box
            draw_fancy_bbox(output, pred["bbox"], color, thickness=2)

            # Emotion label with background
            label = f"{emotion} {confidence:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.65
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            label_y = max(y - 8, th + 8)
            cv2.rectangle(output, (x, label_y - th - 6), (x + tw + 8, label_y + 4), color, cv2.FILLED)
            text_col = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
            cv2.putText(output, label, (x + 4, label_y - 2), font, font_scale, text_col, thickness, cv2.LINE_AA)

            # Probability sidebar
            if self.show_probabilities:
                bx = x + w + 12
                by_start = y
                bw = 110
                bh = 14
                gap = 2
                for i, (emo, prob) in enumerate(pred["probabilities"].items()):
                    by = by_start + i * (bh + gap)
                    if by + bh > frame.shape[0]:
                        break
                    if bx + bw > frame.shape[1]:
                        break
                    cv2.rectangle(output, (bx, by), (bx + bw, by + bh), (40, 40, 40), cv2.FILLED)
                    fill = int(bw * prob)
                    ec = EMOTION_COLORS.get(emo, (200, 200, 200))
                    cv2.rectangle(output, (bx, by), (bx + fill, by + bh), ec, cv2.FILLED)
                    cv2.putText(output, f"{emo[:3]} {prob:.0%}", (bx + 2, by + bh - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        return output

    def draw_hud(self, frame, predictions, fps):
        """Draw heads-up display overlay."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Top bar
        cv2.rectangle(overlay, (0, 0), (w, 42), (0, 0, 0), cv2.FILLED)
        frame_out = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

        cv2.putText(frame_out, "EMOTION RECOGNITION", (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_out, f"FPS: {fps:.1f}", (w - 130, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame_out, f"Faces: {len(predictions)}", (w - 270, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

        # Bottom bar
        overlay2 = frame_out.copy()
        cv2.rectangle(overlay2, (0, h - 28), (w, h), (0, 0, 0), cv2.FILLED)
        frame_out = cv2.addWeighted(overlay2, 0.65, frame_out, 0.35, 0)
        cv2.putText(frame_out, "Q:Quit | S:Screenshot | P:Probabilities | R:Reset",
            (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

        # Dominant emotion badge
        if self.tracker.history:
            dominant = self.tracker.get_dominant_emotion()
            dist = self.tracker.get_distribution()
            total = sum(dist.values())
            if total > 0:
                pct = dist[dominant] / total
                badge = f"Session: {dominant} ({pct:.0%})"
                cv2.putText(frame_out, badge, (w - 280, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    EMOTION_COLORS.get(dominant, (200, 200, 200)), 1, cv2.LINE_AA)

        return frame_out

    def run(self):
        """Main application loop."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam!")
            print("  Try a different camera index with --camera flag")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        window_name = "Real-Time Emotion Recognition"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1024, 600)

        print("\n[INFO] Webcam started! Press Q or ESC to quit.")
        print("[INFO] Controls: S=Screenshot, P=Toggle Bars, R=Reset\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame")
                break

            # Process
            predictions = self.process_frame(frame)
            fps = self.fps_counter.update()

            # Draw
            output = self.draw_results(frame, predictions)
            output = self.draw_hud(output, predictions, fps)

            cv2.imshow(window_name, output)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                break
            elif key in (ord("s"), ord("S")):
                save_frame(output, prefix="emotion_capture")
            elif key in (ord("p"), ord("P")):
                self.show_probabilities = not self.show_probabilities
                state = "ON" if self.show_probabilities else "OFF"
                print(f"[INFO] Probability bars: {state}")
            elif key in (ord("r"), ord("R")):
                self.tracker.clear()
                print("[INFO] Emotion tracker reset")
            elif key in (ord("f"), ord("F")):
                self.is_fullscreen = not self.is_fullscreen
                if self.is_fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        # Cleanup
        print("\n[INFO] Shutting down...")
        if self.tracker.history:
            self.tracker.save()
            dist = self.tracker.get_distribution()
            print("\n  Session Summary:")
            total = sum(dist.values())
            for emo, count in sorted(dist.items(), key=lambda x: -x[1]):
                if count > 0:
                    print(f"    {EMOTION_EMOJIS.get(emo, '')} {emo}: {count} ({count/total*100:.1f}%)")

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Session ended.")


def main():
    """Entry point with CLI arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="Real-Time Emotion Recognition")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--detector", type=str, default="haar",
                        choices=["haar", "dnn", "hybrid"], help="Face detection method")
    args = parser.parse_args()

    app = WebcamEmotionApp(
        model_path=args.model,
        camera_index=args.camera,
        detection_method=args.detector,
    )
    app.run()


if __name__ == "__main__":
    main()
