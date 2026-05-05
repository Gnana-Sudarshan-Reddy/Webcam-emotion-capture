"""
Video File Emotion Analyzer
=============================
Analyze pre-recorded video files for emotions.
Outputs annotated video and emotion statistics.

Usage: python app/video_analyzer.py --input video.mp4 --output result.mp4
"""

import os
import sys
import cv2
import numpy as np
import json
from datetime import datetime
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import EMOTION_LABELS, EMOTION_COLORS, RESULTS_DIR, BEST_MODEL_PATH
from src.face_detector import FaceDetector
from src.preprocessor import FacePreprocessor
from src.emotion_model import load_trained_model
from src.utils import draw_fancy_bbox


def analyze_video(input_path, output_path=None, model_path=None, show_preview=True):
    """
    Analyze a video file for emotions.

    Args:
        input_path (str): Path to input video.
        output_path (str): Path for output annotated video.
        model_path (str): Path to trained model.
        show_preview (bool): Show live preview window.
    """
    if not os.path.exists(input_path):
        print(f"[ERROR] Video not found: {input_path}")
        return

    # Initialize
    detector = FaceDetector(method="haar")
    preprocessor = FacePreprocessor()
    model = load_trained_model(model_path or BEST_MODEL_PATH)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        return

    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Output writer
    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(RESULTS_DIR, f"{base}_analyzed.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Tracking
    emotion_log = []
    frame_num = 0

    print(f"[INFO] Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        face_data = detector.detect_and_extract(frame)

        for bbox, face_roi in face_data:
            try:
                processed = preprocessor.preprocess(face_roi)
                preds = model.predict(processed, verbose=0)[0]
                idx = np.argmax(preds)
                emotion = EMOTION_LABELS[idx]
                confidence = float(preds[idx])

                x, y, w, h = bbox
                color = EMOTION_COLORS.get(emotion, (200, 200, 200))
                draw_fancy_bbox(frame, bbox, color)
                label = f"{emotion} {confidence:.0%}"
                cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                emotion_log.append({
                    "frame": frame_num,
                    "time_sec": frame_num / fps,
                    "emotion": emotion,
                    "confidence": confidence,
                })
            except Exception:
                continue

        # Progress bar
        progress = frame_num / total_frames * 100
        cv2.putText(frame, f"Frame {frame_num}/{total_frames} ({progress:.0f}%)",
            (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        writer.write(frame)

        if show_preview:
            preview = cv2.resize(frame, (800, 450))
            cv2.imshow("Video Analysis", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_num % 100 == 0:
            print(f"  Processed {frame_num}/{total_frames} frames ({progress:.0f}%)")

    # Cleanup
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\n[INFO] Annotated video saved: {output_path}")

    # Save statistics
    if emotion_log:
        counts = Counter(e["emotion"] for e in emotion_log)
        total = sum(counts.values())
        print(f"\n  Video Emotion Summary ({total} detections):")
        for emo, count in counts.most_common():
            print(f"    {emo}: {count} ({count/total*100:.1f}%)")

        stats_path = os.path.splitext(output_path)[0] + "_stats.json"
        with open(stats_path, "w") as f:
            json.dump({"summary": dict(counts), "log": emotion_log}, f, indent=2)
        print(f"  Stats saved: {stats_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze video for emotions")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", default=None, help="Output video path")
    parser.add_argument("--model", default=None, help="Model path")
    parser.add_argument("--no-preview", action="store_true", help="Disable preview")
    args = parser.parse_args()

    analyze_video(args.input, args.output, args.model, show_preview=not args.no_preview)
