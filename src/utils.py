"""
Utility Functions
==================
Helper functions for the emotion recognition system.
"""

import os
import sys
import cv2
import numpy as np
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMOTION_LABELS, RESULTS_DIR


def get_timestamp():
    """Get formatted timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_frame(frame, directory=None, prefix="capture"):
    """Save a frame to disk with timestamp."""
    directory = directory or RESULTS_DIR
    os.makedirs(directory, exist_ok=True)
    filename = f"{prefix}_{get_timestamp()}.png"
    filepath = os.path.join(directory, filename)
    cv2.imwrite(filepath, frame)
    print(f"[INFO] Frame saved: {filepath}")
    return filepath


def calculate_fps(prev_time):
    """Calculate FPS from previous timestamp."""
    current_time = cv2.getTickCount()
    time_diff = (current_time - prev_time) / cv2.getTickFrequency()
    fps = 1.0 / time_diff if time_diff > 0 else 0
    return fps, current_time


class FPSCounter:
    """Smoothed FPS counter using exponential moving average."""

    def __init__(self, smoothing=0.9):
        self.smoothing = smoothing
        self.fps = 0
        self.prev_time = cv2.getTickCount()

    def update(self):
        current_time = cv2.getTickCount()
        time_diff = (current_time - self.prev_time) / cv2.getTickFrequency()
        instant_fps = 1.0 / time_diff if time_diff > 0 else 0
        self.fps = self.smoothing * self.fps + (1 - self.smoothing) * instant_fps
        self.prev_time = current_time
        return self.fps

    def get(self):
        return self.fps


class EmotionTracker:
    """Track emotions over time for analytics."""

    def __init__(self, max_history=500):
        self.history = []
        self.max_history = max_history

    def add(self, emotion, confidence, timestamp=None):
        timestamp = timestamp or datetime.now().isoformat()
        self.history.append({
            "emotion": emotion,
            "confidence": confidence,
            "timestamp": timestamp,
        })
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_distribution(self):
        """Get emotion distribution counts."""
        dist = {e: 0 for e in EMOTION_LABELS}
        for entry in self.history:
            dist[entry["emotion"]] = dist.get(entry["emotion"], 0) + 1
        return dist

    def get_dominant_emotion(self):
        """Get the most frequent emotion."""
        dist = self.get_distribution()
        if not dist:
            return "Neutral"
        return max(dist, key=dist.get)

    def get_recent(self, n=10):
        """Get the last n predictions."""
        return self.history[-n:]

    def save(self, filepath=None):
        """Save tracking history to JSON."""
        filepath = filepath or os.path.join(RESULTS_DIR, f"emotion_history_{get_timestamp()}.json")
        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[INFO] History saved: {filepath}")

    def clear(self):
        """Clear history."""
        self.history = []


def draw_fancy_bbox(frame, bbox, color, thickness=2, corner_length=20):
    """Draw a stylized bounding box with corner accents."""
    x, y, w, h = bbox
    x2, y2 = x + w, y + h

    # Draw corners instead of full rectangle
    # Top-left
    cv2.line(frame, (x, y), (x + corner_length, y), color, thickness + 1)
    cv2.line(frame, (x, y), (x, y + corner_length), color, thickness + 1)
    # Top-right
    cv2.line(frame, (x2, y), (x2 - corner_length, y), color, thickness + 1)
    cv2.line(frame, (x2, y), (x2, y + corner_length), color, thickness + 1)
    # Bottom-left
    cv2.line(frame, (x, y2), (x + corner_length, y2), color, thickness + 1)
    cv2.line(frame, (x, y2), (x, y2 - corner_length), color, thickness + 1)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, thickness + 1)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, thickness + 1)

    # Thin connecting lines
    cv2.rectangle(frame, (x, y), (x2, y2), color, 1)

    return frame


def create_overlay(frame, predictions, fps=None):
    """Create a professional HUD overlay on the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent header
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, overlay)

    # Title
    cv2.putText(overlay, "Emotion Recognition System",
        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # FPS
    if fps is not None:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(overlay, fps_text,
            (w - 150, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Face count
    face_text = f"Faces: {len(predictions)}"
    cv2.putText(overlay, face_text,
        (w - 300, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    # Footer with instructions
    cv2.rectangle(overlay, (0, h - 30), (w, h), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, overlay)
    cv2.putText(overlay, "Q: Quit | S: Screenshot | P: Toggle Probabilities | R: Reset Tracker",
        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    return overlay
