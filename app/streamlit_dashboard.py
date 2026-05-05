"""
Streamlit Web Dashboard for Emotion Recognition
=================================================
Browser-based interface with real-time webcam emotion detection,
live charts, and session analytics.

Run: streamlit run app/streamlit_dashboard.py
"""

import os
import sys
import cv2
import numpy as np
import time
from collections import deque
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd

from config import EMOTION_LABELS, EMOTION_COLORS, BEST_MODEL_PATH, IMG_SIZE
from src.face_detector import FaceDetector
from src.preprocessor import FacePreprocessor
from src.emotion_model import load_trained_model


# ─── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Recognition Dashboard",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0;
    }
    .emotion-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-big {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
    }
    .stApp { background-color: #0e1117; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load face detector, preprocessor, and CNN model (cached)."""
    detector = FaceDetector(method="haar")
    preprocessor = FacePreprocessor()
    try:
        model = load_trained_model(BEST_MODEL_PATH)
    except FileNotFoundError:
        model = None
    return detector, preprocessor, model


def predict_emotion(face_img, preprocessor, model):
    """Run emotion prediction on a face ROI."""
    processed = preprocessor.preprocess(face_img)
    preds = model.predict(processed, verbose=0)[0]
    idx = np.argmax(preds)
    return {
        "emotion": EMOTION_LABELS[idx],
        "confidence": float(preds[idx]),
        "probabilities": {EMOTION_LABELS[i]: float(preds[i]) for i in range(len(EMOTION_LABELS))},
    }


def main():
    st.markdown('<div class="main-header">😊 Real-Time Emotion Recognition Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        confidence_threshold = st.slider("Min Confidence", 0.0, 1.0, 0.3, 0.05)
        show_chart = st.checkbox("Show Distribution Chart", value=True)
        show_timeline = st.checkbox("Show Emotion Timeline", value=True)
        st.markdown("---")
        st.header("📊 Model Info")
        st.info(f"**Input Size:** {IMG_SIZE}x{IMG_SIZE}\n\n**Classes:** {len(EMOTION_LABELS)}\n\n**Emotions:** {', '.join(EMOTION_LABELS)}")

    # Load models
    detector, preprocessor, model = load_models()

    if model is None:
        st.error("⚠️ Trained model not found! Please train the model first:")
        st.code("python src/train.py", language="bash")
        st.info("The model will be saved to `models/emotion_model_best.keras`")
        return

    # Layout
    col_video, col_stats = st.columns([2, 1])

    with col_video:
        st.subheader("📹 Live Feed")
        video_placeholder = st.empty()
        status_placeholder = st.empty()

    with col_stats:
        st.subheader("📈 Live Statistics")
        emotion_placeholder = st.empty()
        chart_placeholder = st.empty()
        timeline_placeholder = st.empty()

    # Session state for tracking
    if "emotion_history" not in st.session_state:
        st.session_state.emotion_history = []
    if "running" not in st.session_state:
        st.session_state.running = False

    # Controls
    col_start, col_stop, col_clear = st.columns(3)
    with col_start:
        start_btn = st.button("▶️ Start Webcam", use_container_width=True)
    with col_stop:
        stop_btn = st.button("⏹️ Stop", use_container_width=True)
    with col_clear:
        clear_btn = st.button("🗑️ Clear History", use_container_width=True)

    if clear_btn:
        st.session_state.emotion_history = []
        st.rerun()

    if stop_btn:
        st.session_state.running = False
        st.rerun()

    if start_btn:
        st.session_state.running = True

    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot access webcam!")
            return

        status_placeholder.success("🟢 Webcam active — detecting emotions...")

        frame_count = 0
        try:
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect and predict
                face_data = detector.detect_and_extract(frame)
                predictions = []

                for bbox, face_roi in face_data:
                    try:
                        pred = predict_emotion(face_roi, preprocessor, model)
                        pred["bbox"] = bbox
                        predictions.append(pred)
                    except Exception:
                        continue

                # Annotate frame
                for pred in predictions:
                    x, y, w, h = pred["bbox"]
                    label = f"{pred['emotion']} {pred['confidence']:.0%}"
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                # Track emotions
                for pred in predictions:
                    if pred["confidence"] >= confidence_threshold:
                        st.session_state.emotion_history.append({
                            "emotion": pred["emotion"],
                            "confidence": pred["confidence"],
                            "time": datetime.now().strftime("%H:%M:%S"),
                        })

                # Update stats periodically
                if frame_count % 15 == 0 and predictions:
                    latest = predictions[0]
                    emotion_placeholder.markdown(
                        f"### Current: **{latest['emotion']}** ({latest['confidence']:.0%})"
                    )

                    if show_chart and st.session_state.emotion_history:
                        df = pd.DataFrame(st.session_state.emotion_history)
                        counts = df["emotion"].value_counts()
                        chart_placeholder.bar_chart(counts)

                    if show_timeline and len(st.session_state.emotion_history) > 1:
                        df = pd.DataFrame(st.session_state.emotion_history[-50:])
                        timeline_placeholder.line_chart(df.set_index("time")["confidence"])

                frame_count += 1
                time.sleep(0.033)  # ~30fps cap

        finally:
            cap.release()
            status_placeholder.info("🔴 Webcam stopped.")

    # Show session summary if there's history
    if st.session_state.emotion_history and not st.session_state.running:
        st.markdown("---")
        st.subheader("📋 Session Summary")
        df = pd.DataFrame(st.session_state.emotion_history)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Detections", len(df))
        with c2:
            if len(df) > 0:
                st.metric("Dominant Emotion", df["emotion"].mode().iloc[0])
        with c3:
            if len(df) > 0:
                st.metric("Avg Confidence", f"{df['confidence'].mean():.1%}")

        if show_chart:
            st.bar_chart(df["emotion"].value_counts())


if __name__ == "__main__":
    main()
