# ============================================================
# STREAMLIT APP — Parkinson's Disease Detection
# Deploy in VS Code using: streamlit run app.py
# ============================================================

# Install dependencies first:
# pip install streamlit tensorflow librosa numpy pillow matplotlib

import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import os
import tempfile

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Parkinson's Detection",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CONFIG (must match training config)
# ─────────────────────────────────────────────

CONFIG = {
    "MODEL_PATH": r"C:\project\parkinson_project\parkinsons_model.h5",   # Put model file in same directory
    "SAMPLE_RATE": 22050,
    "DURATION": 3,
    "N_MELS": 128,
    "HOP_LENGTH": 512,
    "N_FFT": 2048,
    "IMG_SIZE": (128, 128),
    "THRESHOLD": 0.9891 # Updated with best_threshold from training
}

# ─────────────────────────────────────────────
# LOAD MODEL (cached)
# ─────────────────────────────────────────────

@st.cache_resource
def load_parkinsons_model(path):
    if not os.path.exists(path):
        return None
    return load_model(path)

model = load_parkinsons_model(CONFIG["MODEL_PATH"])

# ─────────────────────────────────────────────
# PREPROCESSING FUNCTION
# ─────────────────────────────────────────────

def audio_to_melspectrogram(audio_bytes_or_path, config, is_path=False):
    """Convert audio to a normalized Mel Spectrogram."""
    try:
        if is_path:
            y, sr = librosa.load(audio_bytes_or_path,
                                  sr=config["SAMPLE_RATE"],
                                  duration=config["DURATION"], mono=True)
        else:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(audio_bytes_or_path)
                tmp_path = tmp.name
            y, sr = librosa.load(tmp_path,
                                  sr=config["SAMPLE_RATE"],
                                  duration=config["DURATION"], mono=True)
            os.unlink(tmp_path)

        # Pad or trim
        target_length = config["SAMPLE_RATE"] * config["DURATION"]
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]

        # Compute Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_mels=config["N_MELS"],
            n_fft=config["N_FFT"],
            hop_length=config["HOP_LENGTH"]
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Resize
        img = Image.fromarray(mel_spec_db)
        img_resized = img.resize(config["IMG_SIZE"], Image.LANCZOS)
        mel_array = np.array(img_resized, dtype=np.float32)

        # Normalize
        mel_min, mel_max = mel_array.min(), mel_array.max()
        if mel_max - mel_min > 0:
            mel_array = (mel_array - mel_min) / (mel_max - mel_min)

        return mel_array, y, sr

    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None, None


def predict(mel_array, model, config):
    """Run prediction on spectrogram."""
    mel_input = mel_array[np.newaxis, ..., np.newaxis]
    probability = float(model.predict(mel_input, verbose=0)[0][0])
    label = "Parkinson's Disease" if probability >= config["THRESHOLD"] else "Healthy"
    confidence = probability if probability >= config["THRESHOLD"] else (1 - probability)
    return label, probability, confidence


def plot_spectrogram(mel_array):
    """Generate a spectrogram figure."""
    fig, ax = plt.subplots(figsize=(8, 3))
    img = ax.imshow(mel_array, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title('Mel Spectrogram', fontsize=12)
    ax.set_xlabel('Time frames')
    ax.set_ylabel('Mel frequency bins')
    plt.colorbar(img, ax=ax, format='%+.1f')
    plt.tight_layout()
    return fig


def plot_waveform(y, sr):
    """Generate a waveform figure."""
    fig, ax = plt.subplots(figsize=(8, 2))
    times = np.linspace(0, len(y) / sr, num=len(y))
    ax.plot(times, y, color='steelblue', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Audio Waveform', fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────

# Sidebar
with st.sidebar:
    st.image("https://i.imgur.com/placeholder.png", use_column_width=True)  # Replace with your logo
    st.markdown("## About")
    st.info(
        "This app uses a deep learning CNN model trained on Mel Spectrograms "
        "of voice recordings to detect signs of Parkinson's disease."
    )
    st.markdown("---")
    st.markdown("**Model info**")
    st.markdown(f"- Architecture: CNN (4 Conv blocks)")
    st.markdown(f"- Input: 128×128 Mel Spectrogram")
    st.markdown(f"- Threshold: `{CONFIG['THRESHOLD']}`")
    st.markdown("---")
    st.warning(
        "⚠️ **Disclaimer:** This tool is for research purposes only "
        "and should NOT be used as a medical diagnosis."
    )

# Main UI
st.title("🧠 Parkinson's Disease Voice Detector")
st.markdown(
    "Upload a voice recording (WAV or MP3) and the AI model will analyze "
    "it for signs of Parkinson's disease."
)
st.markdown("---")

# Model check
if model is None:
    st.error(
        f"❌ Model not found at `{CONFIG['MODEL_PATH']}`.\n\n"
        "Please place `parkinsons_model.h5` in the same directory as `app.py`."
    )
    st.stop()
else:
    st.success("✅ Model loaded successfully!")

# Audio input selection
tab_upload, tab_record = st.tabs(["📁 Upload Audio", "🎤 Record Live"])

audio_source = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload Voice Recording",
        type=["wav", "mp3", "flac", "ogg"],
        help="Supported formats: WAV, MP3, FLAC, OGG"
    )
    if uploaded_file is not None:
        audio_source = uploaded_file

with tab_record:
    recorded_audio = st.audio_input("Record your voice")
    if recorded_audio is not None:
        audio_source = recorded_audio

if audio_source is not None:
    st.audio(audio_source, format='audio/wav')

    with st.spinner("🔄 Processing audio and running inference..."):
        audio_bytes = audio_source.read()
        mel_array, y_audio, sr = audio_to_melspectrogram(audio_bytes, CONFIG)

    if mel_array is not None:
        # ── Results ──────────────────────────────────
        label, probability, confidence = predict(mel_array, model, CONFIG)

        st.markdown("---")
        st.subheader("🔍 Prediction Result")

        col1, col2, col3 = st.columns(3)

        if label == "Parkinson's Disease":
            col1.error(f"**{label}**")
        else:
            col1.success(f"**{label}**")

        col2.metric("P(Parkinson's)", f"{probability:.1%}")
        col3.metric("Confidence", f"{confidence:.1%}")

        # Probability bar
        st.markdown("**Probability Breakdown**")
        prob_col1, prob_col2 = st.columns(2)
        prob_col1.progress(probability, text=f"Parkinson's: {probability:.1%}")
        prob_col2.progress(1 - probability, text=f"Healthy: {1-probability:.1%}")

        # ── Visualizations ────────────────────────────
        st.markdown("---")
        st.subheader("📊 Audio Analysis")

        tab1, tab2 = st.tabs(["Mel Spectrogram", "Waveform"])

        with tab1:
            fig_spec = plot_spectrogram(mel_array)
            st.pyplot(fig_spec, use_container_width=True)
            st.caption(
                "Mel Spectrogram: darker colors = lower energy. "
                "Parkinson's typically shows irregular patterns in the mid-frequency bands."
            )

        with tab2:
            if y_audio is not None:
                fig_wave = plot_waveform(y_audio, sr)
                st.pyplot(fig_wave, use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:13px;'>"
    "Built with TensorFlow & Streamlit • For research use only"
    "</div>",
    unsafe_allow_html=True
)