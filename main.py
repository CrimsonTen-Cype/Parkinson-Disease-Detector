import os
import io
import numpy as np
import librosa
import tensorflow as tf  # This is the change
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    "MODEL_PATH": "model.tflite",
    "THRESHOLD_PATH": "best_threshold.txt",
    "SAMPLE_RATE": 22050,
    "DURATION": 3,
    "N_MELS": 128,
    "HOP_LENGTH": 512,
    "N_FFT": 2048,
    "IMG_SIZE": (128, 128)
}

app = FastAPI(title="NeuroScan: Parkinson's AI Detector")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ============================================================
# LOAD MODEL & THRESHOLD
# ============================================================
if not os.path.exists(CONFIG["MODEL_PATH"]):
    print(f"ERROR: Model not found at {CONFIG['MODEL_PATH']}")
    model = None
else:
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("[OK] TFLite Model loaded successfully.")

if os.path.exists(CONFIG["THRESHOLD_PATH"]):
    with open(CONFIG["THRESHOLD_PATH"], "r") as f:
        THRESHOLD = float(f.read().strip())
    print(f"[OK] Threshold loaded: {THRESHOLD}")
else:
    THRESHOLD = 0.5
    print("[WARNING] Threshold file not found. Using default 0.5")

# ============================================================
# UTILS
# ============================================================
def process_audio(audio_bytes):
    """Convert audio bytes to normalized 128x128 Mel Spectrogram and return Base64 image."""
    try:
        # Load audio using librosa
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=CONFIG["SAMPLE_RATE"], duration=CONFIG["DURATION"])

        # Pad or trim to exact duration
        target_length = CONFIG["SAMPLE_RATE"] * CONFIG["DURATION"]
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]

        # Compute Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_mels=CONFIG["N_MELS"],
            n_fft=CONFIG["N_FFT"],
            hop_length=CONFIG["HOP_LENGTH"]
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 1. Generate Visualization (Base64)
        plt.figure(figsize=(4, 4))
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='magma')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        spec_base64 = base64.b64encode(buf.read()).decode('utf-8')

        # 2. Process for Model (Resize & Normalize)
        img = Image.fromarray(mel_spec_db)
        img_resized = img.resize(CONFIG["IMG_SIZE"], Image.LANCZOS)
        mel_array = np.array(img_resized, dtype=np.float32)

        mel_min, mel_max = mel_array.min(), mel_array.max()
        if mel_max - mel_min > 0:
            mel_array = (mel_array - mel_min) / (mel_max - mel_min)
        
        return mel_array, spec_base64

    except Exception as e:
        print(f"Error processing audio: {e}")
        return None, None

# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if interpreter is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded on server."})

    try:
        content = await file.read()
        mel_array, spec_base64 = process_audio(content)

        if mel_array is None:
            return JSONResponse(status_code=400, content={"error": "Invalid audio file or processing failed."})

        # Inference (TFLite)
        mel_input = mel_array[np.newaxis, ..., np.newaxis]
        interpreter.set_tensor(input_details[0]['index'], mel_input)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        probability = float(output_data[0][0])

        label = "Parkinson's Disease" if probability >= THRESHOLD else "Healthy"
        confidence = probability if probability >= THRESHOLD else (1 - probability)

        return {
            "label": label,
            "probability": round(probability, 4),
            "confidence": round(confidence, 4),
            "threshold": THRESHOLD,
            "spectrogram": spec_base64
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
