import os
import io
import base64
import numpy as np
import librosa
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# --- SETUP PATHS ---
# Using absolute paths to ensure Render finds your folders
base_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

static_path = os.path.join(base_dir, "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# ============================================================
# LOAD MODEL & THRESHOLD
# ============================================================
interpreter = None
if os.path.exists(CONFIG["MODEL_PATH"]):
    try:
        interpreter = tf.lite.Interpreter(model_path=CONFIG["MODEL_PATH"])
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("[OK] TFLite Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")

if os.path.exists(CONFIG["THRESHOLD_PATH"]):
    with open(CONFIG["THRESHOLD_PATH"], "r") as f:
        THRESHOLD = float(f.read().strip())
else:
    THRESHOLD = 0.5

# ============================================================
# AUDIO PROCESSING UTILS
# ============================================================
def process_audio(audio_bytes):
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=CONFIG["SAMPLE_RATE"], duration=CONFIG["DURATION"])
        
        # Standardize length
        target_length = CONFIG["SAMPLE_RATE"] * CONFIG["DURATION"]
        y = np.pad(y, (0, max(0, target_length - len(y))), mode='constant')[:target_length]

        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=CONFIG["N_MELS"], 
            n_fft=CONFIG["N_FFT"], hop_length=CONFIG["HOP_LENGTH"]
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Generate Base64 Spectrogram for UI
        plt.figure(figsize=(4, 4))
        plt.axis('off')
        plt.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='magma')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        spec_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Prepare for Model
        img = Image.fromarray(mel_spec_db).convert('L').resize(CONFIG["IMG_SIZE"])
        mel_array = np.array(img, dtype=np.float32)
        mel_array = (mel_array - mel_array.min()) / (mel_array.max() - mel_array.min() + 1e-8)
        
        return mel_array, spec_base64
    except Exception as e:
        print(f"Processing error: {e}")
        return None, None

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # CRITICAL FIX: Explicitly name the arguments to avoid 'unhashable type: dict'
    return templates.TemplateResponse(
        name="index.html", 
        context={"request": request}
    )

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if not interpreter:
        return JSONResponse(status_code=500, content={"error": "Model not loaded."})

    try:
        content = await file.read()
        mel_array, spec_base64 = process_audio(content)

        if mel_array is None:
            return JSONResponse(status_code=400, content={"error": "Processing failed."})

        # Run TFLite Inference
        input_data = mel_array[np.newaxis, ..., np.newaxis]
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        prob = float(output_data[0][0])
        label = "Parkinson's Disease" if prob >= THRESHOLD else "Healthy"

        return {
            "label": label,
            "probability": round(prob, 4),
            "confidence": round(prob if prob >= THRESHOLD else (1 - prob), 4),
            "spectrogram": spec_base64
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
