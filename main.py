import os
import io
import base64
import numpy as np
import librosa
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Request
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
    "DURATION": 6,  # Force analysis to first 6 seconds
    "N_MELS": 128,
    "HOP_LENGTH": 512,
    "N_FFT": 2048,
    "IMG_SIZE": (128, 128)
}

app = FastAPI(title="NeuroScan: Parkinson's AI Detector")

base_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(base_dir, "templates")
static_dir = os.path.join(base_dir, "static")

templates = Jinja2Templates(directory=templates_dir)

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ============================================================
# LOAD MODEL & THRESHOLD
# ============================================================
interpreter = None
try:
    if os.path.exists(CONFIG["MODEL_PATH"]):
        interpreter = tf.lite.Interpreter(model_path=CONFIG["MODEL_PATH"])
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("[OK] TFLite Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")

if os.path.exists(CONFIG["THRESHOLD_PATH"]):
    with open(CONFIG["THRESHOLD_PATH"], "r") as f:
        THRESHOLD = float(f.read().strip())
else:
    THRESHOLD = 0.5

# ============================================================
# AUDIO PROCESSING UTILS (FORCED DURATION)
# ============================================================
def process_audio(audio_bytes):
    try:
        # DURATION parameter forces librosa to stop reading after 6 seconds
        y, sr = librosa.load(io.BytesIO(audio_bytes), 
                             sr=CONFIG["SAMPLE_RATE"], 
                             duration=CONFIG["DURATION"])
        
        target_length = CONFIG["SAMPLE_RATE"] * CONFIG["DURATION"]
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]

        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_mels=CONFIG["N_MELS"],
            n_fft=CONFIG["N_FFT"],
            hop_length=CONFIG["HOP_LENGTH"]
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Generate Spectrogram Image for UI
        plt.figure(figsize=(4, 4))
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='magma')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close('all') # Essential to free memory on Render
        buf.seek(0)
        spec_base64 = base64.b64encode(buf.read()).decode('utf-8')

        # Prepare Mel-array for CNN (0-1 Normalization)
        img = Image.fromarray(((mel_spec_db - mel_spec_db.min()) / 
                               (mel_spec_db.max() - mel_spec_db.min()) * 255).astype(np.uint8))
        img_resized = img.resize(CONFIG["IMG_SIZE"], Image.Resampling.LANCZOS)
        mel_array = np.array(img_resized, dtype=np.float32) / 255.0
        
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
        return JSONResponse(status_code=500, content={"error": "Model not loaded."})

    try:
        content = await file.read()
        mel_array, spec_base64 = process_audio(content)

        if mel_array is None:
            return JSONResponse(status_code=400, content={"error": "Analysis failed."})

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
            "spectrogram": spec_base64
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
