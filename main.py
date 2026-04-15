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
# CONFIGURATION - TUNED FOR SPEED
# ============================================================
CONFIG = {
    "MODEL_PATH": "model.tflite",
    "THRESHOLD_PATH": "best_threshold.txt",
    "SAMPLE_RATE": 16000,   # Lowered to 16k to make math 30% faster
    "DURATION": 6,          # HARD LIMIT: Only reads first 6 seconds
    "N_MELS": 64,           # Reduced from 128 for lightning-fast FFT
    "HOP_LENGTH": 512,
    "N_FFT": 1024,
    "IMG_SIZE": (128, 128)
}

app = FastAPI(title="NeuroScan")

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
input_details = None
output_details = None

try:
    if os.path.exists(CONFIG["MODEL_PATH"]):
        interpreter = tf.lite.Interpreter(model_path=CONFIG["MODEL_PATH"])
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("[OK] TFLite Model loaded.")
except Exception as e:
    print(f"[ERROR] Model load failed: {e}")

THRESHOLD = 0.5
if os.path.exists(CONFIG["THRESHOLD_PATH"]):
    with open(CONFIG["THRESHOLD_PATH"], "r") as f:
        THRESHOLD = float(f.read().strip())

# ============================================================
# FAST AUDIO PROCESSING
# ============================================================
def process_audio(audio_bytes):
    try:
        # Load only the first 6 seconds using the fastest possible resampler
        y, sr = librosa.load(io.BytesIO(audio_bytes), 
                             sr=CONFIG["SAMPLE_RATE"], 
                             duration=CONFIG["DURATION"],
                             res_type='kaiser_fast')

        # Padding/Trimming to exactly 6 seconds
        target_length = CONFIG["SAMPLE_RATE"] * CONFIG["DURATION"]
        y = np.pad(y, (0, max(0, target_length - len(y))), mode='constant')[:target_length]

        # Generate Spectrogram (Small & Fast)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=CONFIG["N_MELS"], 
            n_fft=CONFIG["N_FFT"], hop_length=CONFIG["HOP_LENGTH"]
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Fast Image Generation for UI (Low DPI = Low CPU use)
        plt.figure(figsize=(2, 2), dpi=50) 
        plt.axis('off')
        plt.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='magma')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close('all') # Essential: frees RAM immediately
        buf.seek(0)
        spec_base64 = base64.b64encode(buf.read()).decode('utf-8')

        # Prepare Mel-array for CNN (Normalization)
        # We normalize 0-1 to keep the AI from "freezing" on extreme values
        mel_min, mel_max = mel_spec_db.min(), mel_spec_db.max()
        mel_norm = (mel_spec_db - mel_min) / (mel_max - mel_min) if (mel_max - mel_min) > 0 else mel_spec_db
        
        img = Image.fromarray((mel_norm * 255).astype(np.uint8))
        img_resized = img.resize(CONFIG["IMG_SIZE"], Image.Resampling.NEAREST) # Fast resize
        mel_array = np.array(img_resized, dtype=np.float32) / 255.0
        
        return mel_array, spec_base64

    except Exception as e:
        print(f"Processing Error: {e}")
        return None, None

# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # FIXED: Corrected the request argument that was crashing Render
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if not interpreter:
        return JSONResponse(status
