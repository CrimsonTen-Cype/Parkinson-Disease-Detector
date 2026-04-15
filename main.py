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
# CONFIGURATION - SPEED OPTIMIZED
# ============================================================
CONFIG = {
    "MODEL_PATH": "model.tflite",
    "THRESHOLD_PATH": "best_threshold.txt",
    "SAMPLE_RATE": 16000,
    "DURATION": 6,          # Hard limit to 6 seconds for speed
    "N_MELS": 64,
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
# LOAD MODEL
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
    print(f"Model load error: {e}")

THRESHOLD = 0.5
if os.path.exists(CONFIG["THRESHOLD_PATH"]):
    with open(CONFIG["THRESHOLD_PATH"], "r") as f:
        THRESHOLD = float(f.read().strip())

# ============================================================
# FAST AUDIO PROCESSING
# ============================================================
def process_audio(audio_bytes):
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), 
                             sr=CONFIG["SAMPLE_RATE"], 
                             duration=CONFIG["DURATION"],
                             res_type='kaiser_fast')

        target_length = CONFIG["SAMPLE_RATE"] * CONFIG["DURATION"]
        y = np.pad(y, (0, max(0, target_length - len(y))), mode='constant')[:target_length]

        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=CONFIG["N_MELS"], 
            n_fft=CONFIG["N_FFT"], hop_length=CONFIG["HOP_LENGTH"]
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        plt.figure(figsize=(2, 2), dpi=50) 
        plt.axis('off')
        plt.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='magma')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close('all') 
        buf.seek(0)
        spec_base64 = base64.b64encode(buf.read()).decode('utf-8')

        mel_min, mel_max = mel_spec_db.min(), mel_spec_db.max()
        mel_norm = (mel_spec_db - mel_min) / (mel_max - mel_min) if (mel_max - mel_min) > 0 else mel_spec_db
        
        img = Image.fromarray((mel_norm * 255).astype(np.uint8))
        img_resized = img.resize(CONFIG["IMG_SIZE"], Image.Resampling.NEAREST)
        mel_array = np.array(img_resized, dtype=np.float32) / 255.0
        
        return mel_array, spec_base64
    except Exception as e:
        print(f"Process Error: {e}")
        return None, None

# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if not interpreter:
        return JSONResponse(status_code=500, content={"error": "AI Engine Offline"})

    try:
        content = await file.read()
        mel_array, spec_base64 = process_audio(content)

        if mel_array is None:
            return JSONResponse(status_code=400, content={"error": "Analysis failed."})

        input_data = mel_array[np.newaxis, ..., np.newaxis]
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        probability = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
        label = "Parkinson's Disease" if probability >= THRESHOLD else "Healthy"
        confidence = probability if probability >= THRESHOLD else (1 - probability)

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "spectrogram": spec_base64
        }
    except Exception as e:
        # Syntax error was here; now correctly closed.
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
