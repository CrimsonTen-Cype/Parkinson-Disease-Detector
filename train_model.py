# ============================================================
# IMPROVED LOCAL TRAINING — Parkinson's Disease CNN
# Combines multiple datasets, removes duplicates, better training
# ============================================================

import os
import random
import hashlib
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

print("[OK] All libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    "MODEL_SAVE_PATH": r"C:\project\parkinson_project\parkinsons_model.h5",

    # Audio preprocessing (must match streamlit_app.py)
    "SAMPLE_RATE": 22050,
    "DURATION": 3,
    "N_MELS": 128,
    "HOP_LENGTH": 512,
    "N_FFT": 2048,
    "IMG_SIZE": (128, 128),

    # Model
    "BATCH_SIZE": 16,
    "EPOCHS": 80,
    "LEARNING_RATE": 5e-4,
    "DROPOUT_RATE": 0.5,

    # Split
    "VAL_SPLIT": 0.15,
    "TEST_SPLIT": 0.15,
}

# All data sources: list of (folder_path, label)
DATA_SOURCES = [
    # Documents Dataset
    (r"C:\Users\91999\Documents\Dataset\healthy", 0),
    (r"C:\Users\91999\Documents\Dataset\Parkinson's deasease", 1),

    # Archive Augmented flat folders
    (r"C:\Users\91999\Downloads\archive (1)\data\Augmented\Healthy", 0),
    (r"C:\Users\91999\Downloads\archive (1)\data\Augmented\Parkinson", 1),

    # Archive Augmented pre-split
    (r"C:\Users\91999\Downloads\archive (1)\data\Augmented\train\Healthy", 0),
    (r"C:\Users\91999\Downloads\archive (1)\data\Augmented\train\Parkinson", 1),
    (r"C:\Users\91999\Downloads\archive (1)\data\Augmented\val\Healthy", 0),
    (r"C:\Users\91999\Downloads\archive (1)\data\Augmented\val\Parkinson", 1),
    (r"C:\Users\91999\Downloads\archive (1)\data\Augmented\test\Healthy", 0),
    (r"C:\Users\91999\Downloads\archive (1)\data\Augmented\test\Parkinson", 1),

    # Archive Raw Audio
    (r"C:\Users\91999\Downloads\archive (1)\data\Raw Audio\Healthy", 0),
    (r"C:\Users\91999\Downloads\archive (1)\data\Raw Audio\Parkinson", 1),
    (r"C:\Users\91999\Downloads\archive (1)\data\Raw Audio\Parkinson Dialogue", 1),
    (r"C:\Users\91999\Downloads\archive (1)\data\Raw Audio\Parkinson Read", 1),
]

print("[OK] Configuration set!")


# ─────────────────────────────────────────────
# AUDIO -> MEL SPECTROGRAM
# ─────────────────────────────────────────────

def audio_to_melspectrogram(file_path, config):
    """Load audio and convert to normalized 128x128 Mel Spectrogram."""
    try:
        y, sr = librosa.load(
            file_path,
            sr=config["SAMPLE_RATE"],
            duration=config["DURATION"],
            mono=True
        )

        # Skip very short/silent audio
        if len(y) < config["SAMPLE_RATE"] * 0.5:  # less than 0.5s
            return None
        if np.max(np.abs(y)) < 0.01:  # essentially silence
            return None

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

        # Resize to fixed shape
        img = Image.fromarray(mel_spec_db)
        img_resized = img.resize(config["IMG_SIZE"], Image.LANCZOS)
        mel_array = np.array(img_resized, dtype=np.float32)

        # Normalize to [0, 1]
        mel_min, mel_max = mel_array.min(), mel_array.max()
        if mel_max - mel_min > 0:
            mel_array = (mel_array - mel_min) / (mel_max - mel_min)
        else:
            return None  # Skip degenerate spectrograms

        return mel_array

    except Exception as e:
        return None


# ─────────────────────────────────────────────
# FILE HASH FOR DEDUPLICATION
# ─────────────────────────────────────────────

def file_hash(filepath):
    """Quick hash of first 64KB of file for deduplication."""
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        h.update(f.read(65536))
    return h.hexdigest()


# ─────────────────────────────────────────────
# LOAD ALL DATASETS (with deduplication)
# ─────────────────────────────────────────────

def load_all_data(data_sources, config):
    """Load from multiple folders, deduplicate, and return X, y."""
    X, y = [], []
    seen_hashes = set()
    total_files = 0
    duplicates = 0
    errors = 0

    for folder_path, label in data_sources:
        if not os.path.exists(folder_path):
            print(f"  [SKIP] Not found: {folder_path}")
            continue

        files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]

        label_name = "Healthy" if label == 0 else "Parkinson's"
        print(f"  [{label_name}] {folder_path} ({len(files)} files)")

        folder_added = 0
        for fname in files:
            total_files += 1
            fpath = os.path.join(folder_path, fname)

            # Deduplicate
            fh = file_hash(fpath)
            if fh in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(fh)

            mel = audio_to_melspectrogram(fpath, config)
            if mel is not None:
                X.append(mel)
                y.append(label)
                folder_added += 1
            else:
                errors += 1

        print(f"    -> Added {folder_added} unique samples")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"\n[OK] Dataset loaded!")
    print(f"   Total files scanned: {total_files}")
    print(f"   Duplicates removed: {duplicates}")
    print(f"   Errors/skipped: {errors}")
    print(f"   Final samples: {len(X)}")
    print(f"   Parkinson's (1): {np.sum(y == 1)} | Healthy (0): {np.sum(y == 0)}")

    return X, y


# ─────────────────────────────────────────────
# BUILD IMPROVED CNN MODEL
# ─────────────────────────────────────────────

def build_model(input_shape=(128, 128, 1), dropout_rate=0.5, lr=5e-4):
    """Improved CNN with stronger regularization."""
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Block 4
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Classifier
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, kernel_regularizer=regularizers.l2(1e-3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),

        layers.Dense(32, kernel_regularizer=regularizers.l2(1e-3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),

        # Output: probability of Parkinson's (1 = PD, 0 = Healthy)
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    return model


# ─────────────────────────────────────────────
# FIND BEST THRESHOLD
# ─────────────────────────────────────────────

def find_best_threshold(model, X_val, y_val):
    """Find optimal threshold using Youden's J statistic."""
    y_prob = model.predict(X_val, verbose=0).flatten()
    fpr, tpr, thresholds = roc_curve(y_val, y_prob)

    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = float(thresholds[best_idx])

    print(f"\n[OK] Best threshold: {best_threshold:.4f}")
    print(f"   TPR (Sensitivity): {tpr[best_idx]:.4f}")
    print(f"   FPR: {fpr[best_idx]:.4f}")

    return best_threshold


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # === STEP 1: Load all data ===
    print("\n" + "="*60)
    print("  STEP 1: Loading all datasets (with deduplication)...")
    print("="*60 + "\n")
    X, y = load_all_data(DATA_SOURCES, CONFIG)

    if len(X) < 20:
        print("[ERROR] Not enough data loaded!")
        exit(1)

    # === STEP 2: Split ===
    print("\n" + "="*60)
    print("  STEP 2: Splitting dataset...")
    print("="*60)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=CONFIG["TEST_SPLIT"], random_state=SEED, stratify=y
    )
    val_fraction = CONFIG["VAL_SPLIT"] / (1 - CONFIG["TEST_SPLIT"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_fraction,
        random_state=SEED, stratify=y_train_val
    )

    # Add channel dimension
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    print(f"   Train: {X_train.shape} | PD: {np.sum(y_train==1)} | H: {np.sum(y_train==0)}")
    print(f"   Val  : {X_val.shape}   | PD: {np.sum(y_val==1)} | H: {np.sum(y_val==0)}")
    print(f"   Test : {X_test.shape}  | PD: {np.sum(y_test==1)} | H: {np.sum(y_test==0)}")

    # === STEP 3: Class weights ===
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"   Class weights: {class_weight_dict}")

    # === STEP 4: Build model ===
    print("\n" + "="*60)
    print("  STEP 3: Building CNN model...")
    print("="*60)

    model = build_model(
        input_shape=(128, 128, 1),
        dropout_rate=CONFIG["DROPOUT_RATE"],
        lr=CONFIG["LEARNING_RATE"]
    )
    model.summary()
    print(f"\n   Total parameters: {model.count_params():,}")

    # === STEP 5: Callbacks ===
    callbacks = [
        EarlyStopping(
            monitor='val_auc', mode='max',
            patience=20, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=8, min_lr=1e-6, verbose=1
        ),
        ModelCheckpoint(
            filepath=CONFIG["MODEL_SAVE_PATH"],
            monitor='val_auc', save_best_only=True,
            mode='max', verbose=1
        )
    ]

    # === STEP 6: Train ===
    print("\n" + "="*60)
    print("  STEP 4: Training the model...")
    print("="*60)

    history = model.fit(
        X_train, y_train,
        batch_size=CONFIG["BATCH_SIZE"],
        epochs=CONFIG["EPOCHS"],
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    print("\n[OK] Training complete!")

    # === STEP 7: Evaluate ===
    print("\n" + "="*60)
    print("  STEP 5: Evaluating on test set...")
    print("="*60)

    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    test_loss, test_acc, test_prec, test_rec, test_auc = model.evaluate(
        X_test, y_test, verbose=0
    )
    f1 = f1_score(y_test, y_pred)

    print(f"\n   Accuracy  : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Precision : {test_prec:.4f}")
    print(f"   Recall    : {test_rec:.4f}")
    print(f"   F1-Score  : {f1:.4f}")
    print(f"   AUC-ROC   : {test_auc:.4f}")
    print(f"   Loss      : {test_loss:.4f}")

    print("\nClassification Report (threshold=0.5):")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Parkinson's"]))

    # === STEP 8: Best threshold ===
    print("\n" + "="*60)
    print("  STEP 6: Finding best threshold...")
    print("="*60)

    best_threshold = find_best_threshold(model, X_val, y_val)

    y_pred_opt = (y_prob >= best_threshold).astype(int)
    print(f"\n   With optimal threshold ({best_threshold:.4f}):")
    print(classification_report(y_test, y_pred_opt, target_names=["Healthy", "Parkinson's"]))

    # Save threshold
    threshold_path = os.path.join(
        os.path.dirname(CONFIG["MODEL_SAVE_PATH"]), "best_threshold.txt"
    )
    with open(threshold_path, 'w') as f:
        f.write(str(best_threshold))
    print(f"[OK] Threshold saved to: {threshold_path}")

    # Final save
    model.save(CONFIG["MODEL_SAVE_PATH"])
    print(f"[OK] Model saved to: {CONFIG['MODEL_SAVE_PATH']}")

    print("\n" + "="*60)
    print("  DONE!")
    print("="*60)
    print(f"\n   Model: {CONFIG['MODEL_SAVE_PATH']}")
    print(f"   Best Threshold: {best_threshold:.4f}")
    print(f"\n   Next: Update THRESHOLD in streamlit_app.py to {best_threshold:.4f}")
    print(f"   Then: streamlit run streamlit_app.py")
