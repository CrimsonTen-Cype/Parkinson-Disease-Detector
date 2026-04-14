# 📥 Download Trained Model

Your Parkinson's Detection model has been successfully trained and saved. You can access the model file and the associated optimal threshold using the links below.

### 🚀 Model Files
- **Trained Model**: [parkinsons_model.h5](file:///c:/project/parkinson_project/parkinsons_model.h5)
- **Optimal Threshold**: [best_threshold.txt](file:///c:/project/parkinson_project/best_threshold.txt)

### ℹ️ Model Details
- **Architecture**: CNN (4 Convolutional Blocks + Dense Classifier)
- **Input size**: 128x128 Mel Spectrograms
- **File Format**: Keras/TensorFlow H5

> [!TIP]
> To use this model in your Streamlit application, ensure that the `THRESHOLD` variable in `streamlit_app.py` matches the value found in `best_threshold.txt`.

### 🛠️ How to use
1. Download the `.h5` file.
2. Place it in your project directory.
3. Load it using `tensorflow.keras.models.load_model('parkinsons_model.h5')`.
