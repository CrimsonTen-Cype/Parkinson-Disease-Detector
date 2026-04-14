"""Quick script to inspect the CNN model's label mapping."""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model(r"C:\project\parkinson_project\parkinsons_model.h5")

# Create dummy inputs to see what ranges the model outputs
np.random.seed(42)

# Test with random noise (should be unpredictable)
random_input = np.random.rand(1, 128, 128, 1).astype(np.float32)
prob = model.predict(random_input, verbose=0)[0][0]
print(f"Random noise input -> probability: {prob:.4f}")

# Test with zeros (silence)
zero_input = np.zeros((1, 128, 128, 1), dtype=np.float32)
prob_zero = model.predict(zero_input, verbose=0)[0][0]
print(f"Silence (zeros) input -> probability: {prob_zero:.4f}")

# Test with ones
ones_input = np.ones((1, 128, 128, 1), dtype=np.float32)
prob_ones = model.predict(ones_input, verbose=0)[0][0]
print(f"Ones input -> probability: {prob_ones:.4f}")

# Test with several random samples
print("\n--- 10 random samples ---")
for i in range(10):
    x = np.random.rand(1, 128, 128, 1).astype(np.float32)
    p = model.predict(x, verbose=0)[0][0]
    label = "Parkinson's" if p >= 0.5 else "Healthy"
    print(f"  Sample {i+1}: prob={p:.4f} -> {label}")

print("\nModel output layer:", model.layers[-1].get_config())
print("Model output shape:", model.output_shape)
