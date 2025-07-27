# quantize.py
import joblib
import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os

# Define paths for saving parameters
UNQUANTIZED_PARAMS_PATH = 'unquant_params.joblib'
QUANTIZED_PARAMS_PATH = 'quant_params.joblib'
SKLEARN_MODEL_PATH = 'linear_regression_model.joblib'

print("--- Starting Model Quantization Process ---")

# Step 0: Ensure the scikit-learn model is available
if not os.path.exists(SKLEARN_MODEL_PATH):
    print(f"Error: Scikit-learn model '{SKLEARN_MODEL_PATH}' not found. Please run train.py first.")
    exit(1)

sklearn_model = None
try:
    # Load the saved scikit-learn model
    sklearn_model = joblib.load(SKLEARN_MODEL_PATH)
    print(f"Scikit-learn model loaded successfully from {SKLEARN_MODEL_PATH}")
except Exception as e:
    print(f"Failed to load scikit-learn model from {SKLEARN_MODEL_PATH}: {e}")
    exit(1)

# Explicit check after loading attempt
if sklearn_model is None:
    print("Critical Error: sklearn_model could not be loaded and is None. Exiting.")
    exit(1)

# Extract its learned parameters (coef and intercept_)
sklearn_coef = sklearn_model.coef_
sklearn_intercept = sklearn_model.intercept_

# Store the unquantized parameters in a dictionary and save
unquantized_params = {
    'coef': sklearn_coef,
    'intercept': sklearn_intercept
}
joblib.dump(unquantized_params, UNQUANTIZED_PARAMS_PATH)


# --- Manual Quantization to unsigned 8-bit integer ---

# Convert parameters to PyTorch tensors
weights_fp32 = torch.tensor(sklearn_coef).float().unsqueeze(0) # (1, 8) for PyTorch Linear
bias_fp32 = torch.tensor(sklearn_intercept).float()

print(f"\nOriginal FP32 Weights (first 5): {weights_fp32.flatten()[:5].tolist()}")
print(f"Original FP32 Bias: {bias_fp32.item()}")
print(f"FP32 Weights Range: [{weights_fp32.min().item():.4f}, {weights_fp32.max().item():.4f}]")
print(f"FP32 Bias Range: [{bias_fp32.min().item():.4f}, {bias_fp32.max().item():.4f}]")


# Define quantization range for unsigned 8-bit
q_min, q_max = 0, 255

# --- Quantization Function ---
def quantize_tensor(tensor_fp32, q_min, q_max):
    r_min = tensor_fp32.min().item()
    r_max = tensor_fp32.max().item()

    # Handle cases where range is zero (all values are identical)
    if r_max == r_min:
        scale = 1e-9 # Small non-zero scale to prevent division by zero
        zero_point = q_min # Map to q_min
    else:
        scale = (r_max - r_min) / (q_max - q_min)
        zero_point = q_min - round(r_min / scale)

    # Ensure zero_point is within the integer range [q_min, q_max]
    zero_point = int(np.clip(zero_point, q_min, q_max))

    # Quantize: Q = round(R / S + Z)
    quantized_float = torch.round(tensor_fp32 / scale + zero_point)
    
    # Clip to target integer range and convert to uint8
    quantized_int8 = torch.clamp(quantized_float, q_min, q_max).to(torch.uint8)
    
    return quantized_int8, scale, zero_point, r_min, r_max

# Quantize weights and bias
quantized_weights_int8, scale_w, zero_point_w, min_val_w, max_val_w = quantize_tensor(weights_fp32, q_min, q_max)
quantized_bias_int8, scale_b, zero_point_b, min_val_b, max_val_b = quantize_tensor(bias_fp32, q_min, q_max)


print(f"\nQuantization Parameters for Weights: Min_R={min_val_w:.4f}, Max_R={max_val_w:.4f}, Scale={scale_w:.6f}, ZeroPoint={zero_point_w}")
print(f"Quantization Parameters for Bias: Min_R={min_val_b:.4f}, Max_R={max_val_b:.4f}, Scale={scale_b:.6f}, ZeroPoint={zero_point_b}")
print(f"Quantized Weights (first 5): {quantized_weights_int8.flatten()[:5].tolist()}")
print(f"Quantized Bias: {quantized_bias_int8.item()}")
print(f"Quantized Weights Range (after clamp): [{quantized_weights_int8.min().item()}, {quantized_weights_int8.max().item()}]")
print(f"Quantized Bias Range (after clamp): [{quantized_bias_int8.min().item()}, {quantized_bias_int8.max().item()}]")


# Store the quantized parameters in a dictionary and save
quantized_params = {
    'weights_int8': quantized_weights_int8.numpy(),
    'bias_int8': quantized_bias_int8.numpy(),
    'scale_w': scale_w,
    'zero_point_w': zero_point_w,
    'scale_b': scale_b,
    'zero_point_b': zero_point_b,
    # Store original min/max for debugging, though not strictly needed for de-quantization with this scheme
    'min_val_w': min_val_w,
    'max_val_w': max_val_w,
    'min_val_b': min_val_b,
    'max_val_b': max_val_b
}
joblib.dump(quantized_params, QUANTIZED_PARAMS_PATH)


# --- Perform inference with de-quantized weights ---
print("\n--- Performing Inference with De-quantized Weights ---")

# De-quantize the weights and bias: R = (Q - Z) * S
dequantized_weights = (quantized_weights_int8.float() - zero_point_w) * scale_w
dequantized_bias = (quantized_bias_int8.float() - zero_point_b) * scale_b

print(f"De-quantized Weights (first 5): {dequantized_weights.flatten()[:5].tolist()}")
print(f"De-quantized Bias: {dequantized_bias.item()}")

# For accurate R^2 calculation, load the test data
california_housing = fetch_california_housing(as_frame=True)
X = california_housing.data
y = california_housing.target
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert X_test to a PyTorch tensor
X_test_tensor = torch.tensor(X_test.values).float()

# Perform prediction using de-quantized weights (manual linear layer)
# y_pred = X @ W_T + b
y_pred_quantized = torch.matmul(X_test_tensor, dequantized_weights.T) + dequantized_bias
y_pred_quantized_np = y_pred_quantized.squeeze().numpy()

# Calculate R-squared for quantized model
r2_quantized = r2_score(y_test, y_pred_quantized_np)
print(f"R-squared for Quantized Model: {r2_quantized:.4f}")

# Calculate R-squared for original scikit-learn model for comparison
y_pred_sklearn = sklearn_model.predict(X_test)
r2_sklearn = r2_score(y_test, y_pred_sklearn)
print(f"R-squared for Original Sklearn Model: {r2_sklearn:.4f}")

# --- Calculate Model Sizes ---
unquant_size_kb = os.path.getsize(UNQUANTIZED_PARAMS_PATH) / 1024
quant_size_kb = os.path.getsize(QUANTIZED_PARAMS_PATH) / 1024

print(f"\nModel Size (unquant_params.joblib): {unquant_size_kb:.2f} KB")
print(f"Model Size (quant_params.joblib): {quant_size_kb:.2f} KB")

print("\n--- Comparison Table ---")
print(f"{'Metric':<20}{'Original Sklearn Model':<25}{'Quantized Model':<20}")
print("-" * 65)
print(f"{'R^2 Score':<20}{r2_sklearn:<25.4f}{r2_quantized:<20.4f}")
print(f"{'Model Size (KB)':<20}{unquant_size_kb:<25.2f}{quant_size_kb:<20.2f}")

print("\nQuantization process complete.")
