# predict.py
import joblib
from sklearn.datasets import fetch_california_housing
import numpy as np
import os

print("Starting model prediction verification...")

# Define the path where the model is expected to be saved
model_path = 'linear_regression_model.joblib'

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found. Ensure train.py was run successfully.")
    exit(1) # Exit with an error code

try:
    # Load the saved model
    model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")

    # Fetch a small part of the California Housing dataset for prediction
    # To ensure it doesn't fail if the full dataset isn't available or if it's too large for a quick check.
    # We need to ensure X_test is available or create a dummy input.
    # For simplicity, let's create a dummy input with the correct number of features (8 for California Housing).
    # In a real scenario, you'd load test data or receive input.
    dummy_input = np.array([[0.5, 35.0, 5.0, 1.0, 500.0, 2.0, 35.0, 1200.0]]) # Example dummy input
    # Note: The actual feature range matters for meaningful predictions. This is just for verification.
    # The California Housing dataset has 8 features.

    # Ensure the dummy input has the correct shape (n_samples, n_features)
    if dummy_input.shape[1] != 8: # California Housing has 8 features
        raise ValueError(f"Dummy input has {dummy_input.shape[1]} features, but expected 8.")

    # Perform a prediction
    prediction = model.predict(dummy_input)
    print(f"Prediction for dummy input: {prediction[0]:.2f}")
    print("Prediction verification successful!")

except Exception as e:
    print(f"An error occurred during prediction verification: {e}")
    exit(1) # Exit with an error code