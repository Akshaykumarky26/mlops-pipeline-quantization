import joblib
from sklearn.datasets import fetch_california_housing
import numpy as np
import os

print("Starting model prediction verification...")

model_path = 'linear_regression_model.joblib'


if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found. Ensure train.py was run successfully.")
    exit(1)

try:

    model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")

    dummy_input = np.array([[0.5, 35.0, 5.0, 1.0, 500.0, 2.0, 35.0, 1200.0]]) # Example dummy input


    if dummy_input.shape[1] != 8: # California Housing has 8 features
        raise ValueError(f"Dummy input has {dummy_input.shape[1]} features, but expected 8.")

    # Perform a prediction
    prediction = model.predict(dummy_input)
    print(f"Prediction for dummy input: {prediction[0]:.2f}")
    print("Prediction verification successful!")

except Exception as e:
    print(f"An error occurred during prediction verification: {e}")
    exit(1) # Exit with an error code
    