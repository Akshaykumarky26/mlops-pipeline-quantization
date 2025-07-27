# MLOps Assignment 3 - Model Quantization Pipeline

## Overview

This project demonstrates a basic MLOps pipeline for training, saving, quantizing, and evaluating a machine learning model using:

- **Scikit-learn** for model training
- **Joblib** for model serialization
- **NumPy** for manual quantization
- **GitHub Actions + Docker** for CI/CD

---

## Files

- `train.py`: Trains a linear regression model on the California Housing dataset.
- `predict.py`: Loads the model and performs a prediction on dummy data.
- `quantize.py`: Manually quantizes model weights, dequantizes them, and evaluates R² score.

---

## Quantization Logic

- Only the model **weights (coefficients)** were quantized to `int16`.
- The **bias term (intercept)** was retained in full precision to reduce error.
- A min-max scaling approach was used for manual quantization.

---

## Results

| Metric             | Original Sklearn | Quantized Model |
|--------------------|------------------|-----------------|
| R² Score           | 0.5758           | -0.1799         |
| Model Size (KB)    | 0.40             | 0.42            |

---

## Observations

- The quantized model experienced a **sharp drop in performance**, with a negative R².
- This is due to:
  - The very **small range of coefficient values** in linear regression.
  - **Manual quantization** not preserving decimal precision well enough for this model.
- In real-world scenarios, quantization-aware training or using models like **PyTorch + dynamic quantization** is recommended.

---

## Conclusion

This assignment demonstrates the challenges of quantizing regression models and the trade-offs between model size and accuracy. Despite the loss in R², the pipeline illustrates an end-to-end approach from training to Docker deployment.