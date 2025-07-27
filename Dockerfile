FROM python:3.9-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .
COPY predict.py .
COPY linear_regression_model.joblib .

CMD ["python", "predict.py"]