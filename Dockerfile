# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/model

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# DagsHub / MLflow settings (the token is passed as a build secret, never stored in the image)
ARG MLFLOW_TRACKING_URI=https://dagshub.com/Malick26/mlflow-with-daghub-Course.mlflow
ARG MLFLOW_TRACKING_USERNAME=Malick26
ARG MODEL_NAME=digits-random-forest

# Fetch the latest registered model from DagsHub through MLflow at build time
COPY scripts/download_model.py scripts/
RUN --mount=type=secret,id=dagshub_token,required=true \
    MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
    MLFLOW_TRACKING_USERNAME="$MLFLOW_TRACKING_USERNAME" \
    MLFLOW_TRACKING_PASSWORD="$(cat /run/secrets/dagshub_token)" \
    MODEL_NAME="$MODEL_NAME" \
    python scripts/download_model.py

COPY app/ app/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
