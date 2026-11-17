import os
from pathlib import Path

import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))
N_FEATURES = 64  # load_digits: 8x8 images flattened

model = mlflow.sklearn.load_model(str(MODEL_DIR))
version_file = MODEL_DIR / "registry_version.txt"
MODEL_VERSION = version_file.read_text().strip() if version_file.exists() else "unknown"

# Only one endpoint: /predict (auto-generated docs routes are disabled)
app = FastAPI(title="Digits classifier", docs_url=None, redoc_url=None, openapi_url=None)


class PredictRequest(BaseModel):
    inputs: list[list[float]]


@app.post("/predict")
def predict(request: PredictRequest):
    if not request.inputs or any(len(row) != N_FEATURES for row in request.inputs):
        raise HTTPException(status_code=422, detail=f"Each sample must contain {N_FEATURES} features")
    predictions = model.predict(np.array(request.inputs))
    return {"model_version": MODEL_VERSION, "predictions": predictions.tolist()}
