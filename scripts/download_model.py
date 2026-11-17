"""Download the latest version of a registered model from DagsHub (MLflow Model Registry).

Required environment variables:
    MLFLOW_TRACKING_URI       e.g. https://dagshub.com/<user>/<repo>.mlflow
    MLFLOW_TRACKING_USERNAME  DagsHub username
    MLFLOW_TRACKING_PASSWORD  DagsHub access token
Optional:
    MODEL_NAME  registered model name (default: digits-random-forest)
    MODEL_DIR   destination folder     (default: model)
"""
import os
import shutil
import sys
import tempfile

import mlflow
from mlflow import MlflowClient

MODEL_NAME = os.getenv("MODEL_NAME", "digits-random-forest")
MODEL_DIR = os.getenv("MODEL_DIR", "model")


def main() -> None:
    missing = [v for v in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")
               if not os.getenv(v)]
    if missing:
        sys.exit(f"Missing environment variables: {', '.join(missing)}")

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        sys.exit(f"No registered versions found for model '{MODEL_NAME}'")

    latest = max(versions, key=lambda v: int(v.version))
    print(f"Latest version of '{MODEL_NAME}': v{latest.version} (run {latest.run_id})")

    with tempfile.TemporaryDirectory() as tmp:
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{MODEL_NAME}/{latest.version}", dst_path=tmp
        )
        # Find the folder that contains the MLmodel file, wherever MLflow put it
        model_root = next((root for root, _, files in os.walk(tmp) if "MLmodel" in files), None)
        if model_root is None:
            sys.exit("Downloaded artifacts do not contain an MLmodel file")
        shutil.rmtree(MODEL_DIR, ignore_errors=True)
        shutil.copytree(model_root, MODEL_DIR)

    with open(os.path.join(MODEL_DIR, "registry_version.txt"), "w") as f:
        f.write(str(latest.version))
    print(f"Model saved to {os.path.abspath(MODEL_DIR)}")


if __name__ == "__main__":
    main()
