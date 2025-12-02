from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
import numpy as np

# ---------- Request & response schemas ----------

class PredictRequest(BaseModel):
    # 30 numeric features from the breast cancer dataset
    features: list[float]


class PredictResponse(BaseModel):
    prediction: int          # 0 or 1
    predicted_label: str     # "malignant" or "benign"
    model_name: str          # e.g. "GradientBoostingClassifier"


# ---------- Load model at startup ----------

app = FastAPI(
    title="Breast Cancer Classifier API",
    description="FastAPI service for a scikit-learn model tracked with MLflow.",
    version="1.0.0",
)

MODEL_PATH = Path(__file__).with_name("model.pkl")
model = joblib.load(MODEL_PATH)

# sklearn's breast cancer dataset: 0 = malignant, 1 = benign
LABEL_MAPPING = {0: "malignant", 1: "benign"}


@app.get("/")
@app.get("/")
def read_root():
    return {
        "message": "Breast Cancer Classifier API is running.",
        "predict_endpoint": "/predict"
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Take a list of 30 features and return the model prediction."""
    features_array = np.array(request.features).reshape(1, -1)
    pred = int(model.predict(features_array)[0])
    label = LABEL_MAPPING.get(pred, "unknown")

    return PredictResponse(
        prediction=pred,
        predicted_label=label, 
        model_name=type(model).__name__,
    )