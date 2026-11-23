"""FastAPI prediction service for MP266 Cold Storage Spoilage.
Service Lead: Aayushman Mathpati (S24CSEU1033)

Endpoints:
    GET  /health   -> {"status": "ok"}
    POST /predict  -> {"prediction": float}

Client rules (PROBLEM.md section 5):
    - temp_set_c above 20 is impossible -> reject (422)
    - power_cuts_hours below 0 is a typing mistake -> reject (422)
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# --- JSON structured logging (handbook section 9) ---
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_dict = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "prediction_data"):
            log_dict["prediction"] = record.prediction_data
        return json.dumps(log_dict)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logger = logging.getLogger("mp266_api")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

# --- Load model ---
MODEL_PATH = Path("models/model.joblib")
model = None
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)

# --- FastAPI app ---
app = FastAPI(title="MP266 Cold Storage Spoilage API", version="1.0")


class PredictRequest(BaseModel):
    """Request body for POST /predict.

    Exactly the fields listed in PROBLEM.md section 5.
    Client rules enforced via pydantic Field constraints:
      - temp_set_c: must be <= 20 (above 20 is impossible)
      - power_cuts_hours: must be >= 0 (below 0 is a typing mistake)
    """
    storage_days: float = Field(..., description="Days in storage")
    temp_set_c: float = Field(..., le=20.0, description="Chamber temperature setting (max 20)")
    humidity_pct: float = Field(..., description="Chamber humidity percentage")
    power_cuts_hours: float = Field(..., ge=0.0, description="Power cut hours (min 0)")
    bag_weight_kg: float = Field(..., description="Weight per bag in kg")
    door_openings_per_day: float = Field(..., description="Average door openings per day")
    crop: str = Field(..., description="Crop stored: potato, onion, or apple")
    chamber: str = Field(..., description="Storage chamber: Chamber 1, Chamber 2, or Chamber 3")


class PredictResponse(BaseModel):
    prediction: float


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint for readiness and liveness probes."""
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predict spoilage percentage for a cold storage lot."""
    if model is None:
        return PredictResponse(prediction=0.0)

    input_df = pd.DataFrame([req.model_dump()])
    prediction = float(model.predict(input_df)[0])

    # Structured JSON log for every prediction (never log personal details)
    log_record = logger.makeRecord(
        name="mp266_api",
        level=logging.INFO,
        fn="predict",
        lno=0,
        msg="prediction_served",
        args=(),
        exc_info=None,
    )
    log_record.prediction_data = {
        "storage_days": req.storage_days,
        "temp_set_c": req.temp_set_c,
        "humidity_pct": req.humidity_pct,
        "power_cuts_hours": req.power_cuts_hours,
        "bag_weight_kg": req.bag_weight_kg,
        "door_openings_per_day": req.door_openings_per_day,
        "crop": req.crop,
        "chamber": req.chamber,
        "result": prediction,
    }
    logger.handle(log_record)

    return PredictResponse(prediction=prediction)
