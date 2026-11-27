"""FastAPI prediction service for MP266 Cold Storage Spoilage.

Service Lead: Aayushman Mathpati (S24CSEU1033)
Role: Service Lead (FastAPI App, Request Rules, Structured Logging)

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
from fastapi import FastAPI
from pydantic import BaseModel, Field

try:
    from src.mp266.predict import predict_spoilage
except ImportError:
    import joblib
    import pandas as pd

    _model = None
    _path = Path("models/model.joblib")
    if _path.exists():
        _model = joblib.load(_path)

    def predict_spoilage(data: dict) -> float:
        if _model is None:
            return 0.0
        df = pd.DataFrame([data])
        return float(_model.predict(df)[0])


# JSON structured logging to stdout (never to file)
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "prediction_data"):
            log_entry["prediction"] = record.prediction_data
        return json.dumps(log_entry)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logger = logging.getLogger("mp266_api")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

app = FastAPI(title="MP266 Cold Storage Spoilage API", version="1.0")


class PredictRequest(BaseModel):
    storage_days: float = Field(..., description="Days in storage")
    temp_set_c: float = Field(..., le=20.0, description="Chamber temperature (max 20°C)")
    humidity_pct: float = Field(..., description="Chamber humidity percentage")
    power_cuts_hours: float = Field(..., ge=0.0, description="Power cuts in hours (min 0)")
    bag_weight_kg: float = Field(..., description="Bag weight in kg")
    door_openings_per_day: float = Field(..., description="Average door openings per day")
    crop: str = Field(..., description="Crop stored: potato, onion, or apple")
    chamber: str = Field(..., description="Chamber: Chamber 1, Chamber 2, or Chamber 3")


class PredictResponse(BaseModel):
    prediction: float


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    data = request.model_dump()
    result = predict_spoilage(data)

    record = logger.makeRecord(
        name="mp266_api",
        level=logging.INFO,
        fn="predict",
        lno=0,
        msg="prediction_served",
        args=(),
        exc_info=None,
    )
    record.prediction_data = {**data, "result": result}
    logger.handle(record)

    return PredictResponse(prediction=result)
