# MP266 Prediction Service Guide & Architecture
**Lead Author:** Aayushman Mathpati (Service Lead, S24CSEU1033)  
**Project:** Cold Storage Spoilage Prediction (Surat Cold Chain Cooperative)  
**Group:** G111 | **Batch:** 11  

---

## 1. Overview & Service Architecture

The prediction service provides an HTTP REST API for cold storage operators near Surat to predict the spoilage percentage (`spoilage_pct`) of agricultural produce (potatoes, onions, and apples) before unloading.

The service is built with **FastAPI**, containerized via **Docker**, and configured for deployment on **Kubernetes**.

```
                +---------------------------------------+
                |           Kubernetes Cluster          |
                |                                       |
Incoming        |   +-------------------------------+   |
HTTP Request    |   |     Service: mp266-svc        |   |
(Port 80) ----> |   |     (Port 80 -> Port 8000)    |   |
                |   +---------------+---------------+   |
                |                   |                   |
                |     +-------------+-------------+     |
                |     |                           |     |
                |     v                           v     |
                | +-------------------+   +-------------------+ |
                | | Pod: mp266-api-1  |   | Pod: mp266-api-2  | |
                | | (Uvicorn / App)   |   | (Uvicorn / App)   | |
                | +-------------------+   +-------------------+ |
                +---------------------------------------+
```

---

## 2. API Endpoints

### 2.1 Health Check (`GET /health`)
Used by Kubernetes readiness and liveness probes.
- **Path:** `/health`
- **Method:** `GET`
- **Success Response:** `200 OK`
```json
{
  "status": "ok"
}
```

### 2.2 Predict Spoilage (`POST /predict`)
Predicts the spoilage percentage for a lot based on chamber environment conditions and crop parameters.
- **Path:** `/predict`
- **Method:** `POST`
- **Content-Type:** `application/json`

#### Request Payload Schema
| Field | Type | Description | Validation Rule |
|---|---|---|---|
| `storage_days` | `float` | Duration the produce has been kept in storage | Required |
| `temp_set_c` | `float` | Temperature setting in degrees Celsius | Must be `<= 20.0` (Client Rule 1) |
| `humidity_pct` | `float` | Relative humidity percentage inside chamber | Required |
| `power_cuts_hours` | `float` | Cumulative duration of electrical power interruptions | Must be `>= 0.0` (Client Rule 2) |
| `bag_weight_kg` | `float` | Average weight per bag in kg | Required |
| `door_openings_per_day`| `float` | Average frequency of chamber door openings per day | Required |
| `crop` | `string` | Stored commodity (`potato`, `onion`, `apple`) | Required |
| `chamber` | `string` | Storage chamber identifier (`Chamber 1`, `Chamber 2`, `Chamber 3`) | Required |

#### Success Response
- **Status:** `200 OK`
```json
{
  "prediction": 8.42
}
```

#### Validation Error Response
- **Status:** `422 Unprocessable Entity` (automatically returned by FastAPI / Pydantic if validation fails):
```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": ["body", "temp_set_c"],
      "msg": "Input should be less than or equal to 20",
      "input": 25.0
    }
  ]
}
```

---

## 3. Client Validation Rules Implementation

Per Section 5 of `PROBLEM.md`, the client has two strict operational constraints enforced directly in the Pydantic schema:

1. **Temperature Upper Bound:**
   > "temp_set_c above 20 is impossible for us. Reject it."
   * Implemented using Pydantic `Field(..., le=20.0)` so invalid requests receive HTTP 422 before reaching model inference.

2. **Non-Negative Power Outage Hours:**
   > "power_cuts_hours below 0 is always a typing mistake. Reject it."
   * Implemented using Pydantic `Field(..., ge=0.0)` to block impossible negative durations.

---

## 4. Structured JSON Logging

Per Section 9 of the project handbook:
* All prediction requests log a structured JSON payload to `stdout`.
* Formatter outputs:
  * `timestamp`: Standard ISO/formatted timestamp.
  * `level`: Log severity (`INFO`).
  * `logger`: Logger name (`mp266_api`).
  * `message`: Action descriptor (`prediction_served`).
  * `prediction`: Feature parameters and the predicted output.
* **Privacy:** Customer and operator identifiers are never collected or logged.

---

## 5. Docker Containerization

The service is packaged using Python 3.12 slim with multi-stage build optimization:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
COPY models/ models/
COPY src/ src/
COPY params.yaml .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Building and Running the Container
```bash
# Build the container image
docker build -t mp266-api:1.0 .

# Run the container locally forwarding port 8000
docker run -p 8000:8000 mp266-api:1.0
```

---

## 6. Verification with cURL & Python

### Testing via cURL
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "storage_days": 45.0,
    "temp_set_c": 4.5,
    "humidity_pct": 88.0,
    "power_cuts_hours": 2.0,
    "bag_weight_kg": 50.0,
    "door_openings_per_day": 6.0,
    "crop": "potato",
    "chamber": "Chamber 1"
  }'
```

### Testing via Python `requests`
```python
import requests

payload = {
    "storage_days": 60.0,
    "temp_set_c": 3.0,
    "humidity_pct": 90.0,
    "power_cuts_hours": 1.5,
    "bag_weight_kg": 48.0,
    "door_openings_per_day": 4.0,
    "crop": "onion",
    "chamber": "Chamber 2"
}

resp = requests.post("http://localhost:8000/predict", json=payload)
print("Prediction:", resp.json())
```
