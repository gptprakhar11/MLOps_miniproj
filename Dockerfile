FROM python:3.12-slim

WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, trained models, and configs
COPY app/ app/
COPY models/ models/
COPY src/ src/
COPY params.yaml .

# Expose prediction service port
EXPOSE 8000

# Start uvicorn server as required by handbook section 9
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
