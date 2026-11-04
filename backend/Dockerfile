FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Trained weights are baked in only if you copy backend/models into the build
# context; otherwise the service runs with the configured gateway fallback.
ENV ML_MODEL_DIR=/app/models

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
