FROM python:3.12-slim

# System deps for Pillow / torchvision
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code + model artifacts
COPY inference/ ./inference/
COPY models/ ./models/
COPY api.py app.py predict.py ./

EXPOSE 8000 7860

# Default: run the API. Override with `docker run ... python app.py` for the demo.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
