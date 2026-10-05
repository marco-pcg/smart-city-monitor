FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference/ ./inference/
COPY train/models/ ./train/models/
COPY api.py app.py predict.py ./

EXPOSE 8000 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
