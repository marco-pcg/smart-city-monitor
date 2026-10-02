"""
Unified FastAPI backend for the Smart City Monitor.

Endpoints:
    GET  /health                    — service status
    POST /traffic                   — CNN traffic sign classification
    POST /energy                    — LSTM energy forecast
    POST /fraud                     — GNN fraud detection
    POST /route                     — Transformer report routing
    POST /leak                      — Autoencoder leak detection
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from inference import (
    cnn_traffic,
    lstm_energy,
    gnn_fraud,
    transformer_reports,
    autoencoder_leaks,
)

app = FastAPI(
    title="Smart City Monitor API",
    description="Five neural network architectures behind one REST service.",
    version="1.0.0",
)


# ---------- Health ----------
@app.get("/health")
def health():
    return {"status": "ok", "models": 5}


# ---------- Module 01: CNN Traffic ----------
@app.post("/traffic")
async def traffic_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")
    try:
        contents = await file.read()
        # cnn_traffic.predict expects a path; write to temp
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        try:
            result = cnn_traffic.predict(tmp_path)
        finally:
            os.unlink(tmp_path)
        return result
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")


# ---------- Module 02: LSTM Energy ----------
class EnergyRequest(BaseModel):
    history: list[float] = Field(..., description="Last N daily kWh values")

@app.post("/energy")
def energy_endpoint(req: EnergyRequest):
    try:
        return lstm_energy.forecast(req.history)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")


# ---------- Module 03: GNN Fraud ----------
class FraudRequest(BaseModel):
    features: list[list[float]] = Field(..., description="Node feature matrix")
    edges: list[list[int]] = Field(..., description="Edge index [2, num_edges]")

@app.post("/fraud")
def fraud_endpoint(req: FraudRequest):
    try:
        return gnn_fraud.predict_nodes(req.features, req.edges)
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")


# ---------- Module 04: Transformer Reports ----------
class RouteRequest(BaseModel):
    text: str | None = None
    batch: list[str] | None = None

@app.post("/route")
def route_endpoint(req: RouteRequest):
    if req.batch:
        return {"results": transformer_reports.route_batch(req.batch)}
    if req.text:
        return transformer_reports.route(req.text)
    raise HTTPException(400, "Provide either 'text' or 'batch'.")


# ---------- Module 05: Autoencoder Leaks ----------
class LeakRequest(BaseModel):
    readings: list[float] = Field(..., description="Sensor pressure values")

@app.post("/leak")
def leak_endpoint(req: LeakRequest):
    try:
        return autoencoder_leaks.detect(req.readings)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")


# ---------- Root ----------
@app.get("/")
def root():
    return {
        "service": "Smart City Monitor",
        "endpoints": ["/traffic", "/energy", "/fraud", "/route", "/leak"],
        "docs": "/docs",
    }
