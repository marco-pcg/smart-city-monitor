"""Inference for the water-leak autoencoder (Module 05)."""
import json
from pathlib import Path

import torch
import torch.nn as nn

MODEL_DIR = Path("models")


class LeakAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 6), nn.ReLU(),
            nn.Linear(6, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 6), nn.ReLU(),
            nn.Linear(6, 10),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


_model = None
_device = None
_threshold = None
_num_sensors = None


def _load():
    global _model, _device, _threshold, _num_sensors
    if _model is not None:
        return

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(MODEL_DIR / "anomaly_threshold.json") as f:
        thresh_cfg = json.load(f)
    with open(MODEL_DIR / "sensor_config.json") as f:
        sensor_cfg = json.load(f)

    _threshold = thresh_cfg["threshold"]
    _num_sensors = sensor_cfg["num_sensors"]

    _model = LeakAutoencoder().to(_device)
    _model.load_state_dict(
        torch.load(MODEL_DIR / "leak_autoencoder.pth", map_location=_device)
    )
    _model.eval()


@torch.no_grad()
def detect(sensor_readings: list[float]) -> dict:
    """Detect a leak from N pressure sensor readings."""
    _load()

    if len(sensor_readings) != _num_sensors:
        raise ValueError(f"Expected {_num_sensors} readings, got {len(sensor_readings)}")

    x = torch.tensor(sensor_readings, dtype=torch.float32).unsqueeze(0).to(_device)
    recon = _model(x)
    score = ((recon - x) ** 2).mean().item()

    if score > 2 * _threshold:
        severity = "high"
    elif score > _threshold:
        severity = "moderate"
    else:
        severity = "normal"

    return {
        "anomaly_score": score,
        "threshold": _threshold,
        "is_leak": score > _threshold,
        "severity": severity,
    }


if __name__ == "__main__":
    print("Normal: ", detect([1.0] * 10))
    print("Leak:   ", detect([5.0, 5.0, 0.1, 0.1, 5.0, 5.0, 0.1, 0.1, 5.0, 5.0]))
