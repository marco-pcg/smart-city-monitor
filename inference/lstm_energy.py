"""Inference for the energy demand LSTM forecaster (Module 02)."""
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

MODEL_DIR = Path("train/models")

from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="marco-pcg/Neural-Network-Types-models",
    filename="energy_lstm.pth"
)
scaler_path = hf_hub_download(
    repo_id="marco-pcg/Neural-Network-Types-models",
    filename="energy_scaler.pkl"
)
meta_path = hf_hub_download(
    repo_id="marco-pcg/Neural-Network-Types-models",
    filename="energy_metadata.json"
)

# ---- Architecture (must match training) ----
class EnergyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# ---- Lazy-loaded singleton ----
_model = None
_scaler = None
_meta = None
_device = None


def _load():
    global _model, _scaler, _meta, _device
    if _model is not None:
        return

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(meta_path) as f:
        _meta = json.load(f)

    _scaler = joblib.load(scaler_path)

    _model = EnergyLSTM(
        input_size=_meta["input_size"],
        hidden_size=_meta["hidden_size"],
        num_layers=_meta["num_layers"],
    ).to(_device)
    _model.load_state_dict(
        torch.load(model_path, map_location=_device)
    )
    _model.eval()


@torch.no_grad()
def forecast(last_n_days: list[float]) -> dict:
    """
    Predict next-day kWh consumption.
    `last_n_days` must have exactly `seq_length` values (see metadata).
    """
    _load()

    seq_len = _meta["seq_length"]
    if len(last_n_days) != seq_len:
        raise ValueError(f"Expected {seq_len} values, got {len(last_n_days)}")

    arr = np.array(last_n_days, dtype=np.float32).reshape(-1, 1)
    arr_scaled = _scaler.transform(arr).flatten()
    tensor = torch.tensor(arr_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(_device)

    pred_scaled = _model(tensor).cpu().numpy()
    pred_kwh = _scaler.inverse_transform(pred_scaled).flatten()[0]

    return {
        "forecast_kwh": float(pred_kwh),
        "input_length": seq_len,
        "input_range": [float(min(last_n_days)), float(max(last_n_days))],
    }


if __name__ == "__main__":
    # Replace with real 30-day history
    sample = [1200 + 20 * (i % 7) for i in range(30)]
    print(forecast(sample))
