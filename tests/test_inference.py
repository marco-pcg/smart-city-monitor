"""Smoke tests for all five inference modules."""
import json
from pathlib import Path

import pytest

from inference import (
    cnn_traffic,
    lstm_energy,
    gnn_fraud,
    transformer_reports,
    autoencoder_leaks,
)

MODEL_DIR = Path("models")


def test_all_artifacts_exist():
    required = [
        "traffic_sign_cnn.pth", "class_names.json",
        "energy_lstm.pth", "energy_scaler.pkl", "energy_metadata.json",
        "fraud_gcn.pth", "node_features.json",
        "report_classifier.pkl", "label_map.json", "encoder_config.json",
        "leak_autoencoder.pth", "anomaly_threshold.json", "sensor_config.json",
    ]
    for name in required:
        assert (MODEL_DIR / name).exists(), f"Missing artifact: {name}"


def test_lstm_forecast():
    history = [1200 + 20 * (i % 7) for i in range(30)]
    result = lstm_energy.forecast(history)
    assert "forecast_kwh" in result
    assert isinstance(result["forecast_kwh"], float)


def test_lstm_wrong_length():
    with pytest.raises(ValueError):
        lstm_energy.forecast([1, 2, 3])


def test_gnn_predictions():
    features = [[1.0, 0.5], [0.5, 1.0], [0.1, 0.1], [5.0, 5.0]]
    edges = [[0, 1, 2, 3], [1, 2, 3, 0]]
    result = gnn_fraud.predict_nodes(features, edges)
    assert result["num_nodes"] == 4
    assert all(p["label"] in ("fraud", "legit") for p in result["predictions"])


def test_transformer_routing():
    result = transformer_reports.route("Broken pipe leaking water.")
    assert "department" in result
    assert 0.0 <= result["confidence"] <= 1.0


def test_transformer_batch():
    texts = ["Streetlight out.", "Trash pickup missed."]
    results = transformer_reports.route_batch(texts)
    assert len(results) == 2


def test_leak_normal():
    result = autoencoder_leaks.detect([1.0] * 10)
    assert result["severity"] in ("normal", "moderate", "high")


def test_leak_wrong_length():
    with pytest.raises(ValueError):
        autoencoder_leaks.detect([1.0, 2.0])
