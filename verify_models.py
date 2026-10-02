"""Verify all saved model artifacts load correctly."""
import json
from pathlib import Path

import joblib
import torch

MODEL_DIR = Path("models")

REQUIRED = {
    "cnn":         ["traffic_sign_cnn.pth", "class_names.json"],
    "lstm":        ["energy_lstm.pth", "energy_scaler.pkl", "energy_metadata.json"],
    "gnn":         ["fraud_gcn.pth", "node_features.json"],
    "transformer": ["report_classifier.pkl", "label_map.json", "encoder_config.json"],
    "autoencoder": ["leak_autoencoder.pth", "anomaly_threshold.json", "sensor_config.json"],
}


def check_pth(path):
    torch.load(path, map_location="cpu")


def check_pkl(path):
    joblib.load(path)


def check_json(path):
    with open(path) as f:
        json.load(f)


def main():
    print("=== Smart City Monitor — Model Verification ===\n")
    all_ok = True

    for module, files in REQUIRED.items():
        print(f"[{module}]")
        for fname in files:
            path = MODEL_DIR / fname
            if not path.exists():
                print(f"  ❌ MISSING  {fname}")
                all_ok = False
                continue

            try:
                if fname.endswith(".pth"):
                    check_pth(path)
                elif fname.endswith(".pkl"):
                    check_pkl(path)
                elif fname.endswith(".json"):
                    check_json(path)
                size_kb = path.stat().st_size / 1024
                print(f"  ✅ {fname:35s} {size_kb:6.1f} KB")
            except Exception as e:
                print(f"  ❌ {fname:35s} — {e}")
                all_ok = False
        print()

    if all_ok:
        print("All artifacts verified. Ready to run `python predict.py <command>`.")
    else:
        print("Some artifacts are missing or broken. Re-run the training scripts.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
