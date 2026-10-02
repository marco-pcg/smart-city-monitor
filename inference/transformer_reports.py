"""Inference for the citizen-report router (Module 04)."""
import json
from pathlib import Path

import joblib
from sentence_transformers import SentenceTransformer

MODEL_DIR = Path("models")


_encoder = None
_classifier = None
_label_map = None
_config = None


def _load():
    global _encoder, _classifier, _label_map, _config
    if _encoder is not None:
        return

    with open(MODEL_DIR / "encoder_config.json") as f:
        _config = json.load(f)

    with open(MODEL_DIR / "label_map.json") as f:
        # keys become strings after JSON round-trip
        _label_map = {int(k): v for k, v in json.load(f).items()}

    _encoder = SentenceTransformer(_config["model_name"])
    _classifier = joblib.load(MODEL_DIR / "report_classifier.pkl")


def route(text: str) -> dict:
    """Route a citizen report to the correct department."""
    _load()

    emb = _encoder.encode([text])
    pred = int(_classifier.predict(emb)[0])
    proba = _classifier.predict_proba(emb)[0]

    return {
        "text": text,
        "department": _label_map[pred],
        "confidence": float(proba[pred]),
        "all_scores": {
            _label_map[i]: float(proba[i]) for i in range(len(proba))
        },
    }


def route_batch(texts: list[str]) -> list[dict]:
    """Batch route multiple reports — much faster than looping."""
    _load()

    embs = _encoder.encode(texts)
    preds = _classifier.predict(embs)
    probas = _classifier.predict_proba(embs)

    results = []
    for text, pred, proba in zip(texts, preds, probas):
        pred = int(pred)
        results.append({
            "text": text,
            "department": _label_map[pred],
            "confidence": float(proba[pred]),
        })
    return results


if __name__ == "__main__":
    print(route("There is a broken pipe leaking water on Elm Street."))
    print(route_batch([
        "Streetlight is out on Oak Lane.",
        "Trash hasn't been picked up in two weeks.",
    ]))
