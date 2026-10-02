"""Inference for the traffic sign CNN classifier (Module 01)."""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

MODEL_DIR = Path("models")

# ---- Architecture (must match training) ----
class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        with torch.no_grad():
            n_features = self.conv_layers(torch.zeros(1, 3, 64, 64)).numel()
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))


# ---- Lazy-loaded singleton ----
_model = None
_class_names = None
_device = None
_transform = None


def _load():
    global _model, _class_names, _device, _transform
    if _model is not None:
        return

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(MODEL_DIR / "class_names.json") as f:
        _class_names = json.load(f)

    _model = TrafficSignCNN(num_classes=len(_class_names)).to(_device)
    _model.load_state_dict(
        torch.load(MODEL_DIR / "traffic_sign_cnn.pth", map_location=_device)
    )
    _model.eval()

    _transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])


@torch.no_grad()
def predict(image_path: str, top_k: int = 3) -> dict:
    """Classify a traffic sign image. Returns labels + confidences."""
    _load()

    img = Image.open(image_path).convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(_device)

    probs = torch.softmax(_model(tensor), dim=1)[0]
    confs, idxs = probs.topk(min(top_k, len(_class_names)))

    return {
        "prediction": _class_names[idxs[0].item()],
        "confidence": float(confs[0].item()),
        "top_k": [
            {"label": _class_names[i.item()], "confidence": float(c.item())}
            for c, i in zip(confs, idxs)
        ],
    }


if __name__ == "__main__":
    import sys
    result = predict(sys.argv[1])
    print(f"Prediction: {result['prediction']} ({result['confidence']:.2%})")
    for item in result["top_k"]:
        print(f"  {item['label']:20s} {item['confidence']:.2%}")
