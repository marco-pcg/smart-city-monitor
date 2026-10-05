"""Inference for the fraud-detection GCN (Module 03)."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

MODEL_DIR = Path("models")


class FraudGCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(self.conv2(x, edge_index), dim=1)


_model = None
_device = None
_meta = None


def _load():
    global _model, _device, _meta
    if _model is not None:
        return

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(MODEL_DIR / "node_features.json") as f:
        _meta = json.load(f)

    _model = FraudGCN(
        num_node_features=_meta["num_node_features"],
        hidden_channels=_meta.get("hidden_channels", 16),
    ).to(_device)
    _model.load_state_dict(
        torch.load(MODEL_DIR / "fraud_gcn.pth", map_location=_device)
    )
    _model.eval()


@torch.no_grad()
def predict_nodes(node_features: list[list[float]],
                  edge_index: list[list[int]]) -> dict:
    """
    Classify each node as legit or fraud.
    - node_features: shape (num_nodes, num_features)
    - edge_index:    shape (2, num_edges), each column is (src, dst)
    """
    _load()

    x = torch.tensor(node_features, dtype=torch.float32).to(_device)
    ei = torch.tensor(edge_index, dtype=torch.long).to(_device)

    log_probs = _model(x, ei)
    probs = torch.exp(log_probs)
    preds = probs.argmax(dim=1)

    results = []
    for i in range(len(preds)):
        label = "fraud" if preds[i].item() == 1 else "legit"
        results.append({
            "node": i,
            "label": label,
            "confidence": float(probs[i, preds[i]].item()),
            "fraud_probability": float(probs[i, 1].item()),
        })

    return {"predictions": results, "num_nodes": len(results)}


if __name__ == "__main__":
    x = [[1.0, 0.5], [0.5, 1.0], [0.1, 0.1], [5.0, 5.0]]
    ei = [[0, 1, 2, 3], [1, 2, 3, 0]]
    print(predict_nodes(x, ei))
