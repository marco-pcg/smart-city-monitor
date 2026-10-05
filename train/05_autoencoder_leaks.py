import torch
import torch.nn as nn
import numpy as np

class LeakAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: Compress 10 sensor readings into 3 numbers
        self.encoder = nn.Sequential(
            nn.Linear(10, 6),
            nn.ReLU(),
            nn.Linear(6, 3)
        )
        # Decoder: Reconstruct 10 readings from 3 numbers
        self.decoder = nn.Sequential(
            nn.Linear(3, 6),
            nn.ReLU(),
            nn.Linear(6, 10)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

normal_data = torch.randn(1000, 10) * 0.5 + 1.0  # Mean 1.0, std 0.5

model = LeakAutoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(normal_data)
    loss = criterion(output, normal_data)
    loss.backward()
    optimizer.step()

leak_data = torch.tensor([[5.0, 5.0, 0.1, 0.1, 5.0, 5.0, 0.1, 0.1, 5.0, 5.0]]) # Unusual pattern / Anomaly
reconstruction = model(leak_data)
anomaly_score = criterion(reconstruction, leak_data)
print(f"Reconstruction Error (Anomaly Score): {anomaly_score.item():.4f}")
print("If this score is high, it indicates a leak.")

import json, torch
import numpy as np

torch.save(model.state_dict(), "models/05/leak_autoencoder.pth")

model.eval()
with torch.no_grad():
    train_recon = model(normal_data)
    errors = ((train_recon - normal_data) ** 2).mean(dim=1).numpy()

threshold = float(errors.mean() + 3 * errors.std())
print(f"Anomaly threshold (mean + 3σ): {threshold:.6f}")

with open("models/05/anomaly_threshold.json", "w") as f:
    json.dump({
        "threshold": threshold,
        "method": "mean + 3*std of training reconstruction error",
        "train_mean_error": float(errors.mean()),
        "train_std_error": float(errors.std()),
        "num_sensors": normal_data.shape[1]
    }, f, indent=2)

sensor_config = {
    "num_sensors": int(normal_data.shape[1]),
    "feature_mean": normal_data.mean(dim=0).tolist(),
    "feature_std": normal_data.std(dim=0).tolist()
}
with open("models/05/sensor_config.json", "w") as f:
    json.dump(sensor_config, f, indent=2)
