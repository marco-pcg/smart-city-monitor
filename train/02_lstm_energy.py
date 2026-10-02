"""Module 02 — LSTM energy forecasting (corrected)."""
import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


# ---- Model definition ----
class EnergyLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ---- Data ----
df = pd.read_csv("data/PJMW_hourly.csv", nrows=10000)
df = df.sort_values("Datetime").reset_index(drop=True)
raw = df["PJMW_MW"].values.astype(np.float32).reshape(-1, 1)

# Split first, then fit scaler on train only (avoids data leakage)
split = int(len(raw) * 0.8)
train_raw, test_raw = raw[:split], raw[split:]

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_raw).flatten()
test_scaled = scaler.transform(test_raw).flatten()


# ---- Sequence creation ----
def create_sequences(data, seq_length=30):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i + seq_length])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)


SEQ_LEN = 30
X_train, y_train = create_sequences(train_scaled, SEQ_LEN)
X_test, y_test = create_sequences(test_scaled, SEQ_LEN)

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)   # (N, 30, 1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)   # (N, 1)
X_test  = torch.tensor(X_test,  dtype=torch.float32).unsqueeze(-1)
y_test  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(-1)


# ---- Training ----
HIDDEN = 50
LAYERS = 2

model = EnergyLSTM(input_size=1, hidden_size=HIDDEN, num_layers=LAYERS)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred, y_test).item()
        print(f"Epoch {epoch:2d} | train MSE: {loss.item():.6f} | test MSE: {test_loss:.6f}")

torch.save(model.state_dict(), "models/02/energy_lstm.pth")
joblib.dump(scaler, "models/02/energy_scaler.pkl")

with open("models/02/energy_metadata.json", "w") as f:
    json.dump({
        "input_size": 1,
        "hidden_size": HIDDEN,
        "num_layers": LAYERS,
        "seq_length": SEQ_LEN,
        "target_column": "PJMW_MW",
        "scaler_type": "MinMaxScaler",
    }, f, indent=2)
