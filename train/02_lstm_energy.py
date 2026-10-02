import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class EnergyLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(EnergyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

df = pd.read_csv('data/PJMW_hourly.csv')
data = df['PJMW_MW'].values.astype(np.float32)

def create_sequences(data, seq_length = 30):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)

X, y = create_sequences(data)
X = torch.tensor(X).unsqueeze(-1)
y = torch.tensor(y).unsqueeze(-1)

model = EnergyLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
  optimizer.zero_grad()
  output = model(X)
  loss = criterion(output, y)
  loss.backward()
  optimizer.step()

  if epoch % 5 == 0:
    print(f'Epoch {epoch}, MSE Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'models/02/energy_lstm_model.pth')

print("LSTM trained. Ready to forecast energy usage.")
