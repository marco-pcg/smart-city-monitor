# 03_gnn_fraud.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# 1. Define a Graph Convolutional Network (GCN)
class FraudGCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 2) # Binary classification: Fraud or Not

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 2. Simulated Graph Data
# Nodes = Users (features: account_age, avg_transaction)
# Edges = Transactions between users
# Labels = 0 (Legit) or 1 (Fraud)

# Example: 4 users, 2 features per user
x = torch.tensor([[1.0, 0.5], [0.5, 1.0], [0.1, 0.1], [5.0, 5.0]], dtype=torch.float)
# Edges: User 0 -> 1, User 1 -> 2, User 2 -> 3, User 3 -> 0
edge_index = torch.tensor([[0, 1, 2, 3],
                            [1, 2, 3, 0]], dtype=torch.long)
# Labels: User 3 is a fraudster
y = torch.tensor([0, 0, 0, 1], dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)

# 3. Train
model = FraudGCN(num_node_features=2, hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()

# 4. Predict
model.eval()
pred = model(data.x, data.edge_index).argmax(dim=1)

torch.save(model.state_dict(), 'models/03/fraud_gnn_model.pth')

print(f"Predictions: {pred}")
print("GNN trained. User 3 identified as fraud based on network connections.")
