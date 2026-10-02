# 01_cnn_traffic.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# ---- 1. Model (dynamic flatten) ----
class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        with torch.no_grad():
            n_features = self.conv_layers(torch.zeros(1, 3, 64, 64)).numel()
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))

# ---- 2. Data ----
transform = transforms.Compose([
    transforms.Resize((64, 64)),   # <-- force every image to 64x64
    transforms.ToTensor()
])
dataset = ImageFolder(root='data/images', transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ---- 3. Sanity check ----
images, labels = next(iter(loader))
print(f"Batch shape: {images.shape}")   # should be [B, 3, 64, 64]

# ---- 4. Train ----
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TrafficSignCNN(num_classes=len(dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), 'models/01/traffic_sign_cnn.pth')

import json
with open("models/01/class_names.json", "w") as f:
    json.dump(dataset.classes, f)

print("CNN trained. Ready to classify traffic signs.")
