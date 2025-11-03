import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from model import LandmarkModel

CSV_PATH = "normalized_landmarks.csv"
BATCH_SIZE = 256
EPOCHS = 30
LR = 0.001

class ASLDataset(Dataset):
    def __init__(self, csv_path):
        data = pd.read_csv(csv_path)
        self.labels = LabelEncoder().fit_transform(data["label"])
        self.features = data.drop("label", axis=1).values.astype("float32")
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.features = torch.tensor(self.features, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

dataset = ASLDataset(CSV_PATH)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LandmarkModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    total_loss = 0
    model.train()
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "../model/asl_landmarks_model.pth")
print("Model saved as asl_landmarks_model.pth")
