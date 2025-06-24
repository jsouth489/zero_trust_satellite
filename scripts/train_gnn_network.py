import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np


class GNNModel(torch.nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(3, 128)  # Input: traffic, tpm_attestation, pqc_algorithm
        self.conv2 = GCNConv(128, 64)
        self.fc = torch.nn.Linear(64, 3)  # Output: latency, throughput, security

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = torch.mean(x, dim=0)
        x = self.fc(x)
        return x


class NetworkDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        nodes = torch.tensor([float(x) for x in row['nodes'].split(',')], dtype=torch.float).view(-1, 1)
        # Encode pqc_algorithm as a numeric value
        pqc_map = {'ML-KEM-512': 0, 'ML-KEM-1024': 1, 'ML-DSA-2': 2}
        features = torch.tensor([row['traffic'], row['tpm_attestation'], pqc_map[row['pqc_algorithm']]],
                                dtype=torch.float).view(-1, 3)
        edges = torch.tensor([[int(x.split(',')[0]), int(x.split(',')[1])] for x in row['edges'].split(';') if x],
                             dtype=torch.long).t()
        y = torch.tensor([row['latency'], row['throughput'], row['security']], dtype=torch.float)
        return Data(x=features.repeat(nodes.shape[0], 1), edge_index=edges, y=y)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GNNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    dataset = NetworkDataset("../data/network_training.csv")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=16)

    for epoch in range(100):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            with autocast():
                out = model(batch)
                loss = F.mse_loss(out, batch.y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                with autocast():
                    out = model(batch)
                    val_loss += F.mse_loss(out, batch.y).item()
        val_loss /= len(val_loader)
        print(f'Epoch {epoch}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss}')

    torch.save(model.state_dict(), "../models/gnn_network_model.pth")


if __name__ == "__main__":
    train()