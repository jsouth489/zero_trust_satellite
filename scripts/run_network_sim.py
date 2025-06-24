import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torch.cuda.amp import autocast
import numpy as np


class GNNModel(torch.nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(3, 128)
        self.conv2 = GCNConv(128, 64)
        self.fc = torch.nn.Linear(64, 3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.mean(x, dim=0)
        x = self.fc(x)
        return x


class NetworkTestDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        nodes = torch.tensor([float(x) for x in row['nodes'].split(',')], dtype=torch.float).view(-1, 1)
        pqc_map = {'ML-KEM-512': 0, 'ML-KEM-1024': 1, 'ML-DSA-2': 2}
        features = torch.tensor([row['traffic'], row['tpm_attestation'], pqc_map[row['pqc_algorithm']]],
                                dtype=torch.float).view(-1, 3)
        edges = torch.tensor([[int(x.split(',')[0]), int(x.split(',')[1])] for x in row['edges'].split(';') if x],
                             dtype=torch.long).t()
        return Data(x=features.repeat(nodes.shape[0], 1), edge_index=edges)


def run_network_sim():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GNNModel().to(device)
    model.load_state_dict(torch.load("../models/gnn_network_model.pth"))
    model.eval()

    dataset = NetworkTestDataset("../data/network_configs.csv")
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=16)

    results = []
    with torch.no_grad():
        with autocast():
            for batch in dataloader:
                batch = batch.to(device)
                metrics = model(batch)
                results.append(metrics.cpu().numpy())

    results_df = pd.DataFrame(np.concatenate(results), columns=["latency", "throughput", "security"])
    results_df.to_csv("../results/network_results.csv", index=False)


if __name__ == "__main__":
    run_network_sim()