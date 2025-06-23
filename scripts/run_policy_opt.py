from stable_baselines3 import PPO
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np


class PolicyTestDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx): return np.array([float(x) for x in self.data.iloc[idx]["state"].split(',')],
                                                dtype=np.float32)


def run_policy_opt():
    model = PPO.load("../models/ppo_policy_model")

    dataset = PolicyTestDataset("../data/policy_configs.csv")
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=16)

    results = []
    for batch in dataloader:
        batch = batch.numpy()
        actions, _ = model.predict(batch, deterministic=True)
        results.extend(actions)

    results_df = pd.DataFrame(results, columns=["action"])
    results_df.to_csv("../results/policy_results.csv", index=False)


if __name__ == "__main__":
    run_policy_opt()