from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torch.cuda.amp import autocast
import numpy as np


class PacketTestDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        seq = f"{self.data.iloc[idx]['sequence']} [TPM:{self.data.iloc[idx]['tpm_command']}]"
        encoding = self.tokenizer(seq, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        return {k: v.squeeze(0) for k, v in encoding.items()}


def run_attack_detection():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DistilBertForSequenceClassification.from_pretrained("../models/distilbert_attack_model").to(device)
    tokenizer = DistilBertTokenizer.from_pretrained("../models/distilbert_attack_model")
    model.eval()

    df = pd.read_csv("../data/packet_sequences.csv")
    dataset = PacketTestDataset(df, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=16)

    results = []
    with torch.no_grad():
        with autocast():
            for inputs in dataloader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs).logits
                probs = torch.softmax(outputs, dim=1)
                results.extend(probs.cpu().numpy())

    results_df = pd.DataFrame(results, columns=["benign_prob", "malicious_prob"])
    results_df.to_csv("../results/attack_results.csv", index=False)


if __name__ == "__main__":
    run_attack_detection()