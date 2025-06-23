from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split


class PacketDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        seq = str(self.data.iloc[idx]["sequence"])
        label = int(self.data.iloc[idx]["label"])
        encoding = self.tokenizer(seq, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        return {k: v.squeeze(0) for k, v in encoding.items()}, label


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    df = pd.read_csv("../data/packet_training.csv")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = PacketDataset(train_df, tokenizer)
    val_dataset = PacketDataset(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=16)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    scaler = GradScaler()

    for epoch in range(3):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = torch.tensor(labels).to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = torch.tensor(labels).to(device)
                with autocast():
                    outputs = model(**inputs, labels=labels)
                    val_loss += outputs.loss.item()
        val_loss /= len(val_loader)
        print(f'Epoch {epoch}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss}')

    model.save_pretrained("../models/distilbert_attack_model")
    tokenizer.save_pretrained("../models/distilbert_attack_model")


if __name__ == "__main__":
    train()