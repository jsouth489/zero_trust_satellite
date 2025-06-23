import candle
import pandas as pd
import numpy as np
from datasets import Dataset


class PacketTestDataset:
    def __init__(self, data):
        self.data = data

    def __len__(self): return len(self.data)

    def __getitem__(self, idx): return {"text": str(self.data.iloc[idx]["sequence"])}


def run_attack_detection():
    df = pd.read_csv("../data/packet_sequences.csv")
    dataset = Dataset.from_pandas(df)

    tokenizer = candle.transformers.AutoTokenizer.from_pretrained("../models/distilbert_attack_model_candle")
    model = candle.transformers.AutoModelForSequenceClassification.from_pretrained(
        "../models/distilbert_attack_model_candle")
    model.cuda().eval()

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format("numpy", columns=["input_ids", "attention_mask"])

    batch_size = 16
    results = []
    with candle.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            inputs = {
                "input_ids": candle.Tensor(batch["input_ids"]).cuda(),
                "attention_mask": candle.Tensor(batch["attention_mask"]).cuda()
            }
            outputs = model(**inputs).logits
            probs = candle.softmax(outputs, dim=1)
            results.extend(probs.cpu().numpy())

    results_df = pd.DataFrame(results, columns=["benign_prob", "malicious_prob"])
    results_df.to_csv("../results/attack_results.csv", index=False)


if __name__ == "__main__":
    run_attack_detection()