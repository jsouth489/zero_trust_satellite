#### Candle Fallback Scripts (Optional)

If PyTorch installation fails, use Candle for attack detection:

** `scripts / train_distilbert_attack_candle.py` **

```python
import candle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset


class PacketDataset:
    def __init__(self, data):
        self.data = data

    def __len__(self): return len(self.data)

    def __getitem__(self, idx): return {"text": str(self.data.iloc[idx]["sequence"]),
                                        "label": int(self.data.iloc[idx]["label"])}


def train():
    df = pd.read_csv("../data/packet_training.csv")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    tokenizer = candle.transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = candle.transformers.AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                                                   num_labels=2)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    train_dataset.set_format("numpy", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("numpy", columns=["input_ids", "attention_mask", "label"])

    training_args = {
        "output_dir": "../models/distilbert_attack_model_candle",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "learning_rate": 2e-5,
        "use_cuda": True,
    }

    trainer = candle.Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset)
    trainer.train()

    model.save_pretrained("../models/distilbert_attack_model_candle")
    tokenizer.save_pretrained("../models/distilbert_attack_model_candle")


if __name__ == "__main__":
    train()