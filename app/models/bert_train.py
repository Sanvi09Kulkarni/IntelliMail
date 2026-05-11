import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from datasets import Dataset


# =========================
# LOAD DATA
# =========================

df = pd.read_csv("data/emails.csv")

df = df.dropna()
df = df.drop_duplicates()

# labels
labels = df["label"].unique().tolist()

label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

df["label_id"] = df["label"].map(label2id)


# =========================
# TRAIN TEST SPLIT
# =========================

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(),
    df["label_id"].tolist(),
    test_size=0.2,
    random_state=42
)


# =========================
# TOKENIZER
# =========================

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True
)

test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True
)


# =========================
# DATASET CLASS
# =========================

class EmailDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }

        item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.labels)


train_dataset = EmailDataset(train_encodings, train_labels)
test_dataset = EmailDataset(test_encodings, test_labels)


# =========================
# MODEL
# =========================

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(labels)
)


# =========================
# TRAINING CONFIG
# =========================

training_args = TrainingArguments(
    output_dir="./results",

    num_train_epochs=3,

    per_device_train_batch_size=4,

    logging_dir="./logs",

    logging_steps=10
)

# =========================
# TRAINER
# =========================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)


# =========================
# TRAIN
# =========================

trainer.train()


# =========================
# SAVE MODEL
# =========================

model.save_pretrained("app/models/bert_model")

tokenizer.save_pretrained("app/models/bert_model")


# save label mappings
torch.save(label2id, "app/models/bert_model/label2id.pt")
torch.save(id2label, "app/models/bert_model/id2label.pt")


print("✅ DistilBERT model trained and saved!")