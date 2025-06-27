import os
import json
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─── Config ───────────────────────────────────────────────────────────────
JSON_DIR   = "./definition_output"
MODEL_NAME = "klue/bert-base"
BATCH_SIZE = 16
EPOCHS     = 3
LR         = 2e-5
MAX_LEN    = 128
SEED       = 42
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 데이터 로드 & 분할 ───────────────────────────────────────────────────
def load_records(json_dir):
    records = []
    for path in glob(os.path.join(json_dir, "**", "*.json"), recursive=True):
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
        text  = j.get("question", "").strip()
        label = j.get("disease_name", {}).get("kor", "").strip()
        if text and label:
            records.append((text, label))
    return records

def prepare_data(records):
    texts, labels = zip(*records)
    label_set     = sorted(set(labels))
    label2id      = {l: i for i, l in enumerate(label_set)}
    id2label      = {i: l for l, i in label2id.items()}
    y             = [label2id[l] for l in labels]
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, y, test_size=0.2, stratify=y, random_state=SEED
    )
    train = list(zip(X_tr, y_tr))
    test  = list(zip(X_te, y_te))
    return train, test, id2label, len(label_set)

# ─── PyTorch Dataset ─────────────────────────────────────────────────────
class SymptomDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data      = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        toks = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in toks.items()}
        item["labels"] = torch.tensor(label)
        return item

# ─── 학습 & 평가 루프 ────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer):
    model.train()
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        loss  = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def eval_model(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            mask      = batch["attention_mask"].to(DEVICE)
            labels    = batch["labels"].to(DEVICE)
            logits    = model(input_ids, attention_mask=mask).logits
            preds += logits.argmax(dim=-1).tolist()
            trues += labels.tolist()
    return accuracy_score(trues, preds)

# ─── 메인 ────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)

    records     = load_records(JSON_DIR)
    train_data, test_data, id2label, num_labels = prepare_data(records)
    print(f"Loaded {len(records)} samples ({num_labels} labels)")

    tokenizer   = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds    = SymptomDataset(train_data, tokenizer)
    test_ds     = SymptomDataset(test_data, tokenizer)
    train_loader= DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    model      = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    ).to(DEVICE)
    optimizer  = AdamW(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, train_loader, optimizer)
        acc = eval_model(model, test_loader)
        print(f"Epoch {epoch}/{EPOCHS}  Test Accuracy: {acc:.4f}")

    # ── 대화형 추론 모드 ───────────────────────────────────────────────
    model.eval()
    print("\n=== Inference Mode (종료: 빈 입력) ===")
    while True:
        text = input("증상 입력> ").strip()
        if not text:
            print("종료합니다.")
            break
        toks    = tokenizer(
            text, return_tensors="pt",
            truncation=True, padding="max_length",
            max_length=MAX_LEN
        ).to(DEVICE)
        pred_id = model(**toks).logits.argmax(dim=-1).item()
        print("예측된 질병:", id2label[pred_id], "\n")

if __name__ == "__main__":
    main()
