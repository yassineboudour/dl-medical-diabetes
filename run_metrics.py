"""Execute core training for all 3 parts and save measured metrics to metrics.json."""
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent
METRICS_PATH = ROOT / "metrics.json"


def run_mlp():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Outcome",
    ]
    df = pd.read_csv(url, names=cols)
    for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df[c] = df[c].replace(0, np.nan)
        df[c] = df[c].fillna(df[c].median())

    X = df.drop("Outcome", axis=1).values.astype(np.float32)
    y = df["Outcome"].values.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(8, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)

    def loaders(Xa, ya, shuffle):
        ds = TensorDataset(torch.tensor(Xa), torch.tensor(ya).unsqueeze(1))
        return DataLoader(ds, batch_size=32, shuffle=shuffle)

    train_loader = loaders(X_train, y_train, True)
    val_loader = loaders(X_val, y_val, False)
    test_loader = loaders(X_test, y_test, False)

    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCELoss()
    best_val = float("inf")
    best_state = None
    patience, wait = 10, 0

    for _ in range(80):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        vloss, preds, labels = 0.0, [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                vloss += crit(out, yb).item() * len(xb)
                preds.extend(out.cpu().numpy().ravel())
                labels.extend(yb.cpu().numpy().ravel())
        vloss /= len(val_loader.dataset)
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb.to(device)).cpu().numpy().ravel()
            probs.extend(out)
            labels.extend(yb.numpy().ravel())

    probs = np.array(probs)
    labels = np.array(labels)
    pred = (probs >= 0.5).astype(int)
    return {
        "test_accuracy": float(accuracy_score(labels, pred)),
        "test_auc_roc": float(roc_auc_score(labels, probs)),
        "test_f1_macro": float(f1_score(labels, pred, average="macro")),
        "test_precision": float(precision_score(labels, pred, zero_division=0)),
        "test_recall": float(recall_score(labels, pred, zero_division=0)),
    }


def run_cnn():
    from medmnist import PneumoniaMNIST
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()
    train_data = PneumoniaMNIST(split="train", transform=transform, download=True)
    val_data = PneumoniaMNIST(split="val", transform=transform, download=True)
    test_data = PneumoniaMNIST(split="test", transform=transform, download=True)

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, 1), nn.Sigmoid())

        def forward(self, x):
            return self.classifier(self.features(x))

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=64, shuffle=shuffle, num_workers=0)

    train_loader = loader(train_data, True)
    val_loader = loader(val_data, False)
    test_loader = loader(test_data, False)

    model = CNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCELoss()
    best_val, best_state, wait, patience = float("inf"), None, 0, 7

    for _ in range(25):
        model.train()
        for img, lbl in train_loader:
            img = img.to(device)
            lbl = lbl.float().to(device).view(-1, 1)
            opt.zero_grad()
            crit(model(img), lbl).backward()
            opt.step()
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for img, lbl in val_loader:
                img = img.to(device)
                lbl = lbl.float().to(device).view(-1, 1)
                vloss += crit(model(img), lbl).item() * len(lbl)
        vloss /= len(val_loader.dataset)
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for img, lbl in test_loader:
            out = model(img.to(device)).cpu().numpy().ravel()
            probs.extend(out)
            labels.extend(lbl.numpy().ravel())

    probs = np.array(probs)
    labels = np.array(labels)
    pred = (probs >= 0.5).astype(int)
    return {
        "test_accuracy": float(accuracy_score(labels, pred)),
        "test_auc_roc": float(roc_auc_score(labels, probs)),
        "test_f1_macro": float(f1_score(labels, pred, average="macro")),
    }


def run_rnn():
    from collections import Counter

    url_train = "https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_train.csv"
    url_test = "https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_test.csv"
    df_train = pd.read_csv(url_train)
    df_test = pd.read_csv(url_test)

    def tokenize(text):
        return str(text).lower().split()

    all_tokens = []
    for t in df_train["medical_abstract"]:
        all_tokens.extend(tokenize(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, _ in Counter(all_tokens).most_common(8000):
        if w not in vocab:
            vocab[w] = len(vocab)

    def encode(text, max_len=128):
        ids = [vocab.get(w, 1) for w in tokenize(text)][:max_len]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids

    X_train = np.array([encode(t) for t in df_train["medical_abstract"]], dtype=np.int64)
    y_train = (df_train["condition_label"].values - 1).astype(np.int64)
    X_test = np.array([encode(t) for t in df_test["medical_abstract"]], dtype=np.int64)
    y_test = (df_test["condition_label"].values - 1).astype(np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    V, C, H = len(vocab), 5, 128

    class RecCls(nn.Module):
        def __init__(self, rnn_type):
            super().__init__()
            self.emb = nn.Embedding(V, 128, padding_idx=0)
            rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_type]
            self.rnn = rnn_cls(128, H, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(H * 2, C)

        def forward(self, x):
            out, _ = self.rnn(self.emb(x))
            mask = (x != 0).unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            return self.fc(pooled)

    results = {}
    for name in ["RNN", "LSTM", "GRU"]:
        t0 = time.time()
        model = RecCls(name).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        loader = DataLoader(ds, batch_size=64, shuffle=True)
        for _ in range(8):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                crit(model(xb), yb).backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_test).to(device))
            pred = logits.argmax(1).cpu().numpy()
        results[name] = {
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "test_f1_macro": float(f1_score(y_test, pred, average="macro")),
            "train_time_sec": round(time.time() - t0, 1),
        }

    best = max(results, key=lambda k: results[k]["test_f1_macro"])
    return {"models": results, "best_model": best}


if __name__ == "__main__":
    metrics = {"mlp": {}, "cnn": {}, "rnn": {}}
    print("Training MLP...")
    metrics["mlp"] = run_mlp()
    print("MLP:", metrics["mlp"])
    print("Training CNN...")
    metrics["cnn"] = run_cnn()
    print("CNN:", metrics["cnn"])
    print("Training RNN variants...")
    metrics["rnn"] = run_rnn()
    print("RNN:", metrics["rnn"])
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Saved", METRICS_PATH)
