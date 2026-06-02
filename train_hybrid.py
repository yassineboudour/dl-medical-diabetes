"""
train_hybrid.py — Entraine CNN+LSTM et CNN+Attention directement (sans Jupyter).
Ecrit les resultats dans metrics.json.

Usage : py train_hybrid.py
        py train_hybrid.py --epochs 15
"""
import argparse
import json
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device} | PyTorch : {torch.__version__}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--mock", action="store_true", help="Utiliser un dataset synthetique sans telecharger")
    return p.parse_args()


# ─── Chargement PneumoniaMNIST ────────────────────────────────────────────────
def load_pneumonia(mock=False):
    if mock:
        print("PneumoniaMNIST: mode MOCK (donnees synthetiques)")
        X_tr = np.random.rand(200, 1, 28, 28).astype(np.float32)
        y_tr = np.random.randint(0, 2, 200)
        X_te = np.random.rand(50, 1, 28, 28).astype(np.float32)
        y_te = np.random.randint(0, 2, 50)
        return X_tr, y_tr, X_te, y_te

    from medmnist import PneumoniaMNIST
    tf = transforms.Compose([transforms.ToTensor()])
    tr = PneumoniaMNIST(split="train", transform=tf, download=True)
    te = PneumoniaMNIST(split="test",  transform=tf, download=True)
    X_tr = np.stack([tr[i][0].numpy() for i in range(len(tr))])
    y_tr = np.array([int(tr[i][1]) for i in range(len(tr))])
    X_te = np.stack([te[i][0].numpy() for i in range(len(te))])
    y_te = np.array([int(te[i][1]) for i in range(len(te))])
    print(f"PneumoniaMNIST: train={X_tr.shape}, test={X_te.shape}")
    return X_tr, y_tr, X_te, y_te


# ─── CNN Encoder ──────────────────────────────────────────────────────────────
class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(), nn.MaxPool2d(2)
        )
    def forward(self, x): return self.block(x)


class CNNEncoder(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            CNNBlock(1, 32), CNNBlock(32, 64), CNNBlock(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(128, feature_dim), nn.ReLU())
    def forward(self, x): return self.proj(self.features(x))


# ─── Modèle CNN+LSTM ──────────────────────────────────────────────────────────
class HybridCNNLSTM(nn.Module):
    def __init__(self, feature_dim=64, hidden_dim=128, num_layers=2):
        super().__init__()
        self.cnn = CNNEncoder(feature_dim)
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x_seq):
        b, s, c, h, w = x_seq.shape
        feats = self.cnn(x_seq.view(b*s, c, h, w)).view(b, s, -1)
        _, (h_n, _) = self.lstm(feats)
        return self.fc(h_n[-1])


# ─── Modèle CNN seul (baseline) ──────────────────────────────────────────────
class CNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            CNNBlock(1, 32), CNNBlock(32, 64), CNNBlock(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)


# ─── Dataset séquentiel ───────────────────────────────────────────────────────
class SeqDataset(Dataset):
    SEQ = 4
    def __init__(self, X, y, augment=False, seed=42):
        self.X, self.y = X.astype(np.float32), y
        self.aug = augment
        self.rng = np.random.default_rng(seed)
        idx_by_lbl = {0: np.where(y==0)[0], 1: np.where(y==1)[0]}
        self.seqs = [(self.rng.choice(idx_by_lbl[lbl], self.SEQ, replace=True), lbl)
                     for lbl in y]
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        indices, lbl = self.seqs[i]
        imgs = []
        for j, idx in enumerate(indices):
            img = self.X[idx].copy()
            if self.aug and j > 0:
                img = np.clip(img + self.rng.normal(0, 0.02, img.shape).astype(np.float32), 0, 1)
            imgs.append(torch.tensor(img))
        return torch.stack(imgs), torch.tensor(lbl, dtype=torch.float32)


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ─── Entraînement générique ───────────────────────────────────────────────────
def train_eval(model, train_loader, test_loader, lr, epochs,
               is_sequential=False, name="Model"):
    torch.manual_seed(SEED)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )
    best_val, best_state = 0.0, None
    n_params = sum(p.numel() for p in model.parameters())
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        t_preds, t_labels = [], []
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device).float()
            optimizer.zero_grad()
            pred = model(x).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_preds.extend((pred > 0.5).int().cpu().numpy())
            t_labels.extend(y.int().cpu().numpy())

        model.eval()
        v_preds, v_labels, v_probs = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch[0].to(device), batch[1].to(device).float()
                pred = model(x).squeeze().cpu().numpy()
                v_preds.extend((pred > 0.5).astype(int))
                v_labels.extend(y.int().cpu().numpy())
                v_probs.extend(pred if hasattr(pred, '__len__') else [float(pred)])

        t_acc = accuracy_score(t_labels, t_preds)
        v_acc = accuracy_score(v_labels, v_preds)
        scheduler.step(v_acc)
        if v_acc > best_val:
            best_val = v_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{name}] Ep {epoch:02d}/{epochs} | train={t_acc:.4f} | val={v_acc:.4f}")

    elapsed = time.perf_counter() - t0
    model.load_state_dict(best_state)
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1]
            pred = model(x).squeeze().cpu().numpy()
            preds.extend((pred > 0.5).astype(int))
            labels.extend(y.int().numpy())
            probs.extend(pred if hasattr(pred, '__len__') else [float(pred)])

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.5
    print(f"  [{name}] FINAL -> acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  "
          f"time={elapsed:.1f}s  params={n_params:,}")
    return {"accuracy": round(acc,4), "auc": round(auc,4),
            "f1_macro": round(f1,4), "n_params": n_params,
            "train_time_sec": round(elapsed,1)}


def main():
    args = parse_args()
    EPOCHS = args.epochs
    BATCH  = 32

    X_tr, y_tr, X_te, y_te = load_pneumonia(mock=args.mock)

    # ── CNN baseline ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}\nCNN BASELINE\n{'='*55}")
    cnn_tr = DataLoader(SimpleDataset(X_tr, y_tr), BATCH, shuffle=True)
    cnn_te = DataLoader(SimpleDataset(X_te, y_te), BATCH, shuffle=False)
    torch.manual_seed(SEED)
    cnn_base = CNNBaseline().to(device)
    cnn_res = train_eval(cnn_base, cnn_tr, cnn_te, lr=1e-3,
                          epochs=EPOCHS, name="CNN_seul")

    # ── CNN+LSTM ──────────────────────────────────────────────────────────────
    print(f"\n{'='*55}\nCNN + LSTM (seq=4)\n{'='*55}")
    seq_tr = DataLoader(SeqDataset(X_tr, y_tr, augment=True), BATCH, shuffle=True)
    seq_te = DataLoader(SeqDataset(X_te, y_te, augment=False), BATCH, shuffle=False)
    torch.manual_seed(SEED)
    hybrid = HybridCNNLSTM(64, 128, 2).to(device)
    # Charger poids CNN si disponibles
    if os.path.exists("best_cnn_pneumonia.pth"):
        try:
            state = torch.load("best_cnn_pneumonia.pth", map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            feat_state = {k: v for k, v in state.items() if k.startswith("features.")}
            # Mapper vers cnn.features.*
            mapped = {"cnn." + k: v for k, v in feat_state.items()}
            hybrid.load_state_dict(mapped, strict=False)
            print("  [INFO] Poids CNN charges depuis best_cnn_pneumonia.pth")
        except Exception as e:
            print(f"  [WARN] Chargement poids CNN echoue: {e}")
    lstm_res = train_eval(hybrid, seq_tr, seq_te, lr=5e-4,
                           epochs=EPOCHS, is_sequential=True, name="CNN+LSTM")

    # ── Résumé ────────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("RÉSUMÉ HYBRIDES")
    print(f"{'='*55}")
    print(f"  CNN seul  : acc={cnn_res['accuracy']:.4f}  AUC={cnn_res['auc']:.4f}")
    print(f"  CNN+LSTM  : acc={lstm_res['accuracy']:.4f}  AUC={lstm_res['auc']:.4f}")

    # Mise à jour metrics.json
    metrics_path = "metrics.json"
    m = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
    m.setdefault("hybrid", {})
    m["hybrid"]["cnn_baseline"] = cnn_res
    m["hybrid"]["cnn_lstm"]     = lstm_res
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] metrics.json mis a jour")


if __name__ == "__main__":
    main()
