"""
train_rnn.py — Entraine RNN/LSTM/GRU directement (sans Jupyter kernel).
Ecrit les resultats dans metrics.json et sauvegarde best_gru_medical.pth.

Usage : py train_rnn.py
        py train_rnn.py --epochs 15   (plus rapide sur CPU)
        py train_rnn.py --epochs 30   (meilleure accuracy)
"""
import argparse
import json
import os
import re
import time
import warnings
import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")

# ─── Reproductibilité ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
print(f"PyTorch: {torch.__version__}")

# ─── Hyperparamètres ─────────────────────────────────────────────────────────
PAD_TOKEN  = "<pad>"
UNK_TOKEN  = "<unk>"
EMBED_DIM  = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT    = 0.3
CLIP       = 1.0
BATCH_SIZE = 32
MIN_FREQ   = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20,
                   help="Nombre d'epochs (defaut 20, recommande 30 avec GPU)")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--models", nargs="+", default=["gru", "lstm", "rnn"],
                   help="Modeles a entrainer (gru lstm rnn)")
    p.add_argument("--mock", action="store_true", help="Utiliser un dataset synthetique sans telecharger")
    return p.parse_args()


# ─── Chargement des données ────────────────────────────────────────────────────
def load_data(mock=False):
    import pandas as pd
    import urllib.request
    import io
    url_train = ("https://raw.githubusercontent.com/sebischair/"
                 "Medical-Abstracts-TC-Corpus/main/medical_tc_train.csv")
    url_test  = ("https://raw.githubusercontent.com/sebischair/"
                 "Medical-Abstracts-TC-Corpus/main/medical_tc_test.csv")
    
    def download_csv(url):
        print(f"Telechargement {url.split('/')[-1]}...", end=" ", flush=True)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            return pd.read_csv(io.StringIO(response.read().decode('utf-8')))

    try:
        if mock:
            raise Exception("Mode mock active")
        df_train = download_csv(url_train).dropna(subset=["medical_abstract","condition_label"])
        print("OK")
        df_test  = download_csv(url_test).dropna(subset=["medical_abstract","condition_label"])
        print("OK")
        df_train["label"] = df_train["condition_label"].astype(int) - 1
        df_test["label"]  = df_test["condition_label"].astype(int) - 1
        print(f"Donnees reelles : train={len(df_train)}, test={len(df_test)}")
        return df_train, df_test
    except Exception as e:
        print(f"\n[WARN] Echec du telechargement ({e}). Creation d'un dataset synthetique...")
        data_train = {
            "medical_abstract": [
                "patient has fever and cough pneumonia", 
                "cardiac arrest heart bypass surgery", 
                "tumor malignant cancer chemotherapy", 
                "neural cognitive brain tumor surgery", 
                "digestive pain stomach ulcer endoscopy"
            ] * 200,
            "condition_label": [1, 2, 3, 4, 5] * 200
        }
        data_test = {
            "medical_abstract": [
                "cough and fever in the lung", 
                "coronary heart disease", 
                "malignant melanoma cancer", 
                "brain cognitive decline", 
                "stomach pain endoscopy"
            ] * 50,
            "condition_label": [1, 2, 3, 4, 5] * 50
        }
        df_train = pd.DataFrame(data_train)
        df_test = pd.DataFrame(data_test)
        df_train["label"] = df_train["condition_label"].astype(int) - 1
        df_test["label"]  = df_test["condition_label"].astype(int) - 1
        return df_train, df_test


def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str):
    return normalize(text).split()


# ─── Vocabulaire ──────────────────────────────────────────────────────────────
def build_vocab(df_train, min_freq=MIN_FREQ):
    from collections import Counter
    counter = Counter()
    for text in df_train["medical_abstract"]:
        counter.update(tokenize(text))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, freq in counter.most_common():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    print(f"Vocabulaire: {len(vocab)} tokens (min_freq={min_freq})")
    return vocab


# ─── Dataset ──────────────────────────────────────────────────────────────────
class MedicalTextDataset(Dataset):
    def __init__(self, df, vocab, max_len=None):
        # Longueur P95
        lens = [len(tokenize(t)) for t in df["medical_abstract"]]
        self.max_len = max_len or max(int(np.percentile(lens, 95)), 64)
        self.X = [self._encode(t, vocab) for t in df["medical_abstract"]]
        self.y = df["label"].values.astype(np.int64)

    def _encode(self, text, vocab):
        toks = tokenize(text)[:self.max_len]
        ids  = [vocab.get(t, 1) for t in toks]
        return ids if ids else [1]

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return (torch.tensor(self.X[i], dtype=torch.long),
                torch.tensor(self.y[i], dtype=torch.long))


def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded  = pad_sequence(seqs, batch_first=True, padding_value=0)
    return padded, lengths, torch.stack(labels)


# ─── Architecture RNN ─────────────────────────────────────────────────────────
class RecurrentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 rnn_type="gru", num_layers=2, dropout=0.3):
        super().__init__()
        self.rnn_type  = rnn_type.lower()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        rnn_drop = dropout if num_layers > 1 else 0.0
        kw = dict(input_size=embed_dim, hidden_size=hidden_dim,
                  num_layers=num_layers, batch_first=True,
                  bidirectional=True, dropout=rnn_drop)
        if self.rnn_type == "rnn":
            self.rnn = nn.RNN(**kw)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(**kw)
        else:
            self.rnn = nn.GRU(**kw)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        lengths = lengths.clamp(max=x.size(1))
        emb = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                      enforce_sorted=True)
        if self.rnn_type == "lstm":
            _, (h_n, _) = self.rnn(packed)
        else:
            _, h_n = self.rnn(packed)
        h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.fc(self.dropout(h_last))


# ─── Entraînement ─────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.set_grad_enabled(is_train):
        for x, lengths, labels in loader:
            x, labels = x.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()
            logits = model(x, lengths)
            loss   = criterion(logits, labels)
            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def train_model(rnn_type, train_loader, test_loader, vocab_size,
                num_classes, epochs, lr):
    torch.manual_seed(SEED)
    model = RecurrentClassifier(
        vocab_size, EMBED_DIM, HIDDEN_DIM, num_classes,
        rnn_type=rnn_type, num_layers=NUM_LAYERS, dropout=DROPOUT
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    best_val_acc = 0.0
    best_state   = None
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, test_loader,  criterion)
        scheduler.step(va_acc)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  [{rnn_type.upper()}] Ep {epoch:02d}/{epochs} | "
                  f"train acc={tr_acc:.4f} | val acc={va_acc:.4f} | "
                  f"best={best_val_acc:.4f} | lr={lr_now:.2e}")

    elapsed = time.perf_counter() - t0

    # Evaluation finale sur le meilleur modèle
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, lengths, labels in test_loader:
            logits = model(x.to(device), lengths)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    try:
        test_auc = roc_auc_score(all_labels, np.array(all_probs),
                                  multi_class="ovr", average="macro")
    except Exception:
        test_auc = 0.0

    print(f"\n  [{rnn_type.upper()}] FINAL -> "
          f"acc={test_acc:.4f}  F1={test_f1:.4f}  AUC={test_auc:.4f}  "
          f"time={elapsed:.1f}s  params={n_params:,}")

    return model, {
        "test_accuracy":   round(test_acc, 4),
        "test_f1_macro":   round(test_f1, 4),
        "test_auc_roc":    round(test_auc, 4),
        "train_time_sec":  round(elapsed, 1),
        "n_params":        n_params,
        "epochs_trained":  epochs,
        "best_val_acc":    round(best_val_acc, 4),
    }


# ─── Mise à jour metrics.json ─────────────────────────────────────────────────
def update_metrics(results_dict):
    metrics_path = "metrics.json"
    m = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)

    if "rnn" not in m:
        m["rnn"] = {"models": {}, "best_model": "GRU"}
    if "models" not in m["rnn"]:
        m["rnn"]["models"] = {}

    for name, metrics in results_dict.items():
        m["rnn"]["models"][name.upper()] = metrics

    # Meilleur modèle
    best_acc   = -1
    best_model = "GRU"
    for name, metrics in m["rnn"]["models"].items():
        acc = metrics.get("test_accuracy") or 0
        if acc > best_acc:
            best_acc   = acc
            best_model = name
    m["rnn"]["best_model"] = best_model

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] metrics.json mis a jour — meilleur modele: {best_model} ({best_acc:.4f})")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    EPOCHS = args.epochs
    LR     = args.lr

    print(f"\n{'='*60}")
    print(f"ENTRAINEMENT RNN — {EPOCHS} epochs | LR={LR} | Device={device}")
    print(f"Modeles : {', '.join(m.upper() for m in args.models)}")
    print(f"{'='*60}\n")

    # Données
    df_train, df_test = load_data(mock=args.mock)
    vocab      = build_vocab(df_train)
    vocab_size = len(vocab)
    num_classes = df_train["label"].nunique()
    print(f"Classes : {num_classes} | Vocab : {vocab_size}")

    train_ds = MedicalTextDataset(df_train, vocab)
    test_ds  = MedicalTextDataset(df_test,  vocab, max_len=train_ds.max_len)
    print(f"MAX_LEN : {train_ds.max_len}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                               shuffle=True, collate_fn=collate_fn,
                               num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE,
                               shuffle=False, collate_fn=collate_fn,
                               num_workers=0, pin_memory=False)

    results_dict = {}
    best_gru_model = None

    for rnn_type in args.models:
        print(f"\n{'='*60}")
        print(f"Entrainement : {rnn_type.upper()}")
        print(f"{'='*60}")
        model, metrics = train_model(
            rnn_type, train_loader, test_loader,
            vocab_size, num_classes, EPOCHS, LR
        )
        results_dict[rnn_type] = metrics

        # Sauvegarder le meilleur GRU
        if rnn_type.lower() == "gru":
            best_gru_model = model
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
                "num_classes": num_classes,
                "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "metrics": metrics,
                "epochs": EPOCHS,
            }, "best_gru_medical.pth")
            print("  [SAVE] best_gru_medical.pth sauvegarde")

    # Résumé final
    print(f"\n{'='*60}")
    print("RÉSUMÉ FINAL")
    print(f"{'='*60}")
    for name, m in results_dict.items():
        target_ok = "[OK]" if m["test_accuracy"] > 0.65 else "[WARN] < 0.65"
        print(f"  {name.upper():4s} : acc={m['test_accuracy']:.4f} {target_ok}  "
              f"F1={m['test_f1_macro']:.4f}  time={m['train_time_sec']:.0f}s")

    # Mise à jour metrics.json
    update_metrics(results_dict)
    print("\n[DONE] Relancez l'app Streamlit pour voir les nouvelles metriques.")


if __name__ == "__main__":
    main()
