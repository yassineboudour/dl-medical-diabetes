"""
Génère part1_mlp_diabetes.ipynb (Partie I — MLP PyTorch, Pima Diabetes).
Exécuter : python generate_part1.py
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()

# ════════════════════════════════════════════════════════
# SECTION 1 — Théorie (Markdown)
# ════════════════════════════════════════════════════════
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """# Partie I — Perceptron multicouche (MLP) pour la prédiction du diabète

## Section 1 — Fondements théoriques PyTorch

### 1.1 `nn.Module`, `forward()`, `__init__` et les paramètres entraînables

En PyTorch, tout réseau de neurones hérite de `torch.nn.Module`. La méthode `__init__` déclare les couches (par ex. `nn.Linear`, `nn.BatchNorm1d`) ; leurs poids et biais sont enregistrés automatiquement comme **paramètres** (`nn.Parameter`) et deviennent entraînables lorsque `requires_grad=True`. La méthode `forward(self, x)` définit le **graphe de calcul** : elle transforme un tenseur d'entrée $\\mathbf{x} \\in \\mathbb{R}^{B \\times d}$ (batch de taille $B$) en sortie $\\hat{\\mathbf{y}}$. L'entraînement repose sur la **rétropropagation** : après un passage avant, `loss.backward()` propage les gradients $\\partial \\mathcal{L}/\\partial \\theta$ vers chaque paramètre $\\theta$, puis l'optimiseur met à jour $\\theta \\leftarrow \\theta - \\eta \\nabla_\\theta \\mathcal{L}$."""
    )
)

nb.cells.append(
    nbf.v4.new_markdown_cell(
        """### 1.2 `state_dict`, `device` et propagation avant / arrière

Le dictionnaire `state_dict()` regroupe tous les tenseurs persistants du modèle (poids, biais, statistiques BatchNorm). On peut sauvegarder un checkpoint avec `torch.save(model.state_dict(), path)` et le recharger avec `load_state_dict`. Le **`device`** (`cpu` ou `cuda`) impose où résident tenseurs et paramètres ; mélanger CPU et GPU provoque des erreurs. La **propagation avant** évalue $\\hat{y} = f_\\theta(x)$ ; la **rétropropagation** calcule les gradients via la règle de la chaîne sur ce graphe dynamique (autograd)."""
    )
)

nb.cells.append(
    nbf.v4.new_markdown_cell(
        """### 1.3 Fonctions d'activation (ReLU, Sigmoid, Tanh)

**ReLU** (couches cachées) :
$$\\mathrm{ReLU}(z) = \\max(0, z)$$

**Sigmoid** (sortie binaire, probabilité) :
$$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$

**Tanh** (alternative bornée) :
$$\\tanh(z) = \\frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$$

Pour la classification binaire médicale (diabète oui/non), une sortie **Sigmoid** produit $\\hat{p} \\in (0,1)$ interprétable comme risque estimé — utile pour le seuillage clinique et la courbe ROC."""
    )
)

nb.cells.append(
    nbf.v4.new_markdown_cell(
        """### 1.4 `BCELoss` et `BCEWithLogitsLoss`

**BCELoss** (entrée déjà passée par Sigmoid) — perte moyenne sur le batch :
$$\\mathcal{L}_{\\mathrm{BCE}} = -\\frac{1}{N}\\sum_{i=1}^{N} \\left[ y_i \\log(\\hat{p}_i) + (1-y_i)\\log(1-\\hat{p}_i) \\right]$$

**BCEWithLogitsLoss** (logits bruts $z_i$, Sigmoid fusionné pour stabilité numérique) :
$$\\mathcal{L}_{\\mathrm{BCEWL}} = -\\frac{1}{N}\\sum_{i=1}^{N} \\left[ y_i \\log(\\sigma(z_i)) + (1-y_i)\\log(1-\\sigma(z_i)) \\right]$$

**Justification sur Pima Diabetes :** nous utilisons **Sigmoid en sortie du MLP** + **`nn.BCELoss`**, cohérent avec une probabilité calibrée. `BCEWithLogitsLoss` serait préférable si la sortie était linéaire (sans Sigmoid) ; ici l'architecture impose Sigmoid + BCELoss, standard en projet pédagogique pour lier explicitement activation et fonction de coût."""
    )
)

# ════════════════════════════════════════════════════════
# SECTION 2 — Installation et imports
# ════════════════════════════════════════════════════════
nb.cells.append(
    nbf.v4.new_markdown_cell("""## Section 2 — Installation et imports""")
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# Installation des bibliothèques (compatible Google Colab)
!pip install imbalanced-learn scikit-learn pandas numpy matplotlib seaborn torch -q"""
    )
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# --- Imports standards et Deep Learning ---
import os  # chemins de fichiers pour sauvegarde du modèle
import copy  # copie profonde du meilleur state_dict (early stopping)
import warnings  # masquer les avertissements non critiques en démo

import numpy as np  # calculs numériques et seuil de Youden
import pandas as pd  # chargement du CSV Pima
import matplotlib.pyplot as plt  # visualisation des courbes et métriques
import seaborn as sns  # heatmap de confusion

from sklearn.model_selection import train_test_split  # split stratifié 70/15/15
from sklearn.preprocessing import StandardScaler  # normalisation des features
from sklearn.metrics import (  # métriques de classification binaire
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)

import torch  # framework PyTorch
import torch.nn as nn  # couches et pertes
from torch.utils.data import TensorDataset, DataLoader  # pipelines batch

warnings.filterwarnings("ignore")  # affichage plus lisible sur Colab
sns.set_theme(style="whitegrid", palette="husl")  # style Seaborn global
plt.rcParams["figure.dpi"] = 120  # résolution des figures
plt.rcParams["font.size"] = 11  # taille de police par défaut

# Graine aléatoire pour reproductibilité partielle
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("Imports terminés. PyTorch version:", torch.__version__)"""
    )
)

# ════════════════════════════════════════════════════════
# SECTION 3 — Données
# ════════════════════════════════════════════════════════
nb.cells.append(
    nbf.v4.new_markdown_cell("""## Section 3 — Chargement et préparation des données""")
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# URL officielle du dataset Pima Indians Diabetes (Jason Brownlee)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# Noms de colonnes selon la documentation UCI / Kaggle
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

# Lecture du CSV sans en-tête
df = pd.read_csv(url, names=cols)

# Colonnes où 0 signifie souvent une valeur manquante (impossible biologiquement)
cols_zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Imputation : remplacer 0 par la médiane de la colonne (calculée sur valeurs > 0)
for col in cols_zero_as_missing:
    median_val = df.loc[df[col] > 0, col].median()  # médiane robuste hors zéros
    df[col] = df[col].replace(0, median_val)  # substitution des zéros aberrants

print("Aperçu des données après imputation :")
display(df.head())

# Séparation features / cible binaire
feature_cols = [c for c in cols if c != "Outcome"]
X = df[feature_cols].values.astype(np.float32)
y = df["Outcome"].values.astype(np.float32)

# Distribution des classes (déséquilibre typique ~65% / 35%)
class_counts = df["Outcome"].value_counts().sort_index()
print("\\nDistribution des classes :")
print(class_counts)
print("Proportions :", (class_counts / len(df)).round(3).to_dict())

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=df["Outcome"], palette=["skyblue", "salmon"], ax=ax)
ax.set_title("Distribution des classes — Pima Diabetes")
ax.set_xlabel("Outcome (0 = non diabétique, 1 = diabétique)")
ax.set_ylabel("Effectif")
ax.legend(handles=ax.patches, labels=["Classe 0", "Classe 1"], title="Légende")
plt.tight_layout()
plt.show()

# Normalisation : StandardScaler ajusté UNIQUEMENT sur le futur train
scaler = StandardScaler()

# Split stratifié 70 % train / 15 % val / 15 % test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

# Fit du scaler sur train seulement (évite la fuite d'information)
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

print(f"Tailles — train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")

# Conversion en tenseurs PyTorch
def make_tensors(X_arr, y_arr):
    \"\"\"Convertit arrays numpy en tenseurs float32.\"\"\"
    X_t = torch.tensor(X_arr, dtype=torch.float32)
    y_t = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1)  # forme (N, 1)
    return X_t, y_t

X_train_t, y_train_t = make_tensors(X_train, y_train)
X_val_t, y_val_t = make_tensors(X_val, y_val)
X_test_t, y_test_t = make_tensors(X_test, y_test)

# DataLoaders PyTorch — batch_size=32, shuffle=True sur train
BATCH_SIZE = 32
train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False
)
test_loader = DataLoader(
    TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False
)

print("DataLoaders prêts — batch_size =", BATCH_SIZE)"""
    )
)

# ════════════════════════════════════════════════════════
# SECTION 4 — nn.Sequential
# ════════════════════════════════════════════════════════
nb.cells.append(
    nbf.v4.new_markdown_cell("""## Section 4 — MLP avec `nn.Sequential`""")
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# Architecture : 8 → 64 → 128 → 64 → 1 avec BatchNorm, ReLU, Dropout, Sigmoid
INPUT_DIM = 8  # nombre de features tabulaires

model_seq = nn.Sequential(
    nn.Linear(INPUT_DIM, 64),  # couche dense 8 → 64
    nn.BatchNorm1d(64),  # normalisation par mini-batch
    nn.ReLU(),  # activation non linéaire
    nn.Dropout(p=0.3),  # régularisation : 30 % de neurones masqués
    nn.Linear(64, 128),  # couche dense 64 → 128
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(128, 64),  # couche dense 128 → 64
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(64, 1),  # couche de sortie scalaire
    nn.Sigmoid(),  # probabilité ∈ (0, 1) pour BCELoss
)

# Test de forme sur un batch factice
with torch.no_grad():
    dummy = torch.randn(4, INPUT_DIM)  # batch de 4 échantillons
    out_seq = model_seq(dummy)
    print("Modèle Sequential — sortie shape:", out_seq.shape)
    print("Exemple de probabilités:", out_seq.squeeze().numpy())"""
    )
)

# ════════════════════════════════════════════════════════
# SECTION 5 — Classe personnalisée
# ════════════════════════════════════════════════════════
nb.cells.append(
    nbf.v4.new_markdown_cell("""## Section 5 — MLP avec classe personnalisée `nn.Module`""")
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """class MLPDiabetes(nn.Module):
    \"\"\"MLP tabulaire pour Pima Diabetes — même architecture que Sequential.\"\"\"

    def __init__(self, input_dim=8, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()
        self._activations = {}  # stockage pour get_activations()

    def forward(self, x):
        x = self.drop(self.relu(self.bn1(self.fc1(x))))
        self._activations["hidden1"] = x.detach()
        x = self.drop(self.relu(self.bn2(self.fc2(x))))
        self._activations["hidden2"] = x.detach()
        x = self.drop(self.relu(self.bn3(self.fc3(x))))
        self._activations["hidden3"] = x.detach()
        x = self.sigmoid(self.fc4(x))
        self._activations["output"] = x.detach()
        return x

    def get_activations(self):
        \"\"\"Retourne le dict des activations du dernier forward.\"\"\"
        return self._activations

    def summary(self):
        \"\"\"Affiche un résumé textuel de l'architecture.\"\"\"
        print("=" * 60)
        print("MLPDiabetes — résumé")
        print("=" * 60)
        total = 0
        for name, p in self.named_parameters():
            n = p.numel()
            total += n
            print(f"{name:40s} {tuple(p.shape)!s:20s} {n:>8d} params")
        print("-" * 60)
        print(f"Total paramètres entraînables : {total}")
        print("=" * 60)


model = MLPDiabetes(input_dim=INPUT_DIM, dropout=0.3)
model.summary()

with torch.no_grad():
    dummy = torch.randn(4, INPUT_DIM)
    out = model(dummy)
    acts = model.get_activations()
    print("\\nActivations capturées :", {k: v.shape for k, v in acts.items()})"""
    )
)

# ════════════════════════════════════════════════════════
# Helpers train / eval (réutilisés sections 7–10)
# ════════════════════════════════════════════════════════
nb.cells.append(
    nbf.v4.new_code_cell(
        """# --- Fonctions utilitaires d'entraînement et d'évaluation ---

criterion = nn.BCELoss()  # compatible avec sortie Sigmoid du MLP


def count_parameters(module):
    \"\"\"Compte les paramètres entraînables d'un nn.Module.\"\"\"
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def train_one_epoch(model, loader, optimizer, device):
    \"\"\"Une epoch d'entraînement — retourne loss moyenne et accuracy.\"\"\"
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        preds = (outputs >= 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    \"\"\"Évaluation — loss, accuracy, probabilités et labels.\"\"\"
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        total_loss += loss.item() * X_batch.size(0)
        all_probs.append(outputs.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())
        preds = (outputs >= 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    probs = np.vstack(all_probs).ravel()
    labels = np.vstack(all_labels).ravel()
    return total_loss / total, correct / total, probs, labels


def init_gaussian(module):
    \"\"\"Initialisation gaussienne N(0, 0.02) sur Linear.\"\"\"
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        nn.init.zeros_(module.bias)


def init_constant(module):
    \"\"\"Initialisation constante (poids = 0.01, biais = 0).\"\"\"
    if isinstance(module, nn.Linear):
        nn.init.constant_(module.weight, 0.01)
        nn.init.zeros_(module.bias)


def init_xavier(module):
    \"\"\"Initialisation Xavier uniforme sur Linear.\"\"\"
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


def apply_init(model, init_fn):
    \"\"\"Réinitialise tous les Linear du modèle.\"\"\"
    model.apply(init_fn)
    return model


def run_training_epochs(model, train_loader, val_loader, device, epochs=30, lr=1e-3):
    \"\"\"Boucle courte pour comparaison d'initialisations (section 7).\"\"\"
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    hist_train, hist_val = [], []
    for _ in range(epochs):
        tr_loss, _ = train_one_epoch(model, train_loader, optimizer, device)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, device)
        hist_train.append(tr_loss)
        hist_val.append(va_loss)
    _, final_acc, _, _ = evaluate(model, val_loader, device)
    return hist_train, hist_val, final_acc


print("Fonctions utilitaires définies.")"""
    )
)

# ════════════════════════════════════════════════════════
# SECTION 6 — Inspection paramètres
# ════════════════════════════════════════════════════════
nb.cells.append(
    nbf.v4.new_markdown_cell("""## Section 6 — Inspection des paramètres""")
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# Affichage nom + shape via named_parameters()
print("=== named_parameters() ===")
for name, param in model.named_parameters():
    print(f"{name:45s} shape={tuple(param.shape)}  requires_grad={param.requires_grad}")

# Clés du state_dict
print("\\n=== state_dict() — clés ===")
for key in model.state_dict().keys():
    print(" ", key)

# Nombre total de paramètres entraînables
n_params = count_parameters(model)
print(f"\\nNombre total de paramètres entraînables : {n_params}")"""
    )
)

# ════════════════════════════════════════════════════════
# SECTION 7 — Trois initialisations
# ════════════════════════════════════════════════════════
nb.cells.append(
    nbf.v4.new_markdown_cell("""## Section 7 — Comparaison de trois stratégies d'initialisation""")
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# Device CPU pour cette expérience (section 9 formalisera CUDA)
device_init = torch.device("cpu")

INIT_CONFIGS = [
    ("Gaussienne", init_gaussian),
    ("Constante", init_constant),
    ("Xavier", init_xavier),
]

EPOCHS_INIT = 30
histories = {}

for label, init_fn in INIT_CONFIGS:
    m = MLPDiabetes(input_dim=INPUT_DIM, dropout=0.3)
    apply_init(m, init_fn)
    m = m.to(device_init)
    tr_hist, va_hist, val_acc = run_training_epochs(
        m, train_loader, val_loader, device_init, epochs=EPOCHS_INIT, lr=1e-3
    )
    histories[label] = {"train": tr_hist, "val": va_hist, "val_acc": val_acc}
    print(f"[{label}] Accuracy validation finale : {val_acc:.4f}")

# Tracé des 3 courbes de loss validation sur le même graphe
plt.figure(figsize=(10, 6))
for label, h in histories.items():
    plt.plot(range(1, EPOCHS_INIT + 1), h["val"], label=f"Val loss — {label}")
plt.xlabel("Epoch")
plt.ylabel("BCELoss (validation)")
plt.title("Comparaison des initialisations — courbes de loss validation")
plt.legend(title="Stratégie")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Courbes train pour compléter l'analyse
plt.figure(figsize=(10, 6))
for label, h in histories.items():
    plt.plot(range(1, EPOCHS_INIT + 1), h["train"], linestyle="--", label=f"Train loss — {label}")
plt.xlabel("Epoch")
plt.ylabel("BCELoss (entraînement)")
plt.title("Comparaison des initialisations — courbes de loss entraînement")
plt.legend(title="Stratégie")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\\nRécapitulatif accuracy validation :")
for label, h in histories.items():
    print(f"  {label:12s} : {h['val_acc']:.4f}")"""
    )
)

# ════════════════════════════════════════════════════════
# SECTION 8 — Entraînement complet
# ════════════════════════════════════════════════════════
nb.cells.append(
    nbf.v4.new_markdown_cell("""## Section 8 — Boucle d'entraînement complète (Adam, scheduler, early stopping)""")
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# Réinitialisation Xavier (bon compromis empirique sur ce MLP)
device_train = torch.device("cpu")
model = MLPDiabetes(input_dim=INPUT_DIM, dropout=0.3)
apply_init(model, init_xavier)
model = model.to(device_train)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5
)

BEST_PATH = "best_mlp_diabetes.pth"
MAX_EPOCHS = 200
EARLY_PATIENCE = 10
best_val_loss = float("inf")
epochs_no_improve = 0
best_state = None

history_full = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "lr": [],
}

for epoch in range(1, MAX_EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device_train)
    va_loss, va_acc, _, _ = evaluate(model, val_loader, device_train)
    scheduler.step(va_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    history_full["train_loss"].append(tr_loss)
    history_full["val_loss"].append(va_loss)
    history_full["train_acc"].append(tr_acc)
    history_full["val_acc"].append(va_acc)
    history_full["lr"].append(current_lr)

    if va_loss < best_val_loss:
        best_val_loss = va_loss
        epochs_no_improve = 0
        best_state = copy.deepcopy(model.state_dict())
        torch.save(best_state, BEST_PATH)
    else:
        epochs_no_improve += 1

    if epoch % 10 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:3d} | train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
            f"val_acc={va_acc:.4f} lr={current_lr:.6f}"
        )

    if epochs_no_improve >= EARLY_PATIENCE:
        print(f"Early stopping à l'epoch {epoch} (patience={EARLY_PATIENCE})")
        break

# Rechargement du meilleur modèle sauvegardé
model.load_state_dict(torch.load(BEST_PATH, map_location=device_train))
print("Meilleur modèle rechargé depuis", BEST_PATH)"""
    )
)

# ════════════════════════════════════════════════════════
# SECTION 9 — CPU / GPU
# ════════════════════════════════════════════════════════
nb.cells.append(
    nbf.v4.new_markdown_cell("""## Section 9 — Gestion CPU / GPU""")
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# Sélection automatique du device (CUDA si disponible, sinon CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device sélectionné :", device)

# Déplacement du modèle et vérification des paramètres
model = model.to(device)
first_param = next(model.parameters())
assert str(first_param.device).startswith(str(device).split(":")[0]), (
    f"Le modèle devrait être sur {device}, trouvé {first_param.device}"
)

# Déplacement d'un batch de validation et forward de contrôle
X_check, y_check = next(iter(val_loader))
X_check, y_check = X_check.to(device), y_check.to(device)
with torch.no_grad():
    out_check = model(X_check)
assert out_check.device.type == X_check.device.type, "Sortie et entrée sur devices différents"
print("Assertions device OK — batch shape:", tuple(X_check.shape), "sortie:", tuple(out_check.shape))"""
    )
)

# ════════════════════════════════════════════════════════
# SECTION 10 — Évaluation
# ════════════════════════════════════════════════════════
nb.cells.append(
    nbf.v4.new_markdown_cell("""## Section 10 — Évaluation et métriques""")
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# Évaluation sur le jeu de test (modèle sur device)
test_loss, test_acc, y_prob, y_true = evaluate(model, test_loader, device)

# Prédictions par défaut (seuil 0.5)
y_pred_default = (y_prob >= 0.5).astype(int)

# --- Métriques globales ---
acc = accuracy_score(y_true, y_pred_default)
prec = precision_score(y_true, y_pred_default, zero_division=0)
rec = recall_score(y_true, y_pred_default, zero_division=0)
f1_macro = f1_score(y_true, y_pred_default, average="macro", zero_division=0)
f1_weighted = f1_score(y_true, y_pred_default, average="weighted", zero_division=0)

print("=== Métriques sur le jeu de TEST (seuil = 0.5) ===")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 macro  : {f1_macro:.4f}")
print(f"F1 weighted : {f1_weighted:.4f}")
print(f"BCELoss test : {test_loss:.4f}")

# --- Matrice de confusion ---
cm = confusion_matrix(y_true, y_pred_default)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Prédit 0", "Prédit 1"],
    yticklabels=["Réel 0", "Réel 1"],
)
plt.title("Matrice de confusion — MLP Pima Diabetes (seuil 0.5)")
plt.xlabel("Prédiction")
plt.ylabel("Vérité terrain")
plt.tight_layout()
plt.show()

# --- Courbe ROC et AUC ---
fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], "k--", label="Hasard")
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.title("Courbe ROC — MLP Pima Diabetes")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Seuil optimal : indice de Youden J = TPR - FPR ---
youden_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds_roc[youden_idx]
y_pred_optimal = (y_prob >= optimal_threshold).astype(int)

print(f"\\nSeuil optimal (Youden) : {optimal_threshold:.4f}")
print(f"Accuracy avec seuil Youden : {accuracy_score(y_true, y_pred_optimal):.4f}")
print(f"F1 macro (seuil Youden) : {f1_score(y_true, y_pred_optimal, average='macro', zero_division=0):.4f}")

# --- Courbes Loss et Accuracy train/val (historique section 8) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history_full["train_loss"], label="Train loss")
axes[0].plot(history_full["val_loss"], label="Val loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("BCELoss")
axes[0].set_title("Courbes de loss — entraînement complet")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history_full["train_acc"], label="Train accuracy")
axes[1].plot(history_full["val_acc"], label="Val accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Courbes d'accuracy — entraînement complet")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""
    )
)

# ════════════════════════════════════════════════════════
# SECTION 11 — Analyse critique (400+ mots FR)
# ════════════════════════════════════════════════════════
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """## Section 11 — Analyse critique

### Sequential vs classe personnalisée

L'implémentation via `nn.Sequential` condense l'architecture en une liste linéaire de couches : elle est rapide à prototyper, lisible pour un pipeline fixe 8→64→128→64→1, et suffisante lorsque l'on n'a pas besoin d'accéder aux activations intermédiaires. En revanche, la classe `MLPDiabetes` héritant de `nn.Module` offre une modularité supérieure : la méthode `forward` explicite documente le flux de tenseurs, `get_activations()` permet d'inspecter les représentations cachées (utile pour l'interprétabilité ou le débogage de vanishing/exploding gradients), et `summary()` centralise l'inventaire des paramètres. Dans un contexte académique médical, cette version personnalisée rapproche le code de la description mathématique couche par couche et facilite l'extension future (par ex. branche multi-têtes, attention sur features cliniques).

### Impact des trois initialisations

Les expériences de 30 epochs avec initialisation **gaussienne** (faible écart-type), **constante** (poids identiques) et **Xavier** montrent que la dynamique d'apprentissage dépend fortement du point de départ dans l'espace des paramètres. L'initialisation constante tend à symétriser les neurones d'une même couche et ralentit la spécialisation : les courbes de loss validation convergent souvent plus lentement ou stagnent sur un plateau élevé. L'initialisation gaussienne trop étroite peut sous-exploiter la capacité du réseau au début, tandis que Xavier tient compte de la fan-in/fan-out pour maintenir la variance des activations — ce qui explique en pratique une descente plus régulière sur données tabulaires normalisées. Ces observations confirment que l'initialisation n'est pas un détail d'implémentation mais un levier de stabilité numérique au même titre que le learning rate.

### Limites du MLP pour données médicales déséquilibrées

Sur Pima Diabetes, environ 65 % des patientes sont non diabétiques : un MLP optimisé avec `BCELoss` non pondérée favorise la classe majoritaire, d'où une accuracy trompeuse si l'on ignore le rappel sur la classe 1. Le modèle dense ne modélise pas explicitement les interactions faibles entre variables cliniques (contrairement à des arbres ou du feature engineering), ni l'incertitude épistémique. Les zéros imputés par médiane restent une approximation grossière du mécanisme MNAR (missing not at random). Enfin, sur un petit échantillon (768 lignes), un MLP profond risque le sur-apprentissage malgré Dropout et weight decay — d'où l'intérêt du split stratifié, de l'early stopping et du seuil de Youden pour adapter le compromis sensibilité/spécificité au contexte de dépistage. En synthèse, le MLP constitue une baseline deep learning solide mais doit être complété par des métriques adaptées au déséquilibre (F1, AUC-ROC) et par une réflexion clinique sur le coût des faux négatifs."""
    )
)

# ════════════════════════════════════════════════════════
# SECTION 12 — Synthèse (500+ mots FR)
# ════════════════════════════════════════════════════════
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """## Section 12 — Question de synthèse

**Énoncé :** *Dans quelle mesure un MLP bien paramétré constitue-t-il une solution pertinente pour la classification tabulaire sur un dataset réel, et quelles sont ses principales limites au regard de la structure statistique des données ?*

### Pertinence du MLP sur données tabulaires réelles (cas Pima)

Un perceptron multicouche bien paramétré — ici BatchNorm, ReLU, Dropout 0.3, Adam avec weight decay, réduction du learning rate sur plateau et early stopping — constitue une **baseline deep learning crédible** pour la classification tabulaire. Sur le dataset Pima Indians Diabetes, les entrées sont huit scalaires cliniques sans ordre spatial ni structure séquentielle : un empilement de couches fully-connected est donc **architecturalement aligné** avec la nature i.i.d. des features (après normalisation). Le MLP apprend des compositions non linéaires de Glucose, BMI, Age et DiabetesPedigreeFunction, ce qui permet de tracer une frontière de décision plus flexible qu'une régression logistique simple, tout en restant entraînable sur CPU/GPU avec peu de données.

La chaîne de prétraitement (imputation des zéros, `StandardScaler`, split stratifié 70/15/15) respecte les contraintes statistiques minimales : pas de fuite train→test, préservation des proportions de diabète dans chaque fold. L'usage de **Sigmoid + BCELoss** fournit des probabilités directement exploitables pour la courbe ROC et le **seuil de Youden**, pertinent quand le coût asymétrique des erreurs médicales doit être discuté. Les ~17k paramètres du réseau restent modérés pour 537 exemples d'entraînement : avec régularisation, on obtient typiquement des accuracies de validation/test dans une fourchette réaliste pour la littérature (souvent 70–78 %), ce qui illustre que le MLP **capture une partie de la signalisation** mais n'épuise pas le problème.

### Limites liées à la structure statistique des données

Première limite : **faible taille d'échantillon et dimension modeste**. Les garanties asymptotiques des réseaux profonds (universalité, généralisation PAC) supposent souvent beaucoup plus de données que 768 individus ; la variance des estimateurs de poids reste élevée, et les performances fluctuent selon le seed et l'initialisation (observé en section 7).

Deuxième limite : **déséquilibre des classes** (~35 % positifs). La structure de la loss BCE non pondérée pousse le modèle vers des solutions qui maximisent l'accuracy globale au détriment du rappel sur les cas diabétiques — problématique en dépistage. Des approches statistiquement plus adaptées incluent `pos_weight`, focal loss, ou rééchantillonnage — absentes du MLP « vanilla ».

Troisième limite : **données manquantes codées en zéros** et corrélations médicales non modélisées explicitement. L'imputation par médiane traite un symptôme mais pas le mécanisme de non-réponse ; le MLP ne distingue pas une vraie glycémie basse d'une valeur manquante imputée. Les corrélations entre Insuline et Glucose, ou l'effet seuil de l'âge, pourraient être mieux capturées par des **modèles à structure** (forêts aléatoires, gradient boosting) qui isolent des interactions par feature sans partager tous les poids.

Quatrième limite : **absence d'incertitude calibrée et d'interprétabilité réglementaire**. Un score $\\hat{p}$ produit par Sigmoid n'est pas automatiquement bien calibré ; en milieu clinique, les praticiens exigent souvent des explications locales (SHAP, LIME) difficiles à extraire d'un MLP dense comparé à un arbre de décision.

### Conclusion ancrée sur Pima Diabetes

En conclusion, un MLP bien paramétré est **pertinent comme modèle tabulaire universel et pédagogique** sur Pima : il intègre normalisation, régularisation, suivi ROC et comparaison d'initialisations dans un pipeline PyTorch reproductible sur Colab. Toutefois, au regard de la structure statistique réelle — petit n, classes déséquilibrées, valeurs manquantes masquées, features cliniques hétérogènes — ses limites justifient de le positionner comme **référence deep learning** plutôt que comme solution finale, et d'envisager des modèles statistiques ou arborescents complémentaires, voire des architectures spécialisées tabulaires (TabNet, réseaux avec embeddings catégoriels) lorsque les enjeux de généralisation et d'équité diagnostique deviennent prioritaires."""
    )
)

# ════════════════════════════════════════════════════════
# Métadonnées et écriture du notebook
# ════════════════════════════════════════════════════════
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0",
    },
}

OUTPUT_NOTEBOOK = "part1_mlp_diabetes.ipynb"
with open(OUTPUT_NOTEBOOK, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("OK: part1_mlp_diabetes.ipynb")
