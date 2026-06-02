"""Genere part2_cnn_pneumonia.ipynb — CNN PyTorch sur PneumoniaMNIST (medmnist)."""

import nbformat as nbf

nb = nbf.v4.new_notebook()

# ════════════════════════════════════════════════════════
# SECTION 1 — Introduction theorique (Markdown)
# ════════════════════════════════════════════════════════
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """# Partie 2 — CNN pour la detection de pneumonie (PneumoniaMNIST)

## Contexte et objectif

Ce notebook constitue la **deuxieme partie** d'un projet de deep learning medical. Nous entrainons un **reseau de neurones convolutif (CNN)** en **PyTorch** pour la **classification binaire** de radiographies thoraciques : **normal** versus **pneumonie**, a partir du jeu **PneumoniaMNIST** du benchmark [MedMNIST](https://medmnist.com/).

Contrairement aux donnees tabulaires (MLP), une image possede une **structure spatiale 2D** : les pixels voisins forment des textures et des contours anatomiques. Un CNN exploite cette localite via des **filtres convolutifs** partages sur toute l'image.

## Rappel theorique : la convolution discrete 2D

Une couche convolutive applique un noyau (filtre) $K$ de taille $k \\times k$ sur une carte d'entree $X$. Pour un pixel de sortie en $(i, j)$ :

$$
(Y)_{i,j} = (X * K)_{i,j} = \\sum_{m=0}^{k-1} \\sum_{n=0}^{k-1} X_{i+m,\\,j+n} \\cdot K_{m,n} + b
$$

En deep learning, on empile plusieurs filtres (cartes de features), suivis de **BatchNorm**, **ReLU** et **MaxPooling** pour reduire la dimension spatiale tout en augmentant la richesse semantique des representations. La **classification binaire** finale utilise une sortie scalaire ($\\mathbb{R}^1$) avec **Sigmoid** et une perte **BCE** (Binary Cross-Entropy).

## Pipeline du notebook

1. Installation des dependances et imports  
2. Chargement PneumoniaMNIST (train / val / test) et ponderation des classes  
3. Architecture CNN et boucle d'entrainement (early stopping)  
4. Evaluation (accuracy, precision, recall, F1, matrice de confusion, ROC-AUC)  
5. Visualisation des cartes de features (premiere couche convolutive)  
6. Analyse critique en francais"""
    )
)

# ════════════════════════════════════════════════════════
# SECTION 2 — Installation et imports
# ════════════════════════════════════════════════════════
nb.cells.append(
    nbf.v4.new_code_cell(
        """!pip install torch torchvision medmnist scikit-learn matplotlib seaborn tqdm -q"""
    )
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# --- Imports standard ---
import os  # acces au systeme de fichiers
import copy  # copies profondes du modele (early stopping)
import numpy as np  # calcul numerique et comptage des classes
import matplotlib.pyplot as plt  # visualisation des courbes et images
import seaborn as sns  # heatmaps et style des graphiques
from tqdm.auto import tqdm  # barre de progression pendant l'entrainement

# --- PyTorch ---
import torch  # tenseurs et operations GPU/CPU
import torch.nn as nn  # modules de reseau (Conv2d, Linear, etc.)
import torch.optim as optim  # optimiseurs (Adam)
from torch.utils.data import DataLoader, WeightedRandomSampler  # chargement par mini-batches

# --- MedMNIST ---
from medmnist import PneumoniaMNIST  # dataset radiologie MedMNIST

# --- Metriques sklearn ---
from sklearn.metrics import (
    accuracy_score,  # proportion de bonnes predictions
    precision_score,  # precision sur la classe positive
    recall_score,  # rappel (sensibilite) sur la pneumonie
    f1_score,  # moyenne harmonique precision/rappel
    confusion_matrix,  # tableau croise reel vs predit
    roc_curve,  # courbe ROC (FPR vs TPR)
    auc,  # aire sous la courbe ROC
    roc_auc_score,  # AUC directe a partir des probabilites
)

# --- Style des graphiques ---
sns.set_theme(style="whitegrid", palette="husl")  # theme seaborn global
plt.rcParams["figure.dpi"] = 120  # resolution des figures
plt.rcParams["font.size"] = 11  # taille de police par defaut

# --- Reproductibilite ---
SEED = 42  # graine fixe pour comparer les runs
torch.manual_seed(SEED)  # reproductibilite PyTorch
np.random.seed(SEED)  # reproductibilite NumPy

# --- Affichage device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU si disponible
print(f"Device utilise : {device}")"""
    )
)

# ════════════════════════════════════════════════════════
# SECTION 3 — Chargement des donnees
# ════════════════════════════════════════════════════════
nb.cells.append(
    nbf.v4.new_markdown_cell("""## Chargement de PneumoniaMNIST et DataLoaders""")
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# --- Transformations (MedMNIST v3+ n'expose plus mean/std dans INFO) ---
from torchvision import transforms

transform = transforms.ToTensor()  # pixels dans [0, 1]
mean = np.array([0.0])  # pour affichage inverse si normalisation ajoutee plus tard
std = np.array([1.0])

# --- Telechargement des splits train / val / test ---
train_dataset = PneumoniaMNIST(split="train", download=True, transform=transform, as_rgb=False)
val_dataset = PneumoniaMNIST(split="val", download=True, transform=transform, as_rgb=False)
test_dataset = PneumoniaMNIST(split="test", download=True, transform=transform, as_rgb=False)

print(f"Train : {len(train_dataset)} images")
print(f"Val   : {len(val_dataset)} images")
print(f"Test  : {len(test_dataset)} images")

# --- Extraction des labels d'entrainement (classe 0 = normal, 1 = pneumonie) ---
train_labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))]).astype(int).ravel()
class_counts = np.bincount(train_labels)
print(f"Effectifs par classe (train) : normal={class_counts[0]}, pneumonie={class_counts[1]}")

# --- Poids de classes pour le desequilibre (inverse de la frequence) ---
class_weights = 1.0 / class_counts.astype(np.float64)
class_weights = class_weights / class_weights.sum() * len(class_counts)
print(f"Poids de classes : {class_weights}")

# --- Sampler pondere pour equilibrer les mini-batches ---
sample_weights = class_weights[train_labels]
sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True,
)

# --- Hyperparametres DataLoader ---
BATCH_SIZE = 64
NUM_WORKERS = 0  # 0 requis sous Windows / nbconvert

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

# --- Apercu visuel de quelques radiographies ---
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
class_names = ["Normal", "Pneumonie"]
for ax, idx in zip(axes.ravel(), [0, 50, 100, 200, 300, 400, 500, 600]):
    img, label = train_dataset[idx]
    img_show = img.squeeze().numpy()
    ax.imshow(img_show, cmap="gray")
    ax.set_title(f"Label : {class_names[int(label)]}")
    ax.axis("off")
fig.suptitle("Echantillons PneumoniaMNIST (train)", fontsize=14)
plt.tight_layout()
plt.show()"""
    )
)

# ════════════════════════════════════════════════════════
# SECTION 4 — Architecture CNN
# ════════════════════════════════════════════════════════
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """## Architecture CNN

Bloc convolutif : **1 -> 32 -> 64 -> 128** canaux, avec **BatchNorm**, **ReLU**, **MaxPool**, puis **AdaptiveAvgPool2d(1)** et une couche fully-connected vers **1 sortie + Sigmoid** (classification binaire)."""
    )
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """class PneumoniaCNN(nn.Module):
    \"\"\"CNN binaire pour radiographies 28x28 (1 canal).\"\"\"

    def __init__(self):
        super().__init__()
        # --- Bloc 1 : 1 -> 32 canaux ---
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        # --- Bloc 2 : 32 -> 64 canaux ---
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        # --- Bloc 3 : 64 -> 128 canaux ---
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)

        # --- Pooling global adaptatif ---
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

        # --- Classifieur binaire ---
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # --- Bloc conv 1 ---
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        # --- Bloc conv 2 ---
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        # --- Bloc conv 3 ---
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)

        # --- Vecteur de features global ---
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        # --- Score + probabilite ---
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


# --- Instanciation et transfert sur GPU/CPU ---
model = PneumoniaCNN().to(device)
print(model)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parametres entrainables : {n_params:,}")"""
    )
)

# ════════════════════════════════════════════════════════
# SECTION 5 — Entrainement avec early stopping
# ════════════════════════════════════════════════════════
nb.cells.append(
    nbf.v4.new_markdown_cell("""## Entrainement (Adam, BCELoss, early stopping)""")
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# --- Fonctions utilitaires d'entrainement / evaluation ---
def run_epoch(model, loader, criterion, optimizer=None):
    \"\"\"Une passe forward (+ backward si optimizer fourni).\"\"\"
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for images, labels in tqdm(loader, leave=False, desc="train" if is_train else "eval"):
        images = images.to(device)
        labels = labels.float().to(device).view(-1, 1)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        all_probs.append(outputs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    probs = np.vstack(all_probs).ravel()
    labels = np.vstack(all_labels).ravel().astype(int)
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    return avg_loss, acc, probs, labels


# --- Perte, optimiseur, hyperparametres ---
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
NUM_EPOCHS = 40
PATIENCE = 7
CHECKPOINT_PATH = "best_cnn_pneumonia.pth"

# --- Early stopping ---
best_val_loss = float("inf")
patience_counter = 0
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion, optimizer=None)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
        f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"  -> Meilleur modele sauvegarde : {CHECKPOINT_PATH}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping a l'epoch {epoch} (patience={PATIENCE})")
            break

# --- Courbes d'apprentissage ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs_range = range(1, len(history["train_loss"]) + 1)
axes[0].plot(epochs_range, history["train_loss"], label="Train loss", marker="o")
axes[0].plot(epochs_range, history["val_loss"], label="Val loss", marker="s")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("BCE Loss")
axes[0].set_title("Evolution de la perte")
axes[0].legend()

axes[1].plot(epochs_range, history["train_acc"], label="Train accuracy", marker="o")
axes[1].plot(epochs_range, history["val_acc"], label="Val accuracy", marker="s")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Evolution de l'accuracy")
axes[1].legend()
plt.tight_layout()
plt.show()"""
    )
)

# ════════════════════════════════════════════════════════
# SECTION 6 — Evaluation sur le jeu de test
# ════════════════════════════════════════════════════════
nb.cells.append(
    nbf.v4.new_markdown_cell("""## Evaluation sur le jeu de test""")
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# --- Chargement du meilleur checkpoint ---
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

test_loss, test_acc, y_prob, y_true = run_epoch(model, test_loader, criterion, optimizer=None)
y_pred = (y_prob >= 0.5).astype(int)

# --- Metriques de classification ---
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_true, y_prob)

print("=" * 50)
print("RESULTATS SUR LE JEU DE TEST")
print("=" * 50)
print(f"Loss (BCE)   : {test_loss:.4f}")
print(f"Accuracy     : {test_acc:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1-Score     : {f1:.4f}")
print(f"ROC-AUC      : {roc_auc:.4f}")
print("=" * 50)

# --- Matrice de confusion ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal (pred)", "Pneumonie (pred)"],
    yticklabels=["Normal (reel)", "Pneumonie (reel)"],
)
plt.xlabel("Classe predite")
plt.ylabel("Classe reelle")
plt.title("Matrice de confusion — CNN PneumoniaMNIST (test)")
plt.tight_layout()
plt.show()

# --- Courbe ROC avec AUC reelle ---
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc_curve = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc_curve:.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Classificateur aleatoire")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.title("Courbe ROC — detection de pneumonie")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

assert abs(roc_auc - roc_auc_curve) < 1e-6, "Incoherence AUC entre sklearn et courbe ROC"
print(f"AUC verifiee (sklearn) : {roc_auc:.4f}")"""
    )
)

# ════════════════════════════════════════════════════════
# SECTION 7 — Visualisation des feature maps
# ════════════════════════════════════════════════════════
nb.cells.append(
    nbf.v4.new_markdown_cell("""## Visualisation des cartes de features (premiere couche convolutive)""")
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# --- Hook pour capturer la sortie de conv1 ---
feature_maps = []


def hook_fn(module, input, output):
    \"\"\"Callback : stocke la sortie de la couche accrochee.\"\"\"
    feature_maps.append(output.detach().cpu())


# --- Re-enregistrement du hook sur conv1 ---
handle = model.conv1.register_forward_hook(hook_fn)


def show_feature_maps(dataset, target_label, title_prefix, n_filters=16):
    \"\"\"Affiche une grille de feature maps pour une image de la classe cible.\"\"\"
    global feature_maps
    feature_maps = []
    model.eval()

    # --- Recherche d'un index correspondant a la classe ---
    idx_found = None
    for idx in range(len(dataset)):
        _, lbl = dataset[idx]
        if int(lbl) == target_label:
            idx_found = idx
            break
    assert idx_found is not None, f"Aucune image trouvee pour la classe {target_label}"

    img, lbl = dataset[idx_found]
    with torch.no_grad():
        _ = model(img.unsqueeze(0).to(device))

    fmap = feature_maps[0][0]  # (C, H, W)
    n_show = min(n_filters, fmap.shape[0])

    cols = 4
    rows = int(np.ceil(n_show / cols)) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.ravel()

    # --- Image originale ---
    img_show = img.squeeze().numpy()
    axes[0].imshow(img_show, cmap="gray")
    axes[0].set_title(f"{title_prefix} — image (label={int(lbl)})")
    axes[0].axis("off")

    for i in range(1, n_show + 1):
        axes[i].imshow(fmap[i - 1].numpy(), cmap="viridis")
        axes[i].set_title(f"Filtre {i}")
        axes[i].axis("off")

    for j in range(n_show + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Feature maps conv1 — {title_prefix}", fontsize=14)
    plt.tight_layout()
    plt.show()


# --- Normal vs Pneumonie ---
show_feature_maps(test_dataset, target_label=0, title_prefix="Cas NORMAL")
show_feature_maps(test_dataset, target_label=1, title_prefix="Cas PNEUMONIE")

# --- Retrait du hook ---
handle.remove()
print("Visualisation des feature maps terminee.")"""
    )
)

# ════════════════════════════════════════════════════════
# SECTION 8 — Analyse critique (300+ mots, francais)
# ════════════════════════════════════════════════════════
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """## Analyse critique

Le present travail illustre l'adéquation des réseaux convolutifs à l'analyse d'images médicales de faible résolution, telles que celles fournies par PneumoniaMNIST. Ce jeu de données, dérivé de radiographies thoraciques réelles mais réduites à 28×28 pixels en niveaux de gris, constitue un compromis pédagogique entre fidélité clinique et contraintes computationnelles. Le CNN mis en œuvre — trois blocs convolutifs avec normalisation par batch et pooling — parvient généralement à extraire des motifs locaux (contours, zones d'opacité) suffisants pour discriminer les cas normaux des cas pneumoniques, comme le suggèrent les métriques obtenues sur le jeu de test (accuracy, rappel, F1 et surtout l'AUC-ROC).

Plusieurs limites méritent toutefois une discussion honnête. D'abord, la résolution extrêmement basse efface des détails diagnostiques cruciaux en radiologie (lobes, bronchogramme aérien, épanchement pleural). Un radiologue humain ne trancherait jamais sur une image 28×28 ; le modèle apprend donc une version « compressée » du problème, ce qui limite la transférabilité vers la pratique hospitalière. Ensuite, MedMNIST standardise et pré-découpe les images : le pipeline ignore la segmentation anatomique, l'orientation, les artefacts d'acquisition et la variabilité inter-centres. Un déploiement clinique exigerait des données multi-institutionnelles, un contrôle qualité rigoureux et une traçabilité des prétraitements.

Le déséquilibre de classes entre patients sains et pneumoniques impose une attention particulière. L'utilisation d'un WeightedRandomSampler et le suivi du rappel (sensibilité) sont essentiels : en médecine, manquer une pneumonie (faux négatif) coûte généralement plus cher qu'une fausse alerte. La courbe ROC et l'AUC synthétisent cette capacité à ordonner les risques, mais ne remplacent pas l'étude du seuil de décision selon les coûts métier. Par ailleurs, BCELoss avec Sigmoid en sortie suppose des probabilités bien calibrées, ce qui n'est pas garanti sans calibration post-hoc (Platt, isotonic).

L'interprétabilité via les feature maps de la première couche convolutive montre que le réseau réagit à des structures de bas niveau (contours, gradients d'intensité). Cependant, les cartes profondes deviennent abstraites ; pour un usage clinique, des méthodes comme Grad-CAM ou l'audit par un expert radiologue seraient nécessaires. Enfin, l'absence de validation externe, de mesure d'incertitude (Bayésienne ou ensembles) et de comparaison avec des baselines cliniques simples constitue une faiblesse méthodologique récurrente en apprentissage profond médical.

En conclusion, ce CNN constitue une preuve de concept solide pour un cours de deep learning médical : il démontre la chaîne complète PyTorch (données, entraînement, early stopping, métriques, visualisation). Pour une recherche ou un produit de santé, il faudrait augmenter la résolution, intégrer des architectures plus récentes (ResNet, EfficientNet), enrichir l'augmentation de données, documenter les biais démographiques et soumettre le système à une évaluation réglementaire stricte. Le modèle ne remplace pas le jugement clinique ; il peut seulement assister la décision si sa fiabilité est démontrée sur des cohortes représentatives."""
    )
)

# ════════════════════════════════════════════════════════
# Metadonnees et ecriture du notebook
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

OUTPUT_PATH = "part2_cnn_pneumonia.ipynb"
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"OK: {OUTPUT_PATH} genere avec succes.")
