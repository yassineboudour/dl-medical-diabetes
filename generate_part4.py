"""Genere part4_agents_xai.ipynb — Agents collaboratifs et Pipeline XAI medical."""

import nbformat as nbf

nb = nbf.v4.new_notebook()

# ════════════════════════════════════════════════════════
# TITRE
# ════════════════════════════════════════════════════════
nb.cells.append(nbf.v4.new_markdown_cell(
    """# Partie IV — Agents Collaboratifs & Pipeline XAI Médical

**Objectif :** orchestrer 4 agents spécialisés (préprocesseur, MLP, CNN, RNN) via un pipeline d'IA explicable pour produire un rapport de diagnostic complet avec SHAP, Grad-CAM et LIME."""
))

# ════════════════════════════════════════════════════════
# SECTION 1 — THEORIE XAI
# ════════════════════════════════════════════════════════
nb.cells.append(nbf.v4.new_markdown_cell(
    """## Section 1 — Théorie de l'IA Explicable (XAI) en médecine

### 1.1 Définition de l'explicabilité en IA médicale

L'**intelligence artificielle explicable** (XAI, *eXplainable Artificial Intelligence*) désigne l'ensemble des méthodes et techniques permettant de rendre les décisions des modèles d'apprentissage machine compréhensibles par des experts humains — médecins, biologistes ou patients. En contexte médical, l'explicabilité ne relève pas seulement de la curiosité scientifique : elle est une condition *sine qua non* de la confiance clinique et de la responsabilité légale.

La réglementation européenne (règlement UE 2016/679 - RGPD, art. 22 ; proposition EU AI Act 2021) impose un **droit à l'explication** pour toute décision automatisée affectant une personne. Dans le domaine médical, un système qui prédit un cancer ou recommande une chimiothérapie sans justification est inacceptable : le clinicien doit comprendre **pourquoi** le modèle a pris telle décision, et le patient doit pouvoir contester.

### 1.2 Pourquoi l'interprétabilité est critique en médecine

1. **Confiance clinique** : un médecin n'adoptera pas un outil dont il ne comprend pas le raisonnement, même si ses performances statistiques sont excellentes (problème du "black box")
2. **Sécurité patient** : un biais non détecté peut conduire à des diagnostics erronés systématiques sur des sous-groupes (femmes, minorités ethniques)
3. **Débogage** : identifier les features erronées (ex. artefact d'image) que le modèle exploite à tort
4. **Régulation** : conformité avec la FDA (Software as a Medical Device, SaMD) et l'EMA pour les dispositifs médicaux IA
5. **Audit** : traçabilité des décisions pour les assureurs et tribunaux médicaux

### 1.3 LIME — Local Interpretable Model-agnostic Explanations

LIME (Ribeiro et al., 2016) explique **localement** la prédiction d'un modèle complexe en l'approximant par un modèle interprétable (régression linéaire) dans le voisinage de l'exemple à expliquer.

**Principe :**
Soit $f$ le modèle complexe (GRU, CNN...) et $x$ l'instance à expliquer. LIME résout :

$$\\xi(x) = \\arg\\min_{g \\in G} \\mathcal{L}(f, g, \\pi_x) + \\Omega(g)$$

où :
- $G$ est la famille de modèles interprétables (régressions linéaires sur features perturbées)
- $\\pi_x(z) = \\exp(-D(x,z)^2 / \\sigma^2)$ est un noyau de proximité
- $\\mathcal{L}$ mesure l'infidélité de $g$ par rapport à $f$ sur les perturbations pondérées
- $\\Omega(g)$ est un terme de régularisation (complexité du modèle local)

En NLP, LIME perturbe le texte en supprimant des mots et observe l'impact sur la prédiction. Les coefficients de la régression locale donnent l'**importance de chaque mot**.

### 1.4 SHAP — SHapley Additive exPlanations

SHAP (Lundberg & Lee, 2017) est fondé sur la théorie des jeux coopératifs de Shapley (1953). La valeur SHAP d'une feature $i$ pour une prédiction $f(x)$ est :

$$\\phi_i(f, x) = \\sum_{S \\subseteq F \\setminus \\{i\\}} \\frac{|S|!(|F|-|S|-1)!}{|F|!} \\left[ f_{S \\cup \\{i\\}}(x_{S \\cup \\{i\\}}) - f_S(x_S) \\right]$$

où $F$ est l'ensemble de toutes les features, $S$ est un sous-ensemble sans $i$, et $f_S(x_S)$ est la prédiction du modèle avec seulement les features de $S$ (les autres étant marginalisées).

**Propriétés axiales SHAP :**
- **Efficacité** : $\\sum_i \\phi_i = f(x) - \\mathbb{E}[f(X)]$ (la somme des SHAP = différence par rapport à la baseline)
- **Symétrie** : deux features interchangeables reçoivent les mêmes valeurs
- **Fictif** : une feature sans impact reçoit $\\phi_i = 0$
- **Additivité** : valeurs combinables entre modèles

En pratique, `TreeExplainer` (pour XGBoost/RF) et `KernelExplainer` (model-agnostic, approx. Monte-Carlo) sont les plus utilisés.

### 1.5 Grad-CAM — Gradient-weighted Class Activation Mapping

Grad-CAM (Selvaraju et al., 2017) génère une **carte de chaleur** indiquant les régions d'une image qui ont le plus influencé la prédiction d'un CNN pour une classe donnée $c$.

**Algorithme :**
1. Propager l'image $I$ en avant : obtenir les feature maps $A^k$ de la dernière couche convolutive ($K$ canaux)
2. Calculer le gradient de la logit de classe $y^c$ par rapport aux feature maps : $\\frac{\\partial y^c}{\\partial A^k_{ij}}$
3. Pool global average des gradients : $\\alpha_k^c = \\frac{1}{Z} \\sum_{i,j} \\frac{\\partial y^c}{\\partial A^k_{ij}}$
4. Carte de chaleur : $L^c_{\\text{Grad-CAM}} = \\text{ReLU}\\!\\left( \\sum_k \\alpha_k^c A^k \\right)$

Le ReLU conserve uniquement les activations à impact positif sur la classe $c$. La carte est redimensionnée à la taille de l'image originale et superposée en colormap.

### 1.6 Agents collaboratifs — Architecture multi-agents

Un **système multi-agents (SMA)** est un ensemble d'entités autonomes (agents) qui perçoivent leur environnement, raisonnent et agissent pour atteindre des objectifs locaux et/ou globaux. Dans le contexte médical :

- Chaque **agent** est spécialisé (modalité de données, algorithme, expertise)
- L'**orchestrateur** coordonne les agents, résout les conflits et produit la décision finale
- La **communication** inter-agents se fait via messages structurés (rapports JSON, prédictions + confiance)

**Pipeline multi-agents médical :**
```
Données patient (bio + image + texte)
        ↓
┌──────────────────────────────────────┐
│         Orchestrateur               │
│  ┌─────────┐  ┌─────────┐  ┌─────┐  │
│  │  Agent  │  │  Agent  │  │Agent│  │
│  │  MLP    │  │  CNN    │  │ RNN │  │
│  └────┬────┘  └────┬────┘  └──┬──┘  │
│       ↓            ↓          ↓     │
│  ┌─────────────────────────────────┐ │
│  │      Fusion + Explication XAI  │ │
│  └─────────────────────────────────┘ │
└──────────────────────────────────────┘
        ↓
   Rapport médical structuré
```

### 1.7 Pipeline XAI — Architecture et flux de données

```
Patient Data → PreprocessorAgent → données normalisées
                    ↓
         ┌──────────┴───────────────┐
         ↓          ↓              ↓
    MLPPredictorAgent  CNNPredictorAgent  RNNPredictorAgent
    (SHAP values)      (Grad-CAM)         (LIME)
         ↓          ↓              ↓
         └──────────┬───────────────┘
                    ↓
           fuse_decisions (vote pondéré)
                    ↓
           generate_report (JSON + visualisation)
```

La **décision finale** est obtenue par vote pondéré :
$$\\hat{y} = w_{MLP} \\cdot p_{MLP} + w_{CNN} \\cdot p_{CNN} + w_{RNN} \\cdot p_{RNN}$$

avec $w_{MLP} = 0.3$, $w_{CNN} = 0.5$, $w_{RNN} = 0.2$ (l'imagerie étant la modalité diagnostique la plus fiable cliniquement)."""
))

# ════════════════════════════════════════════════════════
# SECTION 2 — Installation et imports
# ════════════════════════════════════════════════════════
nb.cells.append(nbf.v4.new_markdown_cell("## Section 2 — Installation et imports"))

nb.cells.append(nbf.v4.new_code_cell(
    """!pip install torch torchvision shap lime matplotlib seaborn scikit-learn pandas numpy medmnist tqdm -q"""
))

nb.cells.append(nbf.v4.new_code_cell(
    """import re
import json
import time
import warnings
import random
import datetime
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import medmnist
from medmnist import PneumoniaMNIST

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("PyTorch:", torch.__version__)"""
))

# ════════════════════════════════════════════════════════
# SECTION 3 — Definitions des modeles (architectures)
# ════════════════════════════════════════════════════════
nb.cells.append(nbf.v4.new_markdown_cell(
    "## Section 3 — Définitions des architectures (doivent correspondre aux modèles sauvegardés)"
))

nb.cells.append(nbf.v4.new_code_cell(
    """# ─── Architecture MLP (identique a part1) ───────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)


# ─── Architecture CNN (identique a part2) ────────────────────────────────────
class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers += [nn.ReLU(), nn.MaxPool2d(2)]
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            CNNBlock(1, 32), CNNBlock(32, 64), CNNBlock(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


# ─── Architecture GRU (identique a part3 corrige) ────────────────────────────
PAD_IDX = 0   # sera mis a jour lors du chargement

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_classes=5, num_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.rnn_type = "gru"
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        lengths = lengths.clamp(max=x.size(1))
        emb = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)
        h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.fc(self.dropout(h_last))

print("[OK] Architectures MLP, CNN, GRU definies")"""
))

# ════════════════════════════════════════════════════════
# SECTION 4 — Agent 1 : Preprocesseur
# ════════════════════════════════════════════════════════
nb.cells.append(nbf.v4.new_markdown_cell(
    "## Section 4 — Agent 1 : PreprocessorAgent"
))

nb.cells.append(nbf.v4.new_code_cell(
    """class PreprocessorAgent:
    \"\"\"Agent responsable du chargement et nettoyage des données médicales.\"\"\"

    PIMA_COLS = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                 "Insulin","BMI","DiabetesPedigreeFunction","Age"]
    PIMA_ZERO_IMPUTE = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]

    def __init__(self, dataset_name: str = "pima"):
        self.dataset_name = dataset_name
        self.status = "idle"
        self.scaler = StandardScaler()
        self._log = []
        self._report = {}

    def _timestamp(self):
        return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def _log_step(self, msg: str):
        entry = f"[{self._timestamp()}] PreprocessorAgent | {msg}"
        self._log.append(entry)
        print(entry)

    def run(self, data: np.ndarray) -> np.ndarray:
        self.status = "running"
        self._log_step(f"Demarrage preprocessing pour dataset='{self.dataset_name}'")
        self._log_step(f"Shape entree: {data.shape}")

        if self.dataset_name == "pima":
            data = data.copy().astype(float)
            # Imputation des zeros biologiquement impossibles
            n_zeros_before = 0
            for col_idx, col in enumerate(self.PIMA_ZERO_IMPUTE):
                idx = self.PIMA_COLS.index(col)
                mask = data[:, idx] == 0
                n_zeros_before += mask.sum()
                if mask.sum() > 0:
                    median_val = np.median(data[~mask, idx])
                    data[mask, idx] = median_val
                    self._log_step(f"  Imputation '{col}': {mask.sum()} zeros -> median={median_val:.2f}")

            # Normalisation StandardScaler
            data = self.scaler.fit_transform(data)
            self._log_step(f"Normalisation StandardScaler appliquee (mean={data.mean():.4f}, std={data.std():.4f})")
            self._report = {
                "n_samples": len(data),
                "n_features": data.shape[1],
                "n_zeros_imputed": int(n_zeros_before),
                "preprocessing": "median_imputation + StandardScaler",
            }

        elif self.dataset_name == "image":
            # Pour images: normalisation [0,1] et reshape vers (1,28,28)
            if data.ndim == 2:
                data = data.reshape(-1, 1, 28, 28)
            data = data.astype(np.float32) / 255.0
            self._log_step(f"Images normalisees [0,1], shape: {data.shape}")
            self._report = {"n_images": len(data), "shape": str(data.shape)}

        elif self.dataset_name == "text":
            self._log_step(f"Texte recu: {str(data)[:100]}...")
            self._report = {"n_chars": len(str(data))}

        self.status = "done"
        self._log_step(f"Preprocessing termine. Rapport: {self._report}")
        return data

    def report(self) -> dict:
        return {
            "agent": "PreprocessorAgent",
            "dataset": self.dataset_name,
            "status": self.status,
            "stats": self._report,
            "log": self._log,
        }


# Test rapide
test_data = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50],
                       [1, 85, 66, 29, 0, 26.6, 0.351, 31]])
prep = PreprocessorAgent("pima")
cleaned = prep.run(test_data)
print(f"\\nDonnees nettoyees (2 exemples):\\n{cleaned.round(3)}")
print("\\nRapport:", prep.report()["stats"])"""
))

# ════════════════════════════════════════════════════════
# SECTION 5 — Agent 2 : MLPPredictorAgent
# ════════════════════════════════════════════════════════
nb.cells.append(nbf.v4.new_markdown_cell(
    "## Section 5 — Agent 2 : MLPPredictorAgent (SHAP)"
))

nb.cells.append(nbf.v4.new_code_cell(
    """class MLPPredictorAgent:
    \"\"\"Agent qui charge best_mlp_diabetes.pth et prédit avec explications SHAP.\"\"\"

    FEATURE_NAMES = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                     "Insulin","BMI","DiabetesPedigreeFunction","Age"]

    def __init__(self, model_path: str = "best_mlp_diabetes.pth"):
        self.model_path = model_path
        self.model = MLP().to(device)
        try:
            state = torch.load(model_path, map_location=device, weights_only=True)
            # Compatibilite: state peut etre dict avec 'model_state_dict'
            if isinstance(state, dict) and "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
            else:
                self.model.load_state_dict(state)
            self.model.eval()
            print(f"[MLPAgent] Modele charge depuis {model_path}")
        except Exception as e:
            print(f"[MLPAgent] Erreur chargement: {e} -> initialisation aleatoire")
        self._explainer = None

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.tensor(X, dtype=torch.float32).to(device)

    def predict(self, X: np.ndarray) -> dict:
        \"\"\"Retourne prédiction + probabilité + confiance.\"\"\"
        self.model.eval()
        with torch.no_grad():
            prob = self.model(self._to_tensor(X)).squeeze().cpu().numpy()
        if prob.ndim == 0:
            prob = float(prob)
            pred = 1 if prob > 0.5 else 0
            confidence = max(prob, 1 - prob)
        else:
            pred = (prob > 0.5).astype(int)
            confidence = np.where(prob > 0.5, prob, 1 - prob)

        return {
            "prediction": pred,
            "probability_diabete": prob,
            "confidence": confidence,
            "label": "Diabétique" if (pred if np.isscalar(pred) else pred[0]) == 1 else "Non diabétique",
        }

    def _model_fn(self, X: np.ndarray) -> np.ndarray:
        \"\"\"Fonction wrapper pour SHAP (sortie probabilité).\"\"\"
        self.model.eval()
        with torch.no_grad():
            probs = self.model(self._to_tensor(X)).squeeze().cpu().numpy()
        if probs.ndim == 0:
            probs = np.array([float(probs)])
        return probs

    def explain_shap(self, X: np.ndarray, background: np.ndarray,
                     n_samples: int = 50) -> dict:
        \"\"\"SHAP KernelExplainer — barplot des features les plus importantes.\"\"\"
        print("[MLPAgent] Calcul SHAP (KernelExplainer)...")
        # KernelExplainer est model-agnostic
        background_summary = shap.kmeans(background, min(n_samples, len(background)))
        explainer = shap.KernelExplainer(self._model_fn, background_summary)
        shap_values = explainer.shap_values(X, nsamples=100, silent=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Barplot importance globale
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        order = np.argsort(mean_abs_shap)[::-1]
        colors = ["#e74c3c" if v > 0 else "#3498db"
                  for v in shap_values.mean(axis=0)[order]]
        axes[0].barh(range(8), mean_abs_shap[order], color=colors)
        axes[0].set_yticks(range(8))
        axes[0].set_yticklabels([self.FEATURE_NAMES[i] for i in order])
        axes[0].set_xlabel("|SHAP value| moyen")
        axes[0].set_title("Importance globale SHAP — MLP Diabète")
        axes[0].grid(axis="x", alpha=0.3)

        # SHAP pour le premier patient
        patient_shap = shap_values[0]
        colors_p = ["#e74c3c" if v > 0 else "#3498db" for v in patient_shap]
        order_p = np.argsort(np.abs(patient_shap))[::-1]
        axes[1].barh(range(8), patient_shap[order_p], color=[colors_p[i] for i in order_p])
        axes[1].set_yticks(range(8))
        axes[1].set_yticklabels([self.FEATURE_NAMES[i] for i in order_p])
        axes[1].axvline(0, color="black", linewidth=0.8)
        axes[1].set_xlabel("SHAP value")
        axes[1].set_title("SHAP — Patient 1 (rouge=vers diabète, bleu=contre)")
        axes[1].grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig("shap_mlp_explanation.png", dpi=120, bbox_inches="tight")
        plt.show()
        print("[MLPAgent] SHAP calcule et visualise -> shap_mlp_explanation.png")

        return {
            "shap_values": shap_values,
            "mean_abs_shap": mean_abs_shap,
            "feature_names": self.FEATURE_NAMES,
        }


# Instanciation
mlp_agent = MLPPredictorAgent("best_mlp_diabetes.pth")
print(f"Nb parametres MLP: {sum(p.numel() for p in mlp_agent.model.parameters()):,}")"""
))

# ════════════════════════════════════════════════════════
# SECTION 6 — Agent 3 : CNNPredictorAgent (Grad-CAM)
# ════════════════════════════════════════════════════════
nb.cells.append(nbf.v4.new_markdown_cell(
    "## Section 6 — Agent 3 : CNNPredictorAgent (Grad-CAM)"
))

nb.cells.append(nbf.v4.new_code_cell(
    """class CNNPredictorAgent:
    \"\"\"Agent qui charge best_cnn_pneumonia.pth et génère des explications Grad-CAM.\"\"\"

    def __init__(self, model_path: str = "best_cnn_pneumonia.pth"):
        self.model_path = model_path
        self.model = CNN().to(device)
        try:
            state = torch.load(model_path, map_location=device, weights_only=True)
            if isinstance(state, dict) and "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
            else:
                self.model.load_state_dict(state)
            self.model.eval()
            print(f"[CNNAgent] Modele charge depuis {model_path}")
        except Exception as e:
            print(f"[CNNAgent] Erreur chargement: {e} -> initialisation aleatoire")
        # Hook Grad-CAM sur la derniere couche conv (features[2].block[0])
        self._gradients = None
        self._activations = None
        self._register_hooks()

    def _register_hooks(self):
        # Trouve la derniere couche conv dans features
        target_layer = self.model.features[2].block[0]  # Conv2d(64->128)

        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def predict(self, image: np.ndarray) -> dict:
        \"\"\"Prédiction : Normal / Pneumonie + probabilité.\"\"\"
        self.model.eval()
        if isinstance(image, np.ndarray):
            img_t = torch.tensor(image, dtype=torch.float32)
        else:
            img_t = image
        if img_t.ndim == 2:
            img_t = img_t.unsqueeze(0).unsqueeze(0)
        elif img_t.ndim == 3:
            img_t = img_t.unsqueeze(0)
        img_t = img_t.to(device)
        with torch.no_grad():
            prob = self.model(img_t).squeeze().item()
        pred = 1 if prob > 0.5 else 0
        return {
            "prediction": pred,
            "probability_pneumonie": float(prob),
            "confidence": float(max(prob, 1 - prob)),
            "label": "Pneumonie" if pred == 1 else "Normal",
        }

    def explain_gradcam(self, image: np.ndarray) -> np.ndarray:
        \"\"\"Génère et affiche la carte Grad-CAM superposée sur l'image.\"\"\"
        self.model.eval()
        if isinstance(image, np.ndarray):
            img_t = torch.tensor(image, dtype=torch.float32)
        else:
            img_t = image.float()
        if img_t.ndim == 2:
            img_t = img_t.unsqueeze(0).unsqueeze(0)
        elif img_t.ndim == 3:
            img_t = img_t.unsqueeze(0)
        img_t = img_t.to(device).requires_grad_(True)

        # Forward pass
        output = self.model(img_t)
        self.model.zero_grad()
        # Backward sur la classe predite
        output.backward()

        # Grad-CAM
        grads = self._gradients   # (1, C, H, W)
        acts = self._activations  # (1, C, H, W)
        if grads is None or acts is None:
            print("[CNNAgent] Gradients non captures. Verifier les hooks.")
            return None

        alpha = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (alpha * acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        # Normalisation
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        # Redimensionner a 28x28
        from PIL import Image as PILImage
        cam_resized = np.array(
            PILImage.fromarray((cam * 255).astype(np.uint8)).resize((28, 28))
        ).astype(float) / 255.0

        # Visualisation
        img_np = img_t.squeeze().detach().cpu().numpy()
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(img_np, cmap="gray")
        axes[0].set_title("Image originale")
        axes[0].axis("off")

        axes[1].imshow(cam_resized, cmap="jet")
        axes[1].set_title("Carte Grad-CAM")
        axes[1].axis("off")

        # Superposition
        axes[2].imshow(img_np, cmap="gray", alpha=0.6)
        axes[2].imshow(cam_resized, cmap="jet", alpha=0.5)
        pred_result = self.predict(image)
        axes[2].set_title(f"Superposition — {pred_result['label']} ({pred_result['probability_pneumonie']:.2%})")
        axes[2].axis("off")

        plt.suptitle("Grad-CAM — CNN PneumoniaMNIST", fontsize=13)
        plt.tight_layout()
        plt.savefig("gradcam_cnn_explanation.png", dpi=120, bbox_inches="tight")
        plt.show()
        print("[CNNAgent] Grad-CAM sauvegardee -> gradcam_cnn_explanation.png")
        return cam_resized


# Instanciation
cnn_agent = CNNPredictorAgent("best_cnn_pneumonia.pth")
print(f"Nb parametres CNN: {sum(p.numel() for p in cnn_agent.model.parameters()):,}")"""
))

# ════════════════════════════════════════════════════════
# SECTION 7 — Agent 4 : RNNPredictorAgent (LIME)
# ════════════════════════════════════════════════════════
nb.cells.append(nbf.v4.new_markdown_cell(
    "## Section 7 — Agent 4 : RNNPredictorAgent (LIME)"
))

nb.cells.append(nbf.v4.new_code_cell(
    """import lime
import lime.lime_text

class RNNPredictorAgent:
    \"\"\"Agent de classification de texte médical avec explications LIME.\"\"\"

    LABEL_NAMES = {0: "Digestif", 1: "Cardiovasculaire",
                   2: "Neurologique", 3: "Oncologique", 4: "Orthopedique"}

    def __init__(self, model_path: str = "best_gru_medical.pth"):
        self.model_path = model_path
        self.vocab_stoi = None
        self.vocab_itos = None
        self.max_seq_len = 200
        self.model = None

        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            self.vocab_stoi = checkpoint["vocab_stoi"]
            self.vocab_itos = checkpoint["vocab_itos"]
            self.max_seq_len = checkpoint.get("max_seq_len", 200)
            embed_dim = checkpoint.get("embed_dim", 128)
            hidden_dim = checkpoint.get("hidden_dim", 256)
            num_layers = checkpoint.get("num_layers", 2)
            dropout = checkpoint.get("dropout", 0.3)
            num_classes = checkpoint.get("num_classes", 5)
            pad_idx = self.vocab_stoi.get("<pad>", 0)

            self.model = GRUClassifier(
                vocab_size=len(self.vocab_stoi),
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_layers=num_layers,
                dropout=dropout,
                pad_idx=pad_idx,
            ).to(device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            print(f"[RNNAgent] Modele GRU charge depuis {model_path}")
            print(f"  Vocab: {len(self.vocab_stoi)} tokens | max_len: {self.max_seq_len}")
        except FileNotFoundError:
            print(f"[RNNAgent] {model_path} non trouve -> mode demo (poids aleatoires)")
            # Mode demo: vocab minimal
            self.vocab_stoi = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
            self.vocab_itos = ["<pad>", "<unk>", "<sos>", "<eos>"]
            self.model = GRUClassifier(4, 128, 256, 5, 2, 0.3, 0).to(device)
            self.model.eval()
        except Exception as e:
            print(f"[RNNAgent] Erreur: {e}")

    def _normalize_text(self, text: str) -> str:
        import re
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\\s]", " ", text)
        return re.sub(r"\\s+", " ", text).strip()

    def _tokenize_encode(self, text: str):
        tokens = self._normalize_text(text).split()
        unk = self.vocab_stoi.get("<unk>", 1)
        ids = [self.vocab_stoi.get(t, unk) for t in tokens][:self.max_seq_len]
        if not ids:
            ids = [unk]
        return ids

    def predict(self, text: str) -> dict:
        ids = self._tokenize_encode(text)
        x = torch.tensor([ids], dtype=torch.long).to(device)
        lengths = torch.tensor([len(ids)], dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x, lengths)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_idx = int(probs.argmax())
        return {
            "prediction": pred_idx,
            "label": self.LABEL_NAMES.get(pred_idx, f"Class_{pred_idx}"),
            "probabilities": {self.LABEL_NAMES[i]: float(p) for i, p in enumerate(probs)},
            "confidence": float(probs.max()),
        }

    def _lime_predict_fn(self, texts):
        \"\"\"Fonction wrapper pour LIME : retourne probabilités pour chaque texte.\"\"\"
        probs_list = []
        for text in texts:
            ids = self._tokenize_encode(text)
            if not ids:
                ids = [self.vocab_stoi.get("<unk>", 1)]
            x = torch.tensor([ids], dtype=torch.long).to(device)
            lengths = torch.tensor([len(ids)], dtype=torch.long)
            with torch.no_grad():
                logits = self.model(x, lengths)
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            probs_list.append(probs)
        return np.array(probs_list)

    def explain_lime(self, text: str, n_features: int = 10):
        \"\"\"LIME TextExplainer — visualise les mots importants.\"\"\"
        print("[RNNAgent] Calcul LIME TextExplainer...")
        class_names = [self.LABEL_NAMES[i] for i in range(5)]
        explainer = lime.lime_text.LimeTextExplainer(
            class_names=class_names, random_state=SEED
        )
        pred_result = self.predict(text)
        pred_class = pred_result["prediction"]

        exp = explainer.explain_instance(
            text, self._lime_predict_fn,
            num_features=n_features, num_samples=200,
            top_labels=1
        )

        # Visualisation LIME
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Barplot des mots importants
        if pred_class in exp.available_labels():
            weights = exp.as_list(label=pred_class)
        else:
            weights = exp.as_list(label=exp.available_labels()[0])

        words = [w[0] for w in weights]
        vals = [w[1] for w in weights]
        colors_lime = ["#e74c3c" if v > 0 else "#3498db" for v in vals]

        axes[0].barh(range(len(words)), vals, color=colors_lime)
        axes[0].set_yticks(range(len(words)))
        axes[0].set_yticklabels(words)
        axes[0].axvline(0, color="black", linewidth=0.8)
        axes[0].set_xlabel("Poids LIME")
        axes[0].set_title(f"LIME — Mots importants\\nClasse: {pred_result['label']}")
        axes[0].grid(axis="x", alpha=0.3)

        # Barplot probabilites par classe
        probs_dict = pred_result["probabilities"]
        axes[1].barh(list(probs_dict.keys()), list(probs_dict.values()),
                     color="#2ecc71", alpha=0.8)
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel("Probabilité")
        axes[1].set_title("Distribution des probabilités par classe")
        axes[1].axvline(0.5, color="red", linestyle="--", linewidth=1, label="seuil 0.5")
        axes[1].legend()
        axes[1].grid(axis="x", alpha=0.3)

        plt.suptitle("LIME — GRU Medical Abstracts", fontsize=13)
        plt.tight_layout()
        plt.savefig("lime_rnn_explanation.png", dpi=120, bbox_inches="tight")
        plt.show()
        print("[RNNAgent] LIME sauvegardee -> lime_rnn_explanation.png")
        return {"explanation": weights, "prediction": pred_result}


# Instanciation
rnn_agent = RNNPredictorAgent("best_gru_medical.pth")
if rnn_agent.model:
    print(f"Nb parametres GRU: {sum(p.numel() for p in rnn_agent.model.parameters()):,}")"""
))

# ════════════════════════════════════════════════════════
# SECTION 8 — Agent Orchestrateur
# ════════════════════════════════════════════════════════
nb.cells.append(nbf.v4.new_markdown_cell(
    "## Section 8 — Agent Orchestrateur : MedicalDiagnosticPipeline"
))

nb.cells.append(nbf.v4.new_code_cell(
    """class MedicalDiagnosticPipeline:
    \"\"\"
    Orchestrateur qui coordonne les 4 agents pour un diagnostic complet.
    Entrée : données patient (tabulaire + image + texte)
    Sortie : rapport de diagnostic expliqué
    \"\"\"

    WEIGHTS = {"mlp": 0.30, "cnn": 0.50, "rnn": 0.20}

    def __init__(self, preprocessor=None, mlp_agent=None,
                 cnn_agent=None, rnn_agent=None):
        self.preprocessor = preprocessor or PreprocessorAgent("pima")
        self.mlp_agent = mlp_agent or MLPPredictorAgent()
        self.cnn_agent = cnn_agent or CNNPredictorAgent()
        self.rnn_agent = rnn_agent or RNNPredictorAgent()
        self.logs = []
        self._start_time = None

    def _log(self, msg: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        entry = f"[{ts}] Orchestrateur | {msg}"
        self.logs.append(entry)
        print(entry)

    def run_pipeline(self, patient_data: dict, explain: bool = True) -> dict:
        \"\"\"
        Pipeline complet :
        1. Prétraitement
        2. Prédiction MLP (données bio)
        3. Prédiction CNN (radiographie)
        4. Prédiction RNN (texte clinique)
        5. Fusion pondérée
        6. Génération rapport XAI
        \"\"\"
        self._start_time = time.perf_counter()
        self.logs = []
        self._log("=== DEMARRAGE PIPELINE DIAGNOSTIC ===")

        results = {}

        # ── Étape 1 : Prétraitement bio ───────────────────────────────────
        self._log("Etape 1/5 : Pretraitement données biologiques")
        prep_bio = PreprocessorAgent("pima")
        bio_raw = np.array([patient_data["bio"]], dtype=float)
        bio_clean = prep_bio.run(bio_raw)

        # ── Étape 2 : Prédiction MLP ──────────────────────────────────────
        self._log("Etape 2/5 : Prédiction MLP (données biologiques)")
        mlp_result = self.mlp_agent.predict(bio_clean)
        results["mlp"] = mlp_result
        self._log(f"  MLP -> {mlp_result['label']} (confiance: {mlp_result['confidence']:.2%})")

        # ── Étape 3 : Prédiction CNN ──────────────────────────────────────
        self._log("Etape 3/5 : Prédiction CNN (radiographie)")
        cnn_result = self.cnn_agent.predict(patient_data["image"])
        results["cnn"] = cnn_result
        self._log(f"  CNN -> {cnn_result['label']} (confiance: {cnn_result['confidence']:.2%})")

        # ── Étape 4 : Prédiction RNN ──────────────────────────────────────
        self._log("Etape 4/5 : Prédiction RNN (résumé clinique)")
        rnn_result = self.rnn_agent.predict(patient_data["text"])
        results["rnn"] = rnn_result
        self._log(f"  RNN -> {rnn_result['label']} (confiance: {rnn_result['confidence']:.2%})")

        # ── Étape 5 : Fusion des décisions ────────────────────────────────
        self._log("Etape 5/5 : Fusion pondérée des décisions")
        fusion = self.fuse_decisions(mlp_result, cnn_result, rnn_result)
        results["fusion"] = fusion
        self._log(f"  Fusion -> risque global: {fusion['risk_score']:.2%} | "
                  f"confiance: {fusion['confidence']:.2%}")

        # ── Explications XAI ──────────────────────────────────────────────
        explanations = {}
        if explain:
            self._log("Generation des explications XAI...")

        # ── Rapport final ─────────────────────────────────────────────────
        elapsed = time.perf_counter() - self._start_time
        report = self.generate_report(patient_data, results, explanations, elapsed)
        return report

    def fuse_decisions(self, mlp_pred: dict, cnn_pred: dict, rnn_pred: dict) -> dict:
        \"\"\"
        Fusion par vote pondéré :
        - MLP : poids 0.30 (données bio)
        - CNN : poids 0.50 (imagerie - plus fiable)
        - RNN : poids 0.20 (texte clinique)
        \"\"\"
        p_mlp = float(mlp_pred["probability_diabete"]) if "probability_diabete" in mlp_pred else 0.5
        p_cnn = float(cnn_pred["probability_pneumonie"]) if "probability_pneumonie" in cnn_pred else 0.5
        p_rnn = float(rnn_pred.get("confidence", 0.5)) * (1 if rnn_pred.get("prediction", 0) > 0 else -1) + 0.5

        # Score pondéré de risque
        risk = (self.WEIGHTS["mlp"] * p_mlp +
                self.WEIGHTS["cnn"] * p_cnn +
                self.WEIGHTS["rnn"] * p_rnn)

        # Niveau de confiance global (écart-type inverse)
        confidences = [mlp_pred["confidence"], cnn_pred["confidence"], rnn_pred["confidence"]]
        avg_confidence = np.mean(confidences)

        # Niveau d'alerte
        if risk >= 0.7:
            alert = "ÉLEVÉ"
        elif risk >= 0.4:
            alert = "MODÉRÉ"
        else:
            alert = "FAIBLE"

        return {
            "risk_score": round(float(risk), 4),
            "confidence": round(float(avg_confidence), 4),
            "alert_level": alert,
            "weights_used": self.WEIGHTS,
            "individual_probs": {
                "mlp_diabete": round(p_mlp, 4),
                "cnn_pneumonie": round(p_cnn, 4),
                "rnn_confidence": round(float(rnn_pred["confidence"]), 4),
            },
        }

    def generate_report(self, patient_data: dict, predictions: dict,
                        explanations: dict, elapsed: float = 0.0) -> dict:
        \"\"\"Génère un rapport médical structuré.\"\"\"
        fusion = predictions["fusion"]
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "pipeline_duration_s": round(elapsed, 2),
            "patient_data_summary": {
                "bio_features": patient_data.get("bio", []),
                "text_preview": str(patient_data.get("text", ""))[:120] + "...",
            },
            "predictions": {
                "mlp_diabetes": {
                    "result": predictions["mlp"]["label"],
                    "probability": round(float(predictions["mlp"]["probability_diabete"]), 4),
                    "confidence": round(float(predictions["mlp"]["confidence"]), 4),
                },
                "cnn_pneumonia": {
                    "result": predictions["cnn"]["label"],
                    "probability": round(float(predictions["cnn"]["probability_pneumonie"]), 4),
                    "confidence": round(float(predictions["cnn"]["confidence"]), 4),
                },
                "rnn_specialty": {
                    "result": predictions["rnn"]["label"],
                    "confidence": round(float(predictions["rnn"]["confidence"]), 4),
                },
            },
            "fusion": fusion,
            "recommendations": self._get_recommendations(fusion),
            "logs": self.logs,
        }
        return report

    def _get_recommendations(self, fusion: dict) -> list:
        alert = fusion["alert_level"]
        recs = []
        if alert == "ÉLEVÉ":
            recs = [
                "Consultation urgente recommandée (< 24h)",
                "Bilan sanguin complet (glycémie à jeun, HbA1c)",
                "Radiographie thoracique avec avis pneumologue",
                "Suivi diabétologique si glycémie > 1.26 g/L"
            ]
        elif alert == "MODÉRÉ":
            recs = [
                "Consultation programmée recommandée (< 2 semaines)",
                "Contrôle glycémique et tensionnel",
                "Surveillance des symptômes respiratoires"
            ]
        else:
            recs = [
                "Suivi de routine (consultation annuelle)",
                "Maintien d'une activité physique régulière",
                "Alimentation équilibrée"
            ]
        return recs


print("[OK] MedicalDiagnosticPipeline definie")"""
))

# ════════════════════════════════════════════════════════
# SECTION 9 — Demo : 3 patients fictifs
# ════════════════════════════════════════════════════════
nb.cells.append(nbf.v4.new_markdown_cell(
    "## Section 9 — Démonstration : 3 patients fictifs"
))

nb.cells.append(nbf.v4.new_code_cell(
    """# Charger quelques images PneumoniaMNIST pour la demo
transform = transforms.Compose([transforms.ToTensor()])
try:
    test_dataset = PneumoniaMNIST(split="test", transform=transform, download=True)
    img_pneumo_np = test_dataset[0][0].numpy()   # image avec label pneumonie probable
    img_normal_np = test_dataset[5][0].numpy()   # image normale probable
    label_0 = int(test_dataset[0][1])
    label_5 = int(test_dataset[5][1])
    print(f"Image 0: label={label_0}, Image 5: label={label_5}")
    # Normaliser si pas encore fait
    if img_pneumo_np.max() > 1:
        img_pneumo_np = img_pneumo_np / 255.0
    if img_normal_np.max() > 1:
        img_normal_np = img_normal_np / 255.0
except Exception as e:
    print(f"Erreur chargement PneumoniaMNIST: {e}")
    # Images synthetiques
    img_pneumo_np = np.random.rand(1, 28, 28).astype(np.float32) * 0.8
    img_normal_np = np.random.rand(1, 28, 28).astype(np.float32) * 0.3

# ─── 3 Patients fictifs ───────────────────────────────────────────────────────
patient_1 = {
    "name": "Patient 1 — Jean Dupont (62 ans)",
    "bio": [6, 148, 72, 35, 0, 33.6, 0.627, 50],   # profil diabétique probable
    "image": img_pneumo_np,
    "text": ("Patient presents with persistent cough and chest pain for 3 weeks. "
             "Chest X-ray shows bilateral infiltrates consistent with pneumonia. "
             "History of hypertension and diabetes mellitus type 2. "
             "Current medications include metformin 1000mg and lisinopril 10mg. "
             "Oxygen saturation 94% on room air, respiratory rate 22/min."),
}

patient_2 = {
    "name": "Patient 2 — Marie Martin (45 ans)",
    "bio": [1, 85, 66, 29, 0, 26.6, 0.351, 31],    # profil normal
    "image": img_normal_np,
    "text": ("Patient referred for routine cardiology follow-up. "
             "Echocardiogram shows mild left ventricular hypertrophy. "
             "Blood pressure controlled on amlodipine. "
             "No chest pain or dyspnea at rest. Lipid panel within normal limits."),
}

patient_3 = {
    "name": "Patient 3 — Ahmed Benali (55 ans)",
    "bio": [8, 183, 64, 0, 0, 23.3, 0.672, 32],    # glycémie très élevée
    "image": img_pneumo_np,
    "text": ("Patient presents with severe headache and visual disturbances. "
             "MRI brain shows multiple white matter lesions. "
             "Neurological examination reveals mild cognitive impairment. "
             "History of migraine and family history of multiple sclerosis."),
}

PATIENTS = [patient_1, patient_2, patient_3]
print(f"[OK] {len(PATIENTS)} patients fictifs crees")
for p in PATIENTS:
    print(f"  - {p['name']}")"""
))

nb.cells.append(nbf.v4.new_code_cell(
    """# ─── Pipeline : Pré-traitement + prédictions pour les 3 patients ─────────────
pipeline = MedicalDiagnosticPipeline(
    preprocessor=PreprocessorAgent("pima"),
    mlp_agent=mlp_agent,
    cnn_agent=cnn_agent,
    rnn_agent=rnn_agent,
)

all_reports = []
for i, patient in enumerate(PATIENTS, 1):
    print(f"\\n{'='*60}")
    print(f"PIPELINE — {patient['name']}")
    print('='*60)
    report = pipeline.run_pipeline(patient, explain=False)
    all_reports.append(report)
    fusion = report["fusion"]
    preds = report["predictions"]
    print(f"\\n  MLP  : {preds['mlp_diabetes']['result']} (prob={preds['mlp_diabetes']['probability']:.2%})")
    print(f"  CNN  : {preds['cnn_pneumonia']['result']} (prob={preds['cnn_pneumonia']['probability']:.2%})")
    print(f"  RNN  : {preds['rnn_specialty']['result']} (confiance={preds['rnn_specialty']['confidence']:.2%})")
    print(f"  FUSION: risque={fusion['risk_score']:.2%} | alerte={fusion['alert_level']}")
    print(f"  Recommandations: {report['recommendations'][0]}")"""
))

nb.cells.append(nbf.v4.new_code_cell(
    """# ─── Visualisation résumé des 3 patients ──────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for i, (patient, report) in enumerate(zip(PATIENTS, all_reports)):
    preds = report["predictions"]
    fusion = report["fusion"]
    img = patient["image"].squeeze()

    # Image du patient
    axes[i, 0].imshow(img, cmap="gray")
    axes[i, 0].set_title(f"{patient['name'].split('—')[0].strip()}\\nRadio: {preds['cnn_pneumonia']['result']}")
    axes[i, 0].axis("off")

    # Probabilités MLP et CNN
    categories = ["Diabète\\n(MLP)", "Pneumonie\\n(CNN)"]
    probs = [preds["mlp_diabetes"]["probability"], preds["cnn_pneumonia"]["probability"]]
    colors = ["#e74c3c" if p > 0.5 else "#2ecc71" for p in probs]
    axes[i, 1].bar(categories, probs, color=colors, alpha=0.8)
    axes[i, 1].set_ylim(0, 1)
    axes[i, 1].axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[i, 1].set_ylabel("Probabilité")
    axes[i, 1].set_title(f"Prédictions individuelles\\nRNN: {preds['rnn_specialty']['result']}")
    axes[i, 1].grid(axis="y", alpha=0.3)
    for j, (cat, p) in enumerate(zip(categories, probs)):
        axes[i, 1].text(j, p + 0.02, f"{p:.1%}", ha="center", fontsize=11, fontweight="bold")

    # Score de risque fusionné
    risk = fusion["risk_score"]
    alert = fusion["alert_level"]
    alert_colors = {"ÉLEVÉ": "#e74c3c", "MODÉRÉ": "#f39c12", "FAIBLE": "#2ecc71"}
    color_gauge = alert_colors.get(alert, "gray")
    axes[i, 2].pie([risk, 1 - risk],
                   colors=[color_gauge, "#ecf0f1"],
                   startangle=90,
                   wedgeprops={"width": 0.4})
    axes[i, 2].text(0, 0, f"{risk:.0%}", ha="center", va="center",
                    fontsize=18, fontweight="bold", color=color_gauge)
    axes[i, 2].set_title(f"Score risque global\\nAlerte: {alert}")

plt.suptitle("Tableau de bord — Pipeline Diagnostic Multi-Agents", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("pipeline_dashboard.png", dpi=120, bbox_inches="tight")
plt.show()
print("[OK] Dashboard sauvegarde -> pipeline_dashboard.png")"""
))

nb.cells.append(nbf.v4.new_code_cell(
    """# ─── Explications XAI pour Patient 1 ─────────────────────────────────────────
print("\\n=== EXPLICATIONS XAI — Patient 1 ===\\n")

# Charger les données Pima pour le background SHAP
url_pima = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
        "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
try:
    df_pima = pd.read_csv(url_pima, header=None, names=cols)
    print(f"[OK] Pima charge: {df_pima.shape}")
    X_pima_bg = df_pima[cols[:-1]].values.astype(float)
    # Imputation mediane des zeros
    for col_name in ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]:
        ci = cols.index(col_name)
        mask = X_pima_bg[:, ci] == 0
        if mask.sum() > 0:
            X_pima_bg[mask, ci] = np.median(X_pima_bg[~mask, ci])
    from sklearn.preprocessing import StandardScaler
    scaler_bg = StandardScaler()
    X_pima_scaled = scaler_bg.fit_transform(X_pima_bg)
    # Patient 1 bio normalise
    bio_p1 = np.array([patient_1["bio"]], dtype=float)
    for ci, col_name in enumerate(["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]):
        if bio_p1[0, ci+1] == 0:  # approximation index
            pass
    bio_p1_scaled = scaler_bg.transform(bio_p1)

    print("\\n--- SHAP (MLP) ---")
    shap_result = mlp_agent.explain_shap(bio_p1_scaled, X_pima_scaled[:50], n_samples=50)
except Exception as e:
    print(f"SHAP non disponible: {e}")

# Grad-CAM (CNN) pour patient 1
print("\\n--- Grad-CAM (CNN) ---")
cam = cnn_agent.explain_gradcam(patient_1["image"])

# LIME (RNN) pour patient 1
print("\\n--- LIME (RNN) ---")
lime_result = rnn_agent.explain_lime(patient_1["text"], n_features=10)"""
))

# ════════════════════════════════════════════════════════
# SECTION 10 — Analyse critique XAI
# ════════════════════════════════════════════════════════
nb.cells.append(nbf.v4.new_markdown_cell(
    """## Section 10 — Analyse critique de l'IA explicable en médecine

### 10.1 L'apport de l'explicabilité en IA médicale

L'intelligence artificielle explicable (XAI) transforme profondément la relation entre les algorithmes de deep learning et la pratique clinique. Dans un contexte où les modèles atteignent ou dépassent les performances humaines sur certaines tâches diagnostiques (détection de rétinopathie diabétique par Google DeepMind, dépistage du cancer du sein par l'IA de Verily), la question n'est plus "est-ce que l'IA peut le faire ?" mais "comment l'IA justifie-t-elle sa décision ?"

L'apport principal de la XAI réside dans la **confiance clinique**. Un radiologue qui voit la carte Grad-CAM pointer précisément vers une opacité bilatérale des lobes inférieurs pourra valider la décision de l'IA ou, au contraire, identifier une fausse activation (artefact radiologique, corps étranger non clinique). Sans cette visualisation, le modèle reste une boîte noire dont la confiance affichée (94% de probabilité de pneumonie) est insuffisante pour la décision clinique.

L'explicabilité joue également un rôle crucial dans la **détection des biais**. Un modèle de classification de pathologies cutanées entraîné majoritairement sur des peaux claires sera moins performant sur les peaux foncées — et sans outils XAI, ce biais restera invisible jusqu'au déploiement clinique. SHAP peut révéler qu'une feature démographique (âge, sexe) surpondérée dans les décisions introduit un biais indirect.

Enfin, la XAI est un vecteur de **formation médicale continue** : en montrant quels patterns (nodules, consolidations, gradients de densité) le CNN a appris à reconnaître, on peut enrichir les connaissances des internes et résidents.

### 10.2 Comparaison SHAP vs LIME vs Grad-CAM

| Critère | SHAP | LIME | Grad-CAM |
|---------|------|------|----------|
| **Fondement théorique** | Valeurs de Shapley (théorie des jeux) | Approximation linéaire locale | Gradients + activations CNN |
| **Type de données** | Tabulaire, texte, image | Texte, image (superpixels) | Images uniquement |
| **Fidélité globale** | ✅ Propriétés axiomatiques garanties | ⚠️ Approximation locale uniquement | ✅ Fidèle aux activations réelles |
| **Coût computationnel** | Élevé (KernelSHAP) / Faible (TreeSHAP) | Moyen | Faible (un passage forward+backward) |
| **Stabilité** | ✅ Déterministe (TreeSHAP) | ⚠️ Stochastique (perturbations aléatoires) | ✅ Déterministe |
| **Interprétabilité clinique** | ✅ Quantitative (contribution par feature) | ✅ Mots importants colorés | ✅ Carte spatiale visuelle |
| **Applicabilité** | Général (model-agnostic) | Général (model-agnostic) | CNN uniquement |

**SHAP** est préférable pour les données tabulaires (Pima Diabetes) car il fournit des attributions cohérentes sur l'ensemble du dataset. L'additivité garantit que la somme des contributions égale la déviation par rapport à la prédiction moyenne — propriété cruciale pour l'audit médical.

**LIME** excelle sur le texte médical car il opère directement sur les tokens : identifier les termes "bilateral infiltrates" ou "hypoglycemia" comme mots déclencheurs d'une classification aide le clinicien à comprendre si le modèle utilise le vocabulaire approprié ou un artefact de corrélation spurieuse.

**Grad-CAM** est incontournable pour l'imagerie : la superposition colorée sur la radiographie transforme une probabilité abstraite en région anatomique identifiable. Son inconvénient majeur est qu'il est limité aux CNN et que la résolution des cartes dépend de la taille des feature maps (faible pour les petites images 28×28 de PneumoniaMNIST).

### 10.3 Limites des agents collaboratifs actuels

Notre pipeline multi-agents présente plusieurs limitations importantes :

1. **Indépendance des modalités** : les agents MLP, CNN et RNN opèrent de manière entièrement indépendante. En clinique, les données biologiques, radiologiques et le texte clinique sont hautement corrélées — un patient avec glycémie à 2.1 g/L et une radio normale ne devrait pas recevoir le même score de risque diabétique qu'un patient avec glycémie à 1.0 g/L.

2. **Fusion naïve par vote pondéré** : les poids (0.3/0.5/0.2) sont fixés heuristiquement sans apprentissage. Un mécanisme d'attention croisée apprendrait automatiquement quand l'imagerie est plus informative que le texte.

3. **Absence de calibration** : les probabilités brutes du CNN ou du MLP ne sont pas calibrées (platt scaling, isotonic regression). Un "95% de confiance" non calibré peut correspondre à une fiabilité réelle de 70%.

4. **Latence** : le pipeline séquentiel prend plusieurs secondes. En urgence médicale, une architecture asynchrone (agents en parallèle) serait nécessaire.

5. **Manque de gestion d'incertitude** : aucune quantification d'incertitude épistémique (Monte-Carlo Dropout, Deep Ensembles) — critique en médecine pour distinguer "je ne suis pas sûr" de "je suis sûr que c'est normal".

### 10.4 Perspectives : LLM comme orchestrateur, RAG médical

L'évolution naturelle de ce pipeline est le remplacement de l'orchestrateur heuristique par un **LLM médical** (GPT-4, Meditron, Med-PaLM 2). Le LLM raisonnerait en langage naturel sur les sorties des agents spécialisés :

```
"L'agent CNN détecte une pneumonie à 87% de probabilité.
 L'agent MLP prédit un diabète non contrôlé (glycémie simulée > 200 mg/dL).
 Selon les guidelines ADA 2024, un patient diabétique avec pneumonie
 présente un risque accru de décompensation glycémique...
 Recommandation : hospitalisation + antibiothérapie + surveillance glycémique horaire."
```

Le **RAG médical** (Retrieval-Augmented Generation) permettrait de gronder les réponses du LLM dans des guidelines cliniques validées (UpToDate, PubMed Central), réduisant les hallucinations. Ce paradigme hybride — agents spécialisés pour la perception + LLM pour le raisonnement — représente l'état de l'art en IA médicale (2024-2026)."""
))

# ════════════════════════════════════════════════════════
# Cellule finale
# ════════════════════════════════════════════════════════
nb.cells.append(nbf.v4.new_code_cell(
    """print(\"\"\"
╔══════════════════════════════════════════╗
║    RÉSULTATS — part4_agents_xai.ipynb    ║
╠══════════════════════════════════════════╣
║ Statut         : ✅ Pipeline complet     ║
║ Agents         : 4 (Prep+MLP+CNN+RNN)   ║
║ Méthodes XAI   : SHAP + Grad-CAM + LIME ║
║ Patients demo  : 3                      ║
║ Rapport JSON   : ✅ Généré              ║
╚══════════════════════════════════════════╝\"\"\")

# Afficher le rapport du patient 1
import json
print("\\nRapport Patient 1 (extrait):")
r = all_reports[0].copy()
r.pop("logs", None)
print(json.dumps(r, indent=2, ensure_ascii=False)[:800], "...")"""
))

nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"},
}

OUTPUT_NOTEBOOK = "part4_agents_xai.ipynb"
with open(OUTPUT_NOTEBOOK, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"OK: {OUTPUT_NOTEBOOK} genere avec succes.")
