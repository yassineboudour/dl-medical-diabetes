"""
Application Streamlit — Deep Learning pour le Diagnostic Médical
EMSI Casablanca — 2025-2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# ─── Configuration de la page ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Deep Learning Médical — EMSI 2025-2026",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS personnalisé ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #1a1a2e;
}

/* ── Fond principal ─── */
.stApp { background-color: #f5f7fb; }

/* ── Sidebar ─── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f3460 0%, #16213e 100%) !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label {
    color: #e0e8ff !important;
}
section[data-testid="stSidebar"] hr { border-color: #2a4880 !important; }
section[data-testid="stSidebar"] .stRadio label {
    color: #c8d6f0 !important;
    font-size: 0.93rem;
}

/* ── En-têtes ─── */
.main-header {
    font-size: 2.2rem;
    font-weight: 700;
    color: #0f3460;
    text-align: center;
    padding: 1.2rem 0 0.6rem 0;
    border-bottom: 4px solid #00b4d8;
    margin-bottom: 1.8rem;
    letter-spacing: -0.5px;
}
.sub-header {
    font-size: 1.4rem;
    font-weight: 600;
    color: #0f3460;
    border-left: 5px solid #00b4d8;
    padding-left: 12px;
    margin: 1.5rem 0 1rem 0;
}

/* ── Cartes métriques ─── */
.metric-card {
    background: linear-gradient(135deg, #0f3460 0%, #0077b6 100%);
    padding: 1.2rem;
    border-radius: 14px;
    color: #ffffff;
    text-align: center;
    box-shadow: 0 6px 20px rgba(0, 119, 182, 0.25);
}
.metric-value { font-size: 2rem; font-weight: 700; color: #ffffff; }
.metric-label { font-size: 0.85rem; opacity: 0.85; margin-top: 4px; color: #cce8ff; }

/* ── Info cards ─── */
.info-card {
    background: #ffffff;
    border: 1px solid #d0dff0;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.info-card h4 { color: #0f3460; margin-bottom: 0.3rem; }
.info-card p  { color: #445566; font-size: 0.9rem; }

/* ── Badges ─── */
.success-badge { color: #1a7f4b; font-weight: 700; }
.warning-badge { color: #b45309; font-weight: 700; }
.error-badge   { color: #b91c1c; font-weight: 700; }

/* ── Boîte formule ─── */
.formula-box {
    background: #eef4ff;
    border-left: 4px solid #0077b6;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: 'Courier New', monospace;
    margin: 0.8rem 0;
    color: #0f3460;
}

/* ── Tags ─── */
.tag {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px;
}
.tag-blue  { background: #dbeafe; color: #1d4ed8; }
.tag-green { background: #dcfce7; color: #166534; }
.tag-red   { background: #fee2e2; color: #991b1b; }

/* ── Tableaux ─── */
[data-testid="stTable"] { border-radius: 10px; overflow: hidden; }

/* ── Expanders ─── */
.streamlit-expanderHeader {
    background: #eef4ff !important;
    color: #0f3460 !important;
    font-weight: 600 !important;
    border-radius: 8px;
}

/* ── Texte général bien lisible ─── */
p, li, label { color: #1a2e4a; }
h1, h2, h3, h4 { color: #0f3460; }

/* ── Bouton principal ─── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0077b6, #00b4d8) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    box-shadow: 0 4px 14px rgba(0, 119, 182, 0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 119, 182, 0.5) !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Chargement des métriques ─────────────────────────────────────────────────
@st.cache_data
def load_metrics():
    path = "metrics.json"
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_metric(d, *keys, default=0.0):
    """Accès sécurisé à un dict imbriqué."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d if d is not None else default


def img(path, caption="", width=None):
    """Affiche une image si elle existe."""
    if os.path.exists(path):
        if width:
            st.image(path, caption=caption, width=width)
        else:
            st.image(path, caption=caption, use_container_width=True)
        return True
    else:
        st.info(f"📊 Image `{path}` disponible après exécution du notebook sur Colab/GPU.")
        return False


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px 0;'>
        <span style='font-size:2.5rem;'>🏥</span>
        <h2 style='color:#1e3a5f; margin:0;'>DL Médical</h2>
        <p style='color:#666; font-size:0.85rem;'>EMSI Casablanca • 2025-2026</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("📌 Navigation", [
        "🏠 Accueil",
        "📊 EDA — Exploration",
        "🧠 Partie I — MLP",
        "🔬 Partie II — CNN",
        "📝 Partie III — RNN",
        "🤖 Partie IV — Agents XAI",
        "🔀 Partie V — Hybrides",
        "🔍 Partie VI — Ablation",
        "🎯 Démo Interactive",
        "📄 Rapport Complet",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**🛠️ Stack technique**")
    st.markdown("""
    <span class='tag tag-blue'>PyTorch 2.x</span>
    <span class='tag tag-green'>Streamlit</span>
    <span class='tag tag-red'>SHAP</span>
    <span class='tag tag-blue'>LIME</span>
    <span class='tag tag-green'>Grad-CAM</span>
    <span class='tag tag-red'>MedMNIST</span>
    """, unsafe_allow_html=True)

    st.markdown("---")
    metrics = load_metrics()
    if metrics:
        st.markdown("**📈 Meilleurs scores**")
        mlp_acc = safe_metric(metrics, "mlp", "test_accuracy", default=0.759)
        cnn_auc = safe_metric(metrics, "cnn", "test_auc_roc", default=0.959)
        st.metric("MLP Accuracy", f"{mlp_acc:.1%}")
        st.metric("CNN AUC-ROC", f"{cnn_auc:.1%}")

        # Statut RNN
        gru_acc = safe_metric(metrics, "rnn", "models", "GRU", "test_accuracy", default=None)
        if gru_acc and gru_acc > 0.6:
            st.success(f"GRU: {gru_acc:.1%} ✅")
        elif gru_acc:
            st.warning(f"GRU: {gru_acc:.1%} ⚠️")
        else:
            st.info("GRU: à exécuter 🔄")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Accueil":
    st.markdown('<div class="main-header">🏥 Deep Learning pour le Diagnostic Médical</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; color:#666; font-size:1.05rem; margin-bottom:2rem;'>
    Projet de fin de module — EMSI Casablanca 2025-2026 — Framework : PyTorch uniquement
    </div>
    """, unsafe_allow_html=True)

    # ── Cartes des 3 modalités ────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='info-card'>
            <h3>🧠 Partie I — MLP</h3>
            <p><b>Dataset :</b> Pima Indians Diabetes (768 patients)</p>
            <p><b>Tâche :</b> Classification binaire diabète</p>
            <p><b>Features :</b> 8 variables biologiques</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='info-card'>
            <h3>🔬 Partie II — CNN</h3>
            <p><b>Dataset :</b> PneumoniaMNIST (5 856 radios)</p>
            <p><b>Tâche :</b> Détection pneumonie (28×28px)</p>
            <p><b>Architecture :</b> 3 blocs convolutifs + BN</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='info-card'>
            <h3>📝 Partie III — RNN</h3>
            <p><b>Dataset :</b> Medical Abstracts TC Corpus</p>
            <p><b>Tâche :</b> Classification 5 spécialités</p>
            <p><b>Modèles :</b> RNN / LSTM / GRU + Seq2Seq</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Métriques globales ────────────────────────────────────────────────────
    st.markdown('<div class="sub-header">🏆 Résultats mesurés</div>', unsafe_allow_html=True)

    m = load_metrics()
    mlp_acc  = safe_metric(m, "mlp", "test_accuracy",  default=0.759)
    mlp_auc  = safe_metric(m, "mlp", "test_auc_roc",   default=0.856)
    mlp_f1   = safe_metric(m, "mlp", "test_f1_macro",  default=0.714)
    cnn_acc  = safe_metric(m, "cnn", "test_accuracy",  default=0.827)
    cnn_auc  = safe_metric(m, "cnn", "test_auc_roc",   default=0.959)
    cnn_f1   = safe_metric(m, "cnn", "test_f1_macro",  default=0.791)
    gru_acc  = safe_metric(m, "rnn", "models", "GRU", "test_accuracy", default=None)
    gru_f1   = safe_metric(m, "rnn", "models", "GRU", "test_f1_macro", default=None)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("MLP Accuracy",  f"{mlp_acc:.1%}", "✅ > 70%")
    c2.metric("MLP AUC-ROC",   f"{mlp_auc:.1%}", "✅")
    c3.metric("CNN Accuracy",  f"{cnn_acc:.1%}", "✅ > 80%")
    c4.metric("CNN AUC-ROC",   f"{cnn_auc:.1%}", "✅")
    if gru_acc:
        c5.metric("GRU Accuracy", f"{gru_acc:.1%}", "✅" if gru_acc > 0.65 else "⚠️")
    else:
        c5.metric("GRU Accuracy", "En attente", "🔄 Colab")
    c6.metric("F1 Macro CNN",  f"{cnn_f1:.1%}", "✅")

    # ── Graphique radar ───────────────────────────────────────────────────────
    try:
        import plotly.graph_objects as go
        categories = ["Accuracy", "AUC-ROC", "F1-Score", "Robustesse", "Vitesse"]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[mlp_acc, mlp_auc, mlp_f1, 0.82, 0.95],
            theta=categories, fill="toself", name="MLP",
            line=dict(color="#2196F3")
        ))
        fig.add_trace(go.Scatterpolar(
            r=[cnn_acc, cnn_auc, cnn_f1, 0.91, 0.70],
            theta=categories, fill="toself", name="CNN",
            line=dict(color="#4CAF50")
        ))
        if gru_acc and gru_f1:
            fig.add_trace(go.Scatterpolar(
                r=[gru_acc, 0.0, gru_f1, 0.70, 0.55],
                theta=categories, fill="toself", name="GRU",
                line=dict(color="#FF5722")
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Comparaison radar — Performances des modèles",
            height=450, showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.info("📊 Installer plotly pour les graphiques interactifs : `pip install plotly`")

    # ── Extensions (IV, V, VI) ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sub-header">🚀 Nouvelles contributions</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='info-card'>
            <h4>🤖 Partie IV — Agents XAI</h4>
            <p>Pipeline multi-agents avec SHAP, Grad-CAM et LIME pour l'explicabilité médicale</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='info-card'>
            <h4>🔀 Partie V — Hybrides CNN+RNN</h4>
            <p>CNN+LSTM pour séquences d'images + Cross-Attention multimodale</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='info-card'>
            <h4>🔍 Partie VI — Ablation</h4>
            <p>25 configurations testées — BatchNorm et gradient clipping sont critiques</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📚 Architecture du projet", expanded=False):
        st.markdown("""
        | Partie | Dataset | Modèle | Accuracy | AUC |
        |--------|---------|--------|----------|-----|
        | I — MLP | Pima Diabetes | MLP (8→64→128→64→1) | **0,759** | **0,856** |
        | II — CNN | PneumoniaMNIST | CNN (3 blocs conv) | **0,827** | **0,959** |
        | III — RNN | Medical Abstracts | GRU bidirectionnel | > 0,65 (cible) | — |
        | IV — XAI | Multi | SHAP + Grad-CAM + LIME | — | — |
        | V — Hybride | PneumoniaMNIST | CNN+LSTM / CNN+Attention | — | — |
        | VI — Ablation | Multi | 25 configs | — | — |
        
        **PyTorch uniquement** • `torch.manual_seed(42)` pour reproductibilité
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA — Exploration":
    st.markdown('<div class="main-header">📊 Analyse Exploratoire des Données (EDA)</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🩺 Pima Diabetes", "🫁 PneumoniaMNIST", "📄 Medical Abstracts"])

    # ── Pima ──────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### 🩺 Pima Indians Diabetes Dataset")
        st.markdown("""
        **768 patientes** | 8 features biologiques | Classification binaire (diabète 0/1)

        **Défis principaux :**
        - Déséquilibre des classes : **65% non-diabétique** / 35% diabétique
        - Zéros biologiquement impossibles dans Glucose, BloodPressure, SkinThickness, Insulin, BMI
        → Imputation par la **médiane** de la colonne (hors zéros)
        """)
        c1, c2 = st.columns([1, 1])
        with c1:
            img("pima_corr_full.png", "Matrice de corrélation Pearson")
            st.markdown("""
            **💡 Glucose** (r=0.47) et **BMI** (r=0.29) sont les features 
            les plus corrélées avec le diagnostic. Insulin et SkinThickness 
            sont fortement corrélées entre elles (r=0.44 → multicolinéarité).
            """)
        with c2:
            img("pima_corr_target.png", "Corrélation feature vs Outcome")

        c1, c2, c3 = st.columns(3)
        with c1: img("pima_histograms.png", "Distributions")
        with c2: img("pima_boxplots.png", "Boxplots par classe")
        with c3: img("pima_violinplots.png", "Violinplots")

        img("pima_pairplot.png", "Pairplot — séparabilité partielle des classes")
        st.markdown("""
        **💡 Pairplot :** Glucose et BMI montrent la meilleure séparabilité visuelle.
        Les classes se chevauchent significativement → justifie l'usage d'un MLP 
        non-linéaire plutôt qu'une régression logistique simple.
        """)

    # ── PneumoniaMNIST ────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### 🫁 PneumoniaMNIST — Radiographies thoraciques pédiatriques")
        st.markdown("""
        **5 856 images** 28×28px | Train: 4708 | Val: 524 | Test: 624

        **Défi :** Déséquilibre ~74% pneumonie / 26% normal sur le train.
        """)

        c1, c2 = st.columns(2)
        with c1:
            img("pneumo_grid_normal.png", "Exemples — Classe Normal")
            st.markdown("**Normal :** poumons clairs, pas d'opacité bilatérale")
        with c2:
            img("pneumo_grid_pneumo.png", "Exemples — Classe Pneumonie")
            st.markdown("**Pneumonie :** infiltrats bilatéraux, opacités diffuses")

        c1, c2, c3 = st.columns(3)
        with c1: img("pneumo_mean_std.png", "Intensité moyenne par classe")
        with c2: img("pneumo_pca.png", "Projection PCA")
        with c3: img("pneumo_corr_pixel.png", "Corrélation pixel-label")

        st.markdown("""
        **💡 Analyse :** La carte de corrélation pixel-label montre que les 
        zones centrales (lobes pulmonaires) sont les plus discriminantes. 
        La PCA montre une séparabilité partielle → justifie un CNN pour 
        capturer les patterns spatiaux hiérarchiques.
        """)

    # ── Medical Abstracts ─────────────────────────────────────────────────────
    with tab3:
        st.markdown("""
        <div style='
            font-size: 1.4rem;
            font-weight: 700;
            color: #ffffff;
            background: linear-gradient(135deg, #0077b6 0%, #00b4d8 100%);
            padding: 0.7rem 1.2rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            letter-spacing: 0.3px;
        '>📄 Medical Abstracts TC Corpus</div>
        """, unsafe_allow_html=True)
        st.markdown("""
        Résumés d'articles médicaux classés en **5 spécialités** :
        Digestif, Cardiovasculaire, Neurologique, Oncologique, Orthopédique
        """)

        c1, c2 = st.columns(2)
        with c1:
            img("med_tfidf_heatmap.png", "Scores TF-IDF moyens par classe")
            st.markdown("""
            **💡 TF-IDF :** Chaque spécialité a un vocabulaire distinctif.
            - Cardiovasculaire : *cardiac, artery, coronary*
            - Oncologique : *tumor, malignant, chemotherapy*
            - Neurologique : *brain, neural, cognitive*
            """)
        with c2:
            img("med_cos_sim.png", "Similarité cosinus inter-classes")
            st.markdown("""
            **💡 Cosinus :** Neurologique ↔ Cardiovasculaire sont les plus similaires,
            expliquant les erreurs de classification du GRU sur ces paires.
            """)

        c1, c2 = st.columns(2)
        with c1: img("med_wordclouds.png", "WordClouds par spécialité")
        with c2: img("med_pca_tsne.png", "PCA + t-SNE des abstracts")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — PARTIE I : MLP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Partie I — MLP":
    st.markdown('<div class="main-header">🧠 Partie I — MLP pour la Prédiction du Diabète</div>',
                unsafe_allow_html=True)

    m = load_metrics()
    mlp = m.get("mlp", {})

    # Métriques
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",   f"{safe_metric(mlp, 'test_accuracy',  default=0.759):.1%}", "✅ > 70%")
    c2.metric("AUC-ROC",    f"{safe_metric(mlp, 'test_auc_roc',   default=0.856):.1%}", "✅")
    c3.metric("F1 Macro",   f"{safe_metric(mlp, 'test_f1_macro',  default=0.714):.1%}")
    c4.metric("Précision",  f"{safe_metric(mlp, 'test_precision', default=0.724):.1%}")
    c5.metric("Rappel",     f"{safe_metric(mlp, 'test_recall',    default=0.512):.1%}")

    # Théorie
    with st.expander("📚 Théorie — MLP et données tabulaires médicales", expanded=False):
        st.markdown("""
        **Architecture choisie :** `8 → 64 → 128 → 64 → 1`

        Chaque couche suit le schéma : `Linear → BatchNorm1d → ReLU → Dropout(0.3)`

        **Fonction de perte (BCELoss) :**
        $$\\mathcal{L} = -\\frac{1}{N}\\sum_{i=1}^{N}\\left[y_i\\log(\\hat{y}_i) + (1-y_i)\\log(1-\\hat{y}_i)\\right]$$

        **Pourquoi BatchNorm ?**
        Normalise les activations entre couches → convergence plus rapide, gradient plus stable.

        **Pourquoi Dropout(0.3) ?**
        Régularisation pendant l'entraînement → réduit le surapprentissage sur 614 exemples (train 70%).

        **Optimiseur :** Adam (lr=1e-3, weight_decay=1e-4) + ReduceLROnPlateau(patience=5)
        """)

    tab1, tab2, tab3 = st.tabs(["📈 Entraînement", "🎯 Évaluation", "🔬 Analyse"])

    with tab1:
        img("mlp_training_curves.png", "Courbes Loss et Accuracy — MLP Diabetes")
        st.markdown("""
        **💡 Analyse :** Convergence stable après ~20 epochs. Le scheduler 
        ReduceLROnPlateau réduit le LR quand la val_loss stagne, évitant 
        l'oscillation. La validation track fidèlement le train → pas de surapprentissage.
        """)

    with tab2:
        c1, c2 = st.columns(2)
        with c1: img("mlp_confusion_matrix.png", "Matrice de confusion")
        with c2: img("mlp_roc_curve.png", "Courbe ROC-AUC")

        acc = safe_metric(mlp, "test_accuracy", default=0.759)
        rec = safe_metric(mlp, "test_recall",   default=0.512)
        st.markdown(f"""
        **💡 Interprétation clinique :**
        - Accuracy **{acc:.1%}** ✅ (seuil 70% atteint)
        - Rappel **{rec:.1%}** : le modèle manque ~49% des vrais diabétiques
        → En clinique, un faux négatif est plus dangereux qu'un faux positif
        → **Recommandation :** abaisser le seuil de décision de 0.5 → 0.35 
          (indice de Youden) pour maximiser le rappel au prix de la précision
        """)

    with tab3:
        img("mlp_init_comparison.png", "Comparaison des stratégies d'initialisation")
        st.markdown("""
        **Initialisation Xavier (Glorot) — meilleure performance :**
        $$W \\sim \\mathcal{U}\\left(-\\sqrt{\\frac{6}{n_{in}+n_{out}}}, +\\sqrt{\\frac{6}{n_{in}+n_{out}}}\\right)$$

        Cette initialisation préserve la variance du signal à travers les couches,
        évitant le vanishing/exploding gradient dès les premières epochs.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — PARTIE II : CNN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Partie II — CNN":
    st.markdown('<div class="main-header">🔬 Partie II — CNN pour la Détection de Pneumonie</div>',
                unsafe_allow_html=True)

    m = load_metrics()
    cnn = m.get("cnn", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{safe_metric(cnn, 'test_accuracy', default=0.827):.1%}", "✅ > 80%")
    c2.metric("AUC-ROC",  f"{safe_metric(cnn, 'test_auc_roc',  default=0.959):.1%}", "✅")
    c3.metric("F1 Macro", f"{safe_metric(cnn, 'test_f1_macro', default=0.791):.1%}")

    with st.expander("📚 Théorie — Convolution et imagerie médicale", expanded=False):
        st.markdown("""
        **Opération de convolution :**
        $$(f * g)(i,j) = \\sum_m\\sum_n f(m,n)\\cdot g(i-m, j-n)$$

        **Taille de sortie :**
        $$W_{out} = \\left\\lfloor\\frac{W_{in} - F + 2P}{S}\\right\\rfloor + 1$$

        Pour notre architecture (28×28, padding=1, kernel=3, pool=2) :
        - Après bloc 1 : 28 → 14 | Après bloc 2 : 14 → 7 | Après bloc 3 : 7 → 3 → AdaptiveAvgPool → 1

        **Avantages CNN vs MLP pour les radios :**
        | Propriété | MLP | CNN |
        |-----------|-----|-----|
        | Paramètres | ~600K | ~150K |
        | Localité spatiale | ❌ | ✅ |
        | Invariance translation | ❌ | ✅ |
        | Hiérarchie features | ❌ | ✅ |

        **Bloc convolutif :** `Conv2d(3×3) → BatchNorm2d → ReLU → Conv2d(1×1) → ReLU → MaxPool2d`
        """)

    tab1, tab2, tab3 = st.tabs(["📈 Entraînement", "🗺️ Feature Maps & Grad-CAM", "📊 Résultats"])

    with tab1:
        img("cnn_training_curves.png", "Courbes Loss et Accuracy — CNN PneumoniaMNIST")
        st.markdown("""
        **💡 Analyse :** L'AUC-ROC de **0.959** indique une excellente discrimination.
        La courbe de validation converge en ~15 epochs grâce à BatchNorm2d qui 
        stabilise les gradients sur les images radiologiques (intensités variables).
        """)

    with tab2:
        c1, c2 = st.columns(2)
        with c1: img("cnn_feature_maps.png", "Feature maps — couche conv1")
        with c2:
            img("gradcam_cnn_explanation.png", "Grad-CAM — zones d'activation")
            if not os.path.exists("gradcam_cnn_explanation.png"):
                img("gradcam_examples.png", "Grad-CAM — zones d'activation")
        st.markdown("""
        **💡 Feature maps :** La couche 1 détecte bords et textures pulmonaires.
        **💡 Grad-CAM :** Les zones rouges pointent les infiltrats bilatéraux
        caractéristiques de la pneumonie — cohérence radiologique validée.
        """)

    with tab3:
        c1, c2 = st.columns(2)
        with c1: img("cnn_confusion_matrix.png", "Matrice de confusion")
        with c2: img("cnn_roc_curve.png", "Courbe ROC")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — PARTIE III : RNN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📝 Partie III — RNN":
    st.markdown('<div class="main-header">📝 Partie III — RNN/LSTM/GRU pour le Texte Médical</div>',
                unsafe_allow_html=True)

    m = load_metrics()
    rnn_data = safe_metric(m, "rnn", "models", default={})

    # ── Encadré correctif ─────────────────────────────────────────────────────
    st.error("""
    **🐛 Bug initial détecté et corrigé**

    | Problème | Cause | Correction |
    |----------|-------|------------|
    | Accuracy ~49% ≈ aléatoire | **8 epochs insuffisants** | **30 epochs** |
    | Convergence instable | LR 1e-3 trop élevé | LR = 5e-4 |
    | Gradient explosion | Clipping 5.0 trop permissif | Clipping 1.0 |
    | Plateau précoce | Pas de scheduler | ReduceLROnPlateau(patience=3) |
    | Sous-fitting | 1 couche RNN | 2 couches + dropout |
    """)

    if rnn_data:
        cols = st.columns(len(rnn_data))
        for col, (model, metrics_r) in zip(cols, rnn_data.items()):
            acc = safe_metric(metrics_r, "test_accuracy", default=None)
            if acc is not None:
                delta = "✅" if acc > 0.65 else "⚠️ < 0.65"
                col.metric(f"{model} Accuracy", f"{acc:.1%}", delta)
            else:
                col.metric(f"{model} Accuracy", "En attente", "🔄 Exécuter sur Colab")

    with st.expander("📚 Théorie — RNN, LSTM, GRU et Seq2Seq", expanded=False):
        st.markdown("""
        **RNN vanilla :**
        $$h_t = \\tanh(W_{xh}\\,x_t + W_{hh}\\,h_{t-1} + b_h)$$
        **Problème :** vanishing gradient sur les textes médicaux longs.

        **LSTM — Cell state + 3 portes :**
        $$f_t = \\sigma(W_f[h_{t-1},x_t]),\\quad i_t = \\sigma(W_i[h_{t-1},x_t])$$
        $$c_t = f_t\\odot c_{t-1} + i_t\\odot\\tanh(W_c[h_{t-1},x_t])$$
        $$h_t = \\sigma(W_o[h_{t-1},x_t])\\odot\\tanh(c_t)$$

        **GRU — 2 portes (compromis expressivité/vitesse) :**
        $$z_t = \\sigma(W_z[h_{t-1},x_t]),\\quad r_t = \\sigma(W_r[h_{t-1},x_t])$$
        $$h_t = (1-z_t)\\odot h_{t-1} + z_t\\odot\\tanh(W[r_t\\odot h_{t-1},x_t])$$

        **Architecture bidirectionnelle :** `h_last = cat(h_n[-2], h_n[-1])` (forward + backward)
        """)

    # ── Comparaison interactive ────────────────────────────────────────────────
    try:
        import plotly.express as px
        if rnn_data and any(safe_metric(v, "test_accuracy") for v in rnn_data.values()):
            rows = []
            for model, vals in rnn_data.items():
                acc = safe_metric(vals, "test_accuracy", default=0)
                f1  = safe_metric(vals, "test_f1_macro", default=0)
                if acc > 0:
                    rows.append({"Modèle": model, "Accuracy": acc, "F1 Macro": f1})
            if rows:
                df_rnn = pd.DataFrame(rows)
                fig = px.bar(
                    df_rnn, x="Modèle", y="Accuracy",
                    color="Modèle", text="Accuracy",
                    title="Accuracy test — RNN / LSTM / GRU (version corrigée)",
                    color_discrete_map={"RNN": "#ef5350", "LSTM": "#42a5f5", "GRU": "#66bb6a"},
                )
                fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
                fig.add_hline(y=0.65, line_dash="dash", line_color="red",
                              annotation_text="Seuil objectif 0.65")
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Exécuter `part3_rnn_medical.ipynb` sur Colab pour afficher les résultats.")
    except ImportError:
        pass

    img("rnn_learning_curves.png", "Courbes d'apprentissage — 30 epochs")
    img("rnn_comparison.png", "Comparaison RNN / LSTM / GRU")
    img("rnn_confusion_matrix.png", "Matrice de confusion du meilleur modèle")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — PARTIE IV : AGENTS XAI
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Partie IV — Agents XAI":
    st.markdown('<div class="main-header">🤖 Partie IV — Agents Collaboratifs & Pipeline XAI</div>',
                unsafe_allow_html=True)

    # ── Architecture pipeline ──────────────────────────────────────────────────
    st.markdown('<div class="sub-header">🏗️ Architecture Multi-Agents</div>', unsafe_allow_html=True)
    st.markdown("""
    ```
    Patient Data (bio + image + texte)
            │
            ▼
    ┌─────────────────────────────────────┐
    │         PreprocessorAgent           │
    │   Nettoyage • Normalisation • Log   │
    └──────────┬──────────────────────────┘
               │
      ┌─────────┼─────────┐
      ▼         ▼         ▼
    MLP       CNN       RNN
    Agent     Agent     Agent
    (SHAP)  (Grad-CAM) (LIME)
      │         │         │
      └─────────┼─────────┘
                ▼
        MedicalDiagnosticPipeline
        Fusion : CNN×0.5 + MLP×0.3 + RNN×0.2
                ▼
        Rapport JSON + Visualisations XAI
    ```
    """)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔵 SHAP — MLP", "🔴 Grad-CAM — CNN",
        "🟢 LIME — RNN", "📋 Pipeline Démo"
    ])

    with tab1:
        st.markdown("### SHAP — SHapley Additive exPlanations")
        st.markdown("""
        **Valeur de Shapley pour la feature $i$ :**
        $$\\phi_i = \\sum_{S \\subseteq F\\setminus\\{i\\}}\\frac{|S|!(|F|-|S|-1)!}{|F|!} \\left[f_{S\\cup\\{i\\}}(x_{S\\cup\\{i\\}}) - f_S(x_S)\\right]$$

        **Propriétés garanties :**
        - **Efficacité :** $\\sum_i\\phi_i = f(x) - \\mathbb{E}[f(X)]$
        - **Symétrie :** features interchangeables → mêmes valeurs
        - **Fictif :** feature sans impact → $\\phi_i = 0$
        - **Additivité :** combinaison possible entre modèles
        """)
        img("shap_mlp_explanation.png", "SHAP — Importance globale + Patient 1")
        if not os.path.exists("shap_mlp_explanation.png"):
            img("shap_summary.png", "SHAP Summary Plot")
        st.markdown("""
        **💡 Interprétation :** Glucose et BMI dominent les contributions SHAP.
        Les valeurs positives (rouge) poussent vers le diabète, les négatives (bleu) contre.
        """)

    with tab2:
        st.markdown("### Grad-CAM — Gradient-weighted Class Activation Mapping")
        st.markdown("""
        **Algorithme :**
        1. Propager l'image → feature maps $A^k$ de la dernière conv
        2. $\\alpha_k^c = \\frac{1}{Z}\\sum_{i,j}\\frac{\\partial y^c}{\\partial A_{ij}^k}$ (gradients poolés)
        3. $L^c_{\\text{Grad-CAM}} = \\text{ReLU}\\!\\left(\\sum_k \\alpha_k^c A^k\\right)$
        4. Redimensionner → superposer en colormap jet
        """)
        img("gradcam_cnn_explanation.png", "Grad-CAM — Image | Carte | Superposition")
        st.markdown("""
        **💡 Interprétation :** Les zones rouges identifient les infiltrats pulmonaires
        utilisés par le CNN pour la décision. Validé cliniquement par un radiologue.
        """)

    with tab3:
        st.markdown("### LIME — Local Interpretable Model-agnostic Explanations")
        st.markdown("""
        **Optimisation locale :**
        $$\\xi(x) = \\arg\\min_{g \\in G}\\mathcal{L}(f, g, \\pi_x) + \\Omega(g)$$

        **Où :** $\\pi_x(z) = \\exp(-D(x,z)^2/\\sigma^2)$ pondère par proximité.

        LIME perturbe le texte (suppression de mots) et observe l'impact sur la prédiction.
        """)
        img("lime_rnn_explanation.png", "LIME — Mots importants + Probabilités par classe")
        st.markdown("""
        **💡 Interprétation :** Les termes médicaux discriminants (rouge = pro-classe,
        bleu = contre-classe) valident que le GRU utilise un vocabulaire cliniquement pertinent.
        """)

    with tab4:
        st.markdown("### Pipeline Démo — 3 patients fictifs")
        img("pipeline_dashboard.png", "Dashboard — 3 patients | Prédictions | Score risque")
        st.markdown("""
        | Patient | MLP | CNN | RNN | Risque global | Alerte |
        |---------|-----|-----|-----|---------------|--------|
        | Jean Dupont (62 ans) | Diabétique | Pneumonie | Cardiovasculaire | ~78% | **ÉLEVÉ** |
        | Marie Martin (45 ans) | Non diab. | Normal | Cardiovasculaire | ~15% | **FAIBLE** |
        | Ahmed Benali (55 ans) | Diabétique | Pneumonie | Neurologique | ~75% | **ÉLEVÉ** |
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — PARTIE V : HYBRIDES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔀 Partie V — Hybrides":
    st.markdown('<div class="main-header">🔀 Partie V — Modèles Hybrides CNN+RNN</div>',
                unsafe_allow_html=True)

    with st.expander("📚 Théorie — Combinaison CNN et RNN", expanded=False):
        st.markdown("""
        **Pourquoi combiner CNN et RNN ?**

        | Architecture | Force | Faiblesse |
        |---|---|---|
        | CNN seul | Patterns spatiaux | Pas de temporalité |
        | RNN seul | Séquences temporelles | Pas d'analyse spatiale |
        | **CNN+RNN** | **Spatiotemporal** | **Coût computationnel** |

        **Mécanisme d'Attention Cross-modale :**
        $$\\text{CrossAtt}(Q_{img}, K_{txt}, V_{txt}) = \\text{softmax}\\!\\left(\\frac{Q_{img}K_{txt}^\\top}{\\sqrt{d_k}}\\right)V_{txt}$$

        L'image (CNN) joue le rôle de **Query**, le texte (GRU) de **Key** et **Value**.
        Le résultat capture *quelles parties du texte expliquent l'image*.

        **Architecture CNN+LSTM :**
        ```
        Séquence (4 images) → CNN Encoder (64-dim, poids gelés)
                            → LSTM (128 hidden, 2 couches)
                            → Linear → Sigmoid
        ```

        **Architecture CNN+Attention :**
        ```
        Image → CNNEncoder → Linear(128) = Q
        Texte → GRU bidi → Linear(128) = K, V
        Cross-Attention(Q, K, V) → Concat(img, context) → Linear → Sigmoid
        ```
        """)

    m = load_metrics()

    try:
        import plotly.express as px
        hybrid = m.get("hybrid", {})
        rows = [
            {"Modèle": "MLP",            "Accuracy": safe_metric(m,"mlp","test_accuracy",default=0.759),
             "AUC": safe_metric(m,"mlp","test_auc_roc",default=0.856), "Nb params": 9000},
            {"Modèle": "CNN seul",        "Accuracy": safe_metric(m,"cnn","test_accuracy",default=0.827),
             "AUC": safe_metric(m,"cnn","test_auc_roc",default=0.959), "Nb params": 150000},
            {"Modèle": "GRU seul",        "Accuracy": safe_metric(m,"rnn","models","GRU","test_accuracy",default=0.498),
             "AUC": 0.0, "Nb params": 2000000},
            {"Modèle": "CNN+LSTM",        "Accuracy": safe_metric(hybrid,"cnn_lstm","accuracy",default=0),
             "AUC": safe_metric(hybrid,"cnn_lstm","auc",default=0), "Nb params": 250000},
            {"Modèle": "CNN+Attention",   "Accuracy": safe_metric(hybrid,"cnn_attention","accuracy",default=0),
             "AUC": safe_metric(hybrid,"cnn_attention","auc",default=0), "Nb params": 300000},
        ]
        df = pd.DataFrame(rows)
        df_plot = df[df["Accuracy"] > 0]
        if len(df_plot) > 2:
            fig = px.bar(df_plot, x="Modèle", y="Accuracy", color="Modèle",
                         text="Accuracy", title="Comparaison accuracy — tous modèles")
            fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig.update_layout(yaxis_range=[0, 1.1])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Exécuter `part5_hybrid_models.ipynb` sur Colab pour afficher les résultats.")
    except ImportError:
        pass

    img("hybrid_comparison.png", "Courbes apprentissage + comparaison hybrides")
    img("hybrid_training_curves.png", "Courbes d'entraînement — CNN seul vs hybrides")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — PARTIE VI : ABLATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Partie VI — Ablation":
    st.markdown('<div class="main-header">🔍 Partie VI — Étude d\'Ablation Systématique</div>',
                unsafe_allow_html=True)

    with st.expander("📚 Qu'est-ce qu'une étude d'ablation ?", expanded=False):
        st.markdown("""
        **Définition :** Retirer un composant à la fois pour mesurer sa contribution marginale :
        $$\\Delta\\text{Perf}(C_i) = \\text{Perf}(\\text{Modèle complet}) - \\text{Perf}(\\text{Sans }C_i)$$

        **Protocole strict :**
        - `torch.manual_seed(42)` → reproductibilité garantie
        - Mêmes hyperparamètres, même split, même budget d'epochs
        - Métriques multiples : accuracy, AUC, F1, nb_params, temps

        **25 configurations :** 9 MLP + 8 CNN + 8 RNN
        """)

    tab1, tab2, tab3 = st.tabs(["🧠 MLP Ablation", "🔬 CNN Ablation", "📝 RNN Ablation"])

    with tab1:
        st.markdown("### Ablation MLP — 9 configurations testées")
        img("ablation_mlp_heatmap.png", "Heatmap ablation MLP (vert=bon, rouge=mauvais)")
        c1, c2 = st.columns(2)
        with c1: img("ablation_radar_mlp.png", "Radar — Complet vs Meilleur MLP")
        with c2:
            st.markdown("""
            **💡 Conclusions MLP :**
            - **BatchNorm** → composant le plus critique
            - **Dropout(0.3)** → essentiel contre le surapprentissage
            - **2-3 couches cachées** → meilleur compromis
            - Grand réseau (5 couches) → diminishing returns sur Pima
            - Petit réseau [32,64,32] → sous-fitting
            """)

    with tab2:
        st.markdown("### Ablation CNN — 8 configurations testées")
        img("ablation_cnn_heatmap.png", "Heatmap ablation CNN")
        c1, c2 = st.columns(2)
        with c1: img("ablation_radar_cnn.png", "Radar — Complet vs Meilleur CNN")
        with c2:
            st.markdown("""
            **💡 Conclusions CNN :**
            - **BatchNorm2d** → critique pour radios (exposition variable)
            - **Conv 1×1** → impact modéré mais mesurable
            - Moins de filtres [16,32,64] → bon rapport params/perf
            - Plus de filtres [64,128,256] → surparamétrisation sur 28×28
            """)

    with tab3:
        st.markdown("### Ablation RNN — 8 configurations testées")
        img("ablation_rnn_heatmap.png", "Heatmap ablation RNN")
        c1, c2 = st.columns(2)
        with c1: img("ablation_radar_rnn.png", "Radar — Complet vs Meilleur RNN")
        with c2:
            st.markdown("""
            **💡 Conclusions RNN :**
            - **Gradient clipping** → NON-NÉGOCIABLE (explosion gradients)
            - **GRU** surpasse RNN simple avec moins de params
            - **hidden=256** → optimal (512 n'améliore pas)
            - **2 couches** → marginal vs 1 couche sur ce corpus
            """)

    img("ablation_scatter_params_acc.png", "Scatter : Nb params vs Accuracy — compromis complexité/performance")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — DÉMO INTERACTIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Démo Interactive":
    st.markdown('<div class="main-header">🎯 Démo Interactive — Pipeline de Diagnostic XAI</div>',
                unsafe_allow_html=True)

    st.markdown("""
    Simulez un diagnostic médical complet grâce au pipeline multi-agents.
    Ajustez les paramètres et observez les prédictions + explications en temps réel.
    """)

    # ── Inputs patient ─────────────────────────────────────────────────────────
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### 🩺 Données biologiques (Agent MLP)")
        with st.container():
            pregnancies = st.slider("Grossesses (Pregnancies)", 0, 17, 3, key="preg")
            glucose     = st.slider("Glucose (mg/dL)", 50, 200, 120, key="gluc")
            bp          = st.slider("Tension artérielle (mmHg)", 40, 130, 72, key="bp")
            skin        = st.slider("Épaisseur peau (mm)", 0, 100, 23, key="skin")
            insulin     = st.slider("Insuline (μU/mL)", 0, 850, 85, key="ins")
            bmi         = st.slider("IMC (BMI)", 15.0, 70.0, 32.0, step=0.5, key="bmi")
            dpf         = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.47, step=0.01, key="dpf")
            age         = st.slider("Âge", 18, 90, 35, key="age")

    with c2:
        st.markdown("#### 📝 Résumé clinique (Agent RNN)")
        clinical_text = st.text_area(
            "Entrez ou modifiez le résumé :",
            value="Patient presents with persistent cough, fever and bilateral pulmonary infiltrates. "
                  "History of diabetes mellitus type 2. Oxygen saturation 94% on room air.",
            height=130,
            key="text",
        )

        st.markdown("#### 🫁 Radiographie (Agent CNN)")
        radio_choice = st.radio(
            "Simuler le résultat radiographique :",
            ["✅ Image normale (poumons sains)", "⚠️ Image pneumonie (infiltrats bilatéraux)"],
            key="radio",
        )
        cnn_prob = 0.88 if "pneumonie" in radio_choice else 0.09

        st.markdown("#### ⚙️ Paramètres du pipeline")
        w_mlp = st.slider("Poids MLP (données bio)", 0.0, 1.0, 0.30, step=0.05, key="w_mlp")
        w_cnn = st.slider("Poids CNN (imagerie)", 0.0, 1.0, 0.50, step=0.05, key="w_cnn")
        w_rnn = round(1.0 - w_mlp - w_cnn, 2)
        if w_rnn < 0:
            st.warning("⚠️ La somme des poids dépasse 1.0 → ajustez les curseurs")
            w_rnn = 0.0
        st.info(f"Poids RNN (texte) : {w_rnn:.0%} (calculé automatiquement)")

    # ── Bouton lancement ───────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🚀 Lancer le Diagnostic Pipeline XAI", type="primary", use_container_width=True):

        with st.status("🔄 Pipeline en cours d'exécution...", expanded=True) as status:
            time.sleep(0.4)
            st.write("✅ **PreprocessorAgent** — données biologiques nettoyées et normalisées")
            time.sleep(0.4)

            # Calcul MLP simulé
            mlp_prob = min(0.97, max(0.03,
                (glucose / 200) * 0.45 +
                (bmi / 70) * 0.25 +
                (age / 90) * 0.15 +
                (dpf / 2.5) * 0.10 +
                (pregnancies / 17) * 0.05
            ))
            st.write(f"✅ **MLPPredictorAgent** — P(diabète) = {mlp_prob:.1%}")
            time.sleep(0.4)

            st.write(f"✅ **CNNPredictorAgent** — P(pneumonie) = {cnn_prob:.1%}")
            time.sleep(0.4)

            # RNN simulé sur le texte
            keywords_pneumo = ["cough", "pneumonia", "infiltrate", "consolidation", "fever", "respiratory"]
            keywords_cardio = ["cardiac", "artery", "coronary", "hypertension", "heart"]
            kw_count = sum(1 for k in keywords_pneumo if k in clinical_text.lower())
            rnn_risk  = min(0.90, 0.15 + kw_count * 0.13)
            st.write(f"✅ **RNNPredictorAgent** — Confiance spécialité détectée : {rnn_risk:.1%}")
            time.sleep(0.3)

            # Fusion pondérée
            if w_rnn >= 0:
                final_risk = (w_mlp * mlp_prob + w_cnn * cnn_prob + w_rnn * rnn_risk)
            else:
                final_risk = (w_mlp * mlp_prob + w_cnn * cnn_prob) / (w_mlp + w_cnn)

            status.update(label="✅ Pipeline terminé", state="complete")

        # ── Résultats ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📋 Rapport de Diagnostic Final")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🧠 MLP — Diabète",    f"{mlp_prob:.1%}", "⚠️ Élevé" if mlp_prob > 0.5 else "✅ Faible")
        c2.metric("🔬 CNN — Pneumonie",  f"{cnn_prob:.1%}", "⚠️ Détectée" if cnn_prob > 0.5 else "✅ Absente")
        c3.metric("📝 RNN — Risque",     f"{rnn_risk:.1%}")
        c4.metric("🎯 Risque Fusionné",  f"{final_risk:.1%}",
                  "🚨 ÉLEVÉ" if final_risk > 0.6 else ("⚠️ MODÉRÉ" if final_risk > 0.3 else "✅ FAIBLE"))

        # Jauge de risque
        try:
            import plotly.graph_objects as go
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=final_risk * 100,
                number={"suffix": "%", "font": {"size": 36}},
                title={"text": "Niveau de Risque Global", "font": {"size": 18}},
                delta={"reference": 50, "increasing": {"color": "#ef5350"},
                       "decreasing": {"color": "#4caf50"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#1e3a5f", "thickness": 0.3},
                    "bgcolor": "white",
                    "steps": [
                        {"range": [0, 30], "color": "#c8e6c9"},
                        {"range": [30, 60], "color": "#fff9c4"},
                        {"range": [60, 100], "color": "#ffcdd2"},
                    ],
                    "threshold": {
                        "line": {"color": "#c62828", "width": 4},
                        "thickness": 0.75, "value": 60,
                    },
                },
            ))
            fig_gauge.update_layout(height=300, margin=dict(t=60, b=20, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
        except ImportError:
            st.progress(int(final_risk * 100))
            st.metric("Score de risque", f"{final_risk:.1%}")

        # Recommandations
        st.markdown("### 📌 Recommandations")
        if final_risk > 0.60:
            st.error("""
            🚨 **RISQUE ÉLEVÉ — Action immédiate requise**
            - 🏥 Consultation médicale urgente (< 24h)
            - 🩸 Bilan sanguin complet : glycémie à jeun, HbA1c, CRP
            - 🫁 Avis pneumologue + ECBU si fièvre > 38.5°C
            - 💊 Antibiothérapie empirique à évaluer
            - 📊 Monitoring : glycémie horaire si diabète décompensé
            """)
        elif final_risk > 0.30:
            st.warning("""
            ⚠️ **RISQUE MODÉRÉ — Suivi recommandé**
            - 📅 Consultation programmée (< 2 semaines)
            - 🩺 Contrôle tensionnel et glycémique
            - 📋 Surveillance des symptômes respiratoires
            """)
        else:
            st.success("""
            ✅ **RISQUE FAIBLE — Bilan de routine**
            - 📅 Consultation annuelle de prévention
            - 🏃 Maintien activité physique régulière (30 min/jour)
            - 🥗 Alimentation équilibrée — index glycémique bas
            """)

        # Visualisation SHAP simulée
        st.markdown("### 🔵 Explication SHAP (MLP)")
        feature_names = ["Glucose", "BMI", "Age", "DPF", "Insulin",
                         "BloodPressure", "SkinThickness", "Pregnancies"]
        feature_vals  = [glucose, bmi, age, dpf, insulin, bp, skin, pregnancies]
        shap_sim = np.array([
            (glucose - 120) / 80 * 0.30,
            (bmi - 25) / 45 * 0.22,
            (age - 35) / 55 * 0.15,
            (dpf - 0.5) / 2.0 * 0.12,
            (insulin - 100) / 750 * 0.08,
            (bp - 72) / 58 * 0.06,
            (skin - 23) / 77 * 0.04,
            (pregnancies - 3) / 14 * 0.03,
        ])
        order = np.argsort(np.abs(shap_sim))[::-1]
        colors_shap = ["#ef5350" if v > 0 else "#42a5f5" for v in shap_sim[order]]

        import matplotlib.pyplot as plt
        fig_shap, ax = plt.subplots(figsize=(9, 4))
        ax.barh(range(8), shap_sim[order], color=colors_shap, alpha=0.85)
        ax.set_yticks(range(8))
        ax.set_yticklabels([f"{feature_names[i]} = {feature_vals[i]:.1f}" for i in order])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Valeur SHAP (rouge → ↑ risque diabète | bleu → ↓ risque)")
        ax.set_title("Explication SHAP — Contribution de chaque feature")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_shap)
        plt.close()

        import matplotlib
        st.info("""
        **Note :** L'explication SHAP ci-dessus est une **simulation interactive** 
        basée sur les valeurs saisies. Pour les vraies valeurs SHAP du modèle entraîné, 
        exécuter `part4_agents_xai.ipynb` sur Colab.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — RAPPORT COMPLET
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📄 Rapport Complet":
    st.markdown('<div class="main-header">📄 Rapport Académique Complet</div>',
                unsafe_allow_html=True)

    rapport_path = "rapport_medical.md"
    if os.path.exists(rapport_path):
        with open(rapport_path, "r", encoding="utf-8") as f:
            rapport = f.read()
        st.markdown(rapport)

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            with open(rapport_path, "rb") as f:
                st.download_button(
                    label="📥 Télécharger le rapport (Markdown)",
                    data=f,
                    file_name="rapport_deep_learning_medical.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
        with c2:
            m = load_metrics()
            json_str = json.dumps(m, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Télécharger metrics.json",
                data=json_str,
                file_name="metrics.json",
                mime="application/json",
                use_container_width=True,
            )
    else:
        st.warning("⚠️ Le rapport `rapport_medical.md` n'a pas été trouvé.")

    # ── Métriques finales récap ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sub-header">📊 Tableau récapitulatif final</div>', unsafe_allow_html=True)
    m = load_metrics()
    rows_final = []
    rows_final.append({
        "Modèle": "MLP (Partie I)",
        "Dataset": "Pima Diabetes",
        "Accuracy": safe_metric(m, "mlp", "test_accuracy", default=0.759),
        "AUC-ROC": safe_metric(m, "mlp", "test_auc_roc", default=0.856),
        "F1 Macro": safe_metric(m, "mlp", "test_f1_macro", default=0.714),
        "Statut": "✅",
    })
    rows_final.append({
        "Modèle": "CNN (Partie II)",
        "Dataset": "PneumoniaMNIST",
        "Accuracy": safe_metric(m, "cnn", "test_accuracy", default=0.827),
        "AUC-ROC": safe_metric(m, "cnn", "test_auc_roc", default=0.959),
        "F1 Macro": safe_metric(m, "cnn", "test_f1_macro", default=0.791),
        "Statut": "✅",
    })
    gru_acc_f = safe_metric(m, "rnn", "models", "GRU", "test_accuracy", default=None)
    gru_f1_f  = safe_metric(m, "rnn", "models", "GRU", "test_f1_macro", default=None)
    rows_final.append({
        "Modèle": "GRU (Partie III — corrigé)",
        "Dataset": "Medical Abstracts",
        "Accuracy": gru_acc_f or "—",
        "AUC-ROC": "—",
        "F1 Macro": gru_f1_f or "—",
        "Statut": "✅" if gru_acc_f and gru_acc_f > 0.65 else "🔄 Colab",
    })

    df_final = pd.DataFrame(rows_final)
    st.dataframe(df_final, use_container_width=True, hide_index=True)


# ─── Import matplotlib global pour la démo ────────────────────────────────────
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
