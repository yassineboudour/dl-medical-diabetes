Tu es un expert PyTorch. Génère un notebook Jupyter complet et exécutable 
sur Google Colab pour la Partie I d'un projet Deep Learning académique.

THÈME : Prédiction du diabète — Pima Indians Diabetes Dataset

STRUCTURE DU NOTEBOOK (cellules Markdown + Code) :

## Section 1 — Théorie (cellules Markdown)
- nn.Module, forward(), __init__, paramètres, gradient
- state_dict, device, propagation avant et rétropropagation
- Fonctions d'activation : ReLU, Sigmoid, Tanh (formules LaTeX)
- BCELoss et BCEWithLogitsLoss (formules LaTeX)
- Justification du choix pour classification binaire médicale

## Section 2 — Installation et imports (cellule Code)
!pip install imbalanced-learn scikit-learn pandas numpy matplotlib seaborn torch

## Section 3 — Chargement et préparation des données (cellule Code)
- Charger Pima Diabetes depuis :
  url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
- Colonnes : ['Pregnancies','Glucose','BloodPressure','SkinThickness',
              'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
- Remplacer les 0 aberrants dans Glucose, BloodPressure, SkinThickness, 
  Insulin, BMI par la médiane de chaque colonne
- Analyser la distribution des classes
- StandardScaler pour normalisation
- Split stratifié : train(70%) / val(15%) / test(15%)
- DataLoader PyTorch : batch_size=32, shuffle=True

## Section 4 — Version MLP avec nn.Sequential (cellule Code)
Architecture : 8 → 64 → 128 → 64 → 1
- BatchNorm1d + ReLU + Dropout(0.3) après chaque couche cachée
- Sigmoid en sortie

## Section 5 — Version MLP avec classe personnalisée (cellule Code)
Même architecture mais héritant de nn.Module.
Ajouter get_activations() et summary().

## Section 6 — Inspection des paramètres (cellule Code)
- named_parameters() : afficher nom + shape
- state_dict() : afficher toutes les clés
- Compter le total des paramètres entraînables

## Section 7 — Trois initialisations (cellule Code)
Pour chaque stratégie (Gaussienne, Constante, Xavier) :
- Réinitialiser le modèle
- Entraîner 30 epochs
- Tracer la courbe de loss train/val
- Afficher l'accuracy finale sur validation
Afficher les 3 courbes sur le même graphe pour comparaison.

## Section 8 — Boucle d'entraînement complète (cellule Code)
- Optimiseur : Adam lr=0.001, weight_decay=1e-4
- Scheduler : ReduceLROnPlateau patience=5, factor=0.5
- Early stopping patience=10
- Sauvegarder : torch.save(model.state_dict(), 'best_mlp_diabetes.pth')
- Recharger le meilleur modèle pour évaluation

## Section 9 — Gestion CPU/GPU (cellule Code)
- device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
- Déplacer modèle et tenseurs, vérifier avec assertions

## Section 10 — Évaluation et métriques (cellule Code)
- Accuracy, Precision, Recall, F1-score (macro + weighted)
- Matrice de confusion (Seaborn heatmap)
- ROC-AUC curve
- Courbes Loss et Accuracy train/val
- Seuil optimal via Youden Index

## Section 11 — Analyse critique (cellule Markdown)
Minimum 400 mots en français sur :
- Sequential vs classe personnalisée
- Impact des 3 initialisations
- Limites du MLP pour les données médicales déséquilibrées

## Section 12 — Question de synthèse (cellule Markdown)
Minimum 500 mots en français, style académique, répondant à :
"Dans quelle mesure un MLP bien paramétré constitue-t-il une solution 
pertinente pour la classification tabulaire sur un dataset réel, et quelles 
sont ses principales limites au regard de la structure statistique des données ?"
Ancre la réponse sur le dataset Pima Diabetes.

CONTRAINTES :
- PyTorch uniquement
- Chaque cellule code commentée ligne par ligne
- Formules mathématiques en LaTeX dans les cellules Markdown
- Chaque graphique avec titre, axes et légende
- Compatible Google Colab sans modification