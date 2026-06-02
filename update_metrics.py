"""
update_metrics.py
Lit les sorties des notebooks executes et met a jour metrics.json.
A lancer apres chaque notebook :
    py update_metrics.py --part 3
    py update_metrics.py --part 5
    py update_metrics.py --part 6
    py update_metrics.py --all
"""

import argparse
import json
import os
import re

METRICS_PATH = "metrics.json"


def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_metrics(m):
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    print(f"[OK] {METRICS_PATH} mis a jour")


def extract_from_notebook(nb_path: str):
    """
    Extrait les valeurs numeriques des cellules de sortie du notebook.
    Cherche des patterns de type 'acc=0.7234' ou 'AUC: 0.893'.
    """
    if not os.path.exists(nb_path):
        print(f"[WARN] {nb_path} introuvable")
        return {}

    import nbformat
    nb = nbformat.read(nb_path, as_version=4)

    results = {}
    current_model = None

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        outputs = cell.get("outputs", [])
        for output in outputs:
            text = ""
            if output.get("output_type") == "stream":
                text = "".join(output.get("text", []))
            elif output.get("output_type") in ("execute_result", "display_data"):
                text = "".join(output.get("text/plain", []))

            if not text:
                continue

            # Detecter le modele courant
            for model_name in ["RNN", "LSTM", "GRU", "CNN+LSTM", "CNN+Attention", "CNN_seul"]:
                if model_name in text:
                    current_model = model_name

            # Extraire test_acc et test_auc
            acc_match = re.search(r"acc[uracy]*[=:]\s*([0-9]\.[0-9]+)", text, re.IGNORECASE)
            auc_match = re.search(r"auc[_\-roc]*[=:]\s*([0-9]\.[0-9]+)", text, re.IGNORECASE)
            f1_match  = re.search(r"f1[_macro]*[=:]\s*([0-9]\.[0-9]+)", text, re.IGNORECASE)

            if current_model and acc_match:
                if current_model not in results:
                    results[current_model] = {}
                results[current_model]["test_accuracy"] = float(acc_match.group(1))
            if current_model and auc_match:
                if current_model not in results:
                    results[current_model] = {}
                results[current_model]["auc"] = float(auc_match.group(1))
            if current_model and f1_match:
                if current_model not in results:
                    results[current_model] = {}
                results[current_model]["test_f1_macro"] = float(f1_match.group(1))

    return results


def update_part3():
    print("\n=== Mise a jour Part III (RNN/LSTM/GRU) ===")
    nb_results = extract_from_notebook("part3_rnn_medical.ipynb")
    m = load_metrics()

    if "rnn" not in m:
        m["rnn"] = {"models": {}, "best_model": "GRU"}

    for model_name, vals in nb_results.items():
        key = model_name.upper()
        if key in ["RNN", "LSTM", "GRU"]:
            m["rnn"]["models"][key] = {
                "test_accuracy": vals.get("test_accuracy"),
                "test_f1_macro": vals.get("test_f1_macro"),
                "auc": vals.get("auc"),
            }
            print(f"  {key}: acc={vals.get('test_accuracy')} f1={vals.get('test_f1_macro')}")

    # Meilleur modele
    best_acc = -1
    best_model = "GRU"
    for model_name, vals in m["rnn"]["models"].items():
        acc = vals.get("test_accuracy") or 0
        if acc > best_acc:
            best_acc = acc
            best_model = model_name
    m["rnn"]["best_model"] = best_model
    print(f"  Meilleur modele: {best_model} ({best_acc:.4f})")

    save_metrics(m)


def update_part5():
    print("\n=== Mise a jour Part V (Hybrides) ===")
    nb_results = extract_from_notebook("part5_hybrid_models.ipynb")
    m = load_metrics()

    if "hybrid" not in m:
        m["hybrid"] = {}

    for model_name, vals in nb_results.items():
        if "CNN+LSTM" in model_name:
            m["hybrid"]["cnn_lstm"] = {
                "accuracy": vals.get("test_accuracy"),
                "auc": vals.get("auc"),
                "f1_macro": vals.get("test_f1_macro"),
            }
        elif "CNN+Attention" in model_name or "MultiModal" in model_name:
            m["hybrid"]["cnn_attention"] = {
                "accuracy": vals.get("test_accuracy"),
                "auc": vals.get("auc"),
                "f1_macro": vals.get("test_f1_macro"),
            }
        elif "CNN_seul" in model_name or "CNN seul" in model_name:
            m["hybrid"]["cnn_baseline"] = {
                "accuracy": vals.get("test_accuracy"),
                "auc": vals.get("auc"),
            }

    save_metrics(m)


def update_part6():
    print("\n=== Mise a jour Part VI (Ablation) ===")
    m = load_metrics()
    if "ablation" not in m:
        m["ablation"] = {"mlp": [], "cnn": [], "rnn": []}
    # L'ablation est plus complexe a extraire (tableaux pandas)
    # Elle sera mise a jour manuellement ou via export CSV depuis le notebook
    print("  Note: ablation exportee sous forme de heatmaps (voir ablation_*.png)")
    print("  Pour les donnees brutes, exporter depuis le notebook: df_mlp_ablation.to_csv('ablation_mlp.csv')")
    save_metrics(m)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["3", "5", "6", "all"], default="all")
    args = parser.parse_args()

    if args.part in ("3", "all"):
        update_part3()
    if args.part in ("5", "all"):
        update_part5()
    if args.part in ("6", "all"):
        update_part6()

    print("\n[OK] metrics.json mis a jour. Relancez l'app Streamlit pour voir les resultats.")


if __name__ == "__main__":
    main()
