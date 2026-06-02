"""Extract metrics from executed notebook outputs."""
import json
import re
from pathlib import Path

import nbformat


def cell_text(cell):
    parts = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            parts.append("".join(out.get("text", [])))
        elif out.get("output_type") == "execute_result":
            data = out.get("data", {})
            parts.append("".join(data.get("text/plain", [])))
    return "\n".join(parts)


def parse_notebook(path):
    nb = nbformat.read(path, as_version=4)
    total = len([c for c in nb.cells if c.cell_type == "code"])
    ok = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("outputs"))
    text = "\n".join(cell_text(c) for c in nb.cells if c.cell_type == "code")
    metrics = {
        "file": path.name,
        "cells_ok": ok,
        "cells_total": total,
        "status": "ok" if ok == total else "partial",
    }
    for key, pattern in [
        ("accuracy", r"accuracy[:\s]+([0-9.]+)"),
        ("test_accuracy", r"test[_ ]?accuracy[:\s]+([0-9.]+)"),
        ("auc_roc", r"AUC[- ]?ROC[:\s]+([0-9.]+)"),
        ("auc", r"\bAUC[:\s]+([0-9.]+)"),
        ("loss", r"loss[:\s]+([0-9.]+)"),
        ("bleu", r"BLEU[:\s]+([0-9.]+)"),
        ("params", r"param[eè]tres[^\d]*([0-9,]+)"),
    ]:
        m = re.search(pattern, text, re.I)
        if m:
            metrics[key] = m.group(1).replace(",", "")
    return metrics, text


if __name__ == "__main__":
    root = Path(__file__).parent
    for name in [
        "eda_medical_datasets.ipynb",
        "part1_mlp_diabetes.ipynb",
        "part2_cnn_pneumonia.ipynb",
        "part3_rnn_medical.ipynb",
    ]:
        p = root / name
        if p.exists():
            m, _ = parse_notebook(p)
            print(json.dumps(m, indent=2, ensure_ascii=False))
