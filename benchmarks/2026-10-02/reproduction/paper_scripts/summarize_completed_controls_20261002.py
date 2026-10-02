"""Portable individual-seed tables and independent cross-target count checks."""
from pathlib import Path
import csv
import json
import numpy as np
from analyze_results import metrics as independent_metrics

P = Path(__file__).resolve().parents[1]
O = P / "SourceData/resolution_20260930"


def table(headers, rows, fmt):
    return ("\\begin{tabular}{" + fmt + "}\n\\toprule\n" + " & ".join(headers)
            + r"\\" + "\n\\midrule\n"
            + "\n".join(" & ".join(row) + r"\\" for row in rows)
            + "\n\\bottomrule\n\\end{tabular}\n")


def main():
    metrics = list(csv.DictReader((O / "eight_seed_metrics.csv").open()))
    names = ["reference_ag3", "reference_multifilter", "gong_cnn", "gong_cnn_bilstm"]
    seeds = sorted({int(x["seed"]) for x in metrics})
    values = {(x["method"], int(x["seed"])): float(x["macro_f1_percent"]) for x in metrics}
    assert len(metrics) == len(values) == 32
    body = [[str(s)] + [f"{values[n, s]:.5f}" for n in names] for s in seeds]
    body.append(["Mean $\\pm$ SD"] + [f"${np.mean([values[n,s] for s in seeds]):.3f}\\pm{np.std([values[n,s] for s in seeds],ddof=1):.3f}$" for n in names])
    (O / "eight_seed_individuals_compact.tex").write_text(table(
        ["Seed", "Reference--AG3", "Reference--multi-filter", "Gong-CNN", "Gong-CNN--BiLSTM"], body, "lrrrr"))
    z = np.load(O / "cross_protocol_predictions.npz", allow_pickle=False)
    assert len(z["sample_ids"]) == len(set(z["sample_ids"])) == 18305
    assert len(set(z["stations"])) == 22
    lookup = dict(zip(z["run_names"], z["probabilities"]))
    rows = list(csv.DictReader((O / "cross_protocol_metrics.csv").open()))
    assert len(rows) == len(lookup) == 6
    count_rows, score_rows, audit = [], [], []
    for x in rows:
        direction, seed = x["direction"], int(x["seed"])
        name = f"{direction}_seed{seed}"
        prob = lookup[name]
        threshold = float(x["threshold"])
        labels = ("strict_labels", "ag3_labels") if direction == "strict_to_ag3" else ("ag3_labels", "strict_labels")
        short = r"Strict $\to$ AG3" if direction == "strict_to_ag3" else r"AG3 $\to$ strict"
        score_rows.append([short, str(seed), f"{float(x['source_macro_f1_percent']):.3f}",
                           f"{float(x['target_macro_f1_percent']):.3f}", f"{float(x['target_ap_percent']):.3f}",
                           f"[{float(x['target_minus_source_lower_pp']):+.3f}, {float(x['target_minus_source_upper_pp']):+.3f}]"])
        for role, key in zip(("source", "target"), labels):
            y = z[key]
            pred = prob >= threshold
            tn = int(np.sum((y == 0) & ~pred)); fp = int(np.sum((y == 0) & pred))
            fn = int(np.sum((y == 1) & ~pred)); tp = int(np.sum((y == 1) & pred))
            f1 = .5 * (2*tp/(2*tp+fp+fn) + 2*tn/(2*tn+fp+fn))
            np.testing.assert_allclose(100*f1, float(x[role + "_macro_f1_percent"]), rtol=1e-12)
            recomputed = independent_metrics(y, prob, threshold)
            np.testing.assert_allclose(100*recomputed['good_auprc'], float(x[role + '_ap_percent']), rtol=1e-12)
            assert tn + fp + fn + tp == 18305
            count_rows.append([short, str(seed), role.title(), f"{threshold:.2f}", str(tn), str(fp), str(fn), str(tp)])
            audit.append(dict(run=name, evaluation_labels=role, threshold=threshold, tn=tn, fp=fp, fn=fn, tp=tp,
                              macro_f1_percent=100*f1))
    (O / "cross_protocol_individuals.tex").write_text(table(
        ["Direction", "Seed", "Source $F_1$", "Target $F_1$", "Target AP", "Difference 95\\% interval"], score_rows, "llrrrr"))
    (O / "cross_protocol_counts.tex").write_text(table(
        ["Direction", "Seed", "Labels", "$\\tau$", "TN", "FP", "FN", "TP"], count_rows, "lllrrrrr"))
    with (O / "cross_protocol_contingencies.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(audit[0])); w.writeheader(); w.writerows(audit)
    (O / "portable_contingency_verification.json").write_text(json.dumps(dict(
        passed=True, prediction_vectors=6, records_per_vector=18305,
        independently_recomputed_contingency_tables=12, macro_f1_checks=12, average_precision_checks=12,
        individual_seed_scores=32), indent=2) + "\n")


if __name__ == "__main__":
    main()
