"""Verify the frozen timing record and regenerate paper cost tables/figure.

Run from any directory with NumPy and Matplotlib. No GPU or private cache is
needed to recompute summaries. --import-root copies an existing timing archive;
it never runs inference or changes the registered measurement protocol.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parents[1]
O = P / "SourceData/inference_20261002"
NAMES = {
    "li2021_cnn": "Li-CNN (2021)", "gan2021_cnn": "Gan-CNN (2021)",
    "gong_cnn": "Gong-CNN (2022)", "gong_cnn_bilstm": "Gong-CNN--BiLSTM (2022)",
    "gan_cnn": "Gan-CNN (2023)", "deeprfqc": "DeepRFQC (2024)",
    "hegaz_capsule": "RF-Capsule (2025)", "chen2026_image": "Chen-AlexNet (2026)",
    "reference_ag3": "Reference--AG3", "reference_multifilter": "Reference--multi-filter",
    "descriptors_ag3": "Descriptors--AG3", "descriptors_multifilter": "Descriptors--multi-filter",
    "combined_ag3": "Combined--AG3", "combined_multifilter": "Combined--multi-filter",
    "xiong2025_fcm": "Xiong-FCM (2025)", "logreg_ag3": "LogReg--AG3",
    "logreg_multifilter": "LogReg--multi-filter", "reference_ag1": "Reference--AG1",
    "reference_ag5": "Reference--AG5",
}
PRIMARY = list(NAMES)[:10]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def table(headers, rows, alignment):
    return ("\\begin{tabular}{" + alignment + "}\n\\toprule\n"
            + " & ".join(headers) + r"\\" + "\n\\midrule\n"
            + "\n".join(" & ".join(row) + r"\\" for row in rows)
            + "\n\\bottomrule\n\\end{tabular}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--import-root", type=Path)
    args = parser.parse_args()
    O.mkdir(exist_ok=True)
    if args.import_root:
        for name in ("protocol.json", "benchmark_inference.py", "completion.json",
                     "dependency_resume.json", "launch.json", "status.json"):
            shutil.copy2(args.import_root / name, O / name)
        for name in ("results", "bundles", "sample_ids"):
            shutil.copytree(args.import_root / name, O / name, dirs_exist_ok=True)

    protocol = json.loads((O / "protocol.json").read_text())
    complete = json.loads((O / "completion.json").read_text())
    assert complete["protocol_sha256"] == sha(O / "protocol.json")
    assert complete["script_sha256"] == sha(O / "benchmark_inference.py")
    entries = {x["name"]: x for x in protocol["methods"]}
    assert set(entries) == set(NAMES)
    expected = {f"results/{n}_round{r}.json" for n in entries for r in range(3)}
    assert complete["completed"] == 57
    assert set(complete["sha256"]) == expected
    assert {str(p.relative_to(O)) for p in (O / "results").glob("*.json")} == expected
    rounds, replay = [], []
    shared_ids = None
    for n, entry in entries.items():
        assert sha(O / "bundles" / f"{n}.json") == entry["bundle_sha256"]
        ids_sha = None
        if n != "xiong2025_fcm":
            ids = json.loads((O / "sample_ids" / f"{n}.json").read_text())
            assert len(ids) == len(set(ids)) == 512
            if shared_ids is None:
                shared_ids = ids
            assert ids == shared_ids
            ids_sha = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        for r in range(3):
            rel = f"results/{n}_round{r}.json"
            assert sha(O / rel) == complete["sha256"][rel]
            d = json.loads((O / rel).read_text())
            assert (d["method"], d["round"], d["seed"]) == (n, r, 20260929)
            assert d["protocol_sha256"] == complete["protocol_sha256"]
            assert d["script_sha256"] == complete["script_sha256"]
            assert d["bundle_sha256"] == entry["bundle_sha256"]
            assert d["torch"] == "2.8.0+cu128"
            assert all(x["num_threads"] == 4 for x in d["threadpools"])
            assert protocol["gpu_uuid"] in d["hardware_before"]
            assert protocol["gpu_uuid"] in d["hardware_after"]
            row = dict(method=n, round=r, device=d["device"], parameters=d["parameters"])
            if d["kind"] == "fcm":
                assert d["records"] == 24533 and d["predicted_good"] == 5145
                rate = d["records"] / d["pool_seconds"]
                np.testing.assert_allclose(rate, d["pool_records_per_second"], rtol=1e-12)
                row.update(pool_seconds=d["pool_seconds"], pool_records_per_second=rate)
            else:
                assert d["sample_ids_sha256"] == ids_sha and d["records"] == 512
                assert d["views"] == ([7] if "multifilter" in n else [1])
                single = np.array(d["latency_seconds"])
                bulk = np.array(d["batch32_seconds"])
                assert len(single) == 100 and len(bulk) == 32
                assert np.all(single > 0) and np.all(bulk > 0)
                row.update(latency_p50_ms=1000 * np.median(single),
                           latency_p95_ms=1000 * np.percentile(single, 95),
                           batch32_records_per_second=32 * len(bulk) / bulk.sum())
                for k in ("latency_p50_ms", "latency_p95_ms", "batch32_records_per_second"):
                    np.testing.assert_allclose(row[k], d[k], rtol=1e-12)
                if d["kind"] == "neural":
                    assert d["source_model_sha256"] == entry["source_model_sha256"]
                    net = np.array(d["forward_batch32_seconds"])
                    assert len(net) == 32 and np.all(net > 0)
                    rate = 32 * len(net) / net.sum()
                    np.testing.assert_allclose(rate, d["forward_batch32_records_per_second"], rtol=1e-12)
                    row.update(forward_batch32_records_per_second=rate,
                               peak_gpu_mib=d["pipeline_peak_gpu_allocated_bytes"] / 2**20)
                    if r == 0:
                        assert d["finite_replay"]
                        replay.append(dict(method=n, maximum_score_difference=d["archive_max_abs_score_difference"],
                                           changed_decisions=d["archive_decisions_changed"], records=512))
                if r == 0:
                    assert d["finite_replay"]
            rounds.append(row)

    summary = []
    for n in NAMES:
        selected = [x for x in rounds if x["method"] == n]
        row = dict(method=n, label=NAMES[n], device=selected[0]["device"], parameters=selected[0]["parameters"])
        assert len({x["parameters"] for x in selected}) == 1
        for k in selected[0]:
            if k in ("method", "round", "device", "parameters"):
                continue
            vals = [x[k] for x in selected]
            row.update({k: float(np.median(vals)), k + "_min": float(min(vals)), k + "_max": float(max(vals))})
        summary.append(row)
    by = {x["method"]: x for x in summary}
    csv_write(O / "round_summaries.csv", rounds)
    csv_write(O / "summary.csv", summary)
    csv_write(O / "score_replay.csv", replay)
    mainrows = []
    for n in PRIMARY:
        x = by[n]
        mainrows.append([NAMES[n], f"{x['parameters']/1000:.2f}", f"{x['latency_p50_ms']:.2f}",
                         f"{x['latency_p95_ms']:.2f}", f"{x['batch32_records_per_second']:,.0f}",
                         f"{x['forward_batch32_records_per_second']:,.0f}", f"{x['peak_gpu_mib']:.1f}"])
    (P / "tables/inference_primary.tex").write_text(table(
        ["Configuration", "Params (k)", "p50 (ms)", "p95 (ms)", "API (RF/s)", "Forward (RF/s)", "MiB"],
        mainrows, "lrrrrrr"))
    allrows = []
    for n in NAMES:
        x = by[n]
        if n == "xiong2025_fcm":
            allrows.append([NAMES[n], "CPU", "Pool", "--",
                            f"{x['pool_records_per_second']:,.0f}",
                            f"[{x['pool_records_per_second_min']:,.0f}, {x['pool_records_per_second_max']:,.0f}]"])
        else:
            allrows.append([NAMES[n], "GPU" if x["device"].startswith("cuda") else "CPU", "32",
                            f"{x['latency_p50_ms']:.3f} / {x['latency_p95_ms']:.3f}",
                            f"{x['batch32_records_per_second']:,.0f}",
                            f"[{x['batch32_records_per_second_min']:,.0f}, {x['batch32_records_per_second_max']:,.0f}]"])
    (P / "tables/inference_all.tex").write_text(table(
        ["Configuration", "Device", "Batch", "p50 / p95 (ms)", "RF/s", "Round range (RF/s)"], allrows, "lllrrr"))

    accuracy = {r["key"]: r for r in csv.DictReader((P / "SourceData/benchmark_v1/summary.csv").open())}
    # Registry names differ from the original primary-table keys for the reference.
    mapping = {"reference_ag3": "ours_ag3", "reference_multifilter": "ours_multiband"}
    for n in PRIMARY:
        if n not in accuracy and mapping.get(n) not in accuracy:
            matches = [k for k, a in accuracy.items() if a["group"] == "primary" and
                       ((n == "reference_ag3" and a["arm"] == "waveform_random" and a["input"] == "AG3") or
                        (n == "reference_multifilter" and a["arm"] == "waveform_random" and a["input"] != "AG3"))]
            assert len(matches) == 1, (n, list(accuracy))
            mapping[n] = matches[0]
    joined = []
    for n in PRIMARY:
        a = accuracy[n if n in accuracy else mapping[n]]
        joined.append(dict(method=n, label=NAMES[n], macro_f1_mean_percent=100*float(a["macro_f1_mean"]),
                           macro_f1_sd_percent=100*float(a["macro_f1_std"]), **{
                               k: by[n][k] for k in ("batch32_records_per_second", "batch32_records_per_second_min", "batch32_records_per_second_max")}))
    csv_write(O / "accuracy_cost.csv", joined)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 5.0), layout="constrained", sharey=True)
    for i, x in enumerate(joined):
        color = "#be5734" if x["method"] == "reference_multifilter" else "#2c6586"
        axes[0].errorbar(x["macro_f1_mean_percent"], i, xerr=x["macro_f1_sd_percent"], fmt="o", color=color, capsize=3)
        v = x["batch32_records_per_second"]
        axes[1].errorbar(v, i, xerr=[[v-x["batch32_records_per_second_min"]], [x["batch32_records_per_second_max"]-v]], fmt="o", color=color, capsize=3)
    axes[0].set_yticks(range(len(joined)), [x["label"].replace("--", "–") for x in joined])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Test macro-F1 (%)\nMean ± SD of three training seeds")
    axes[0].set_xlim(85.5, 90)
    axes[1].set_xlabel("API throughput (RF/s), batch 32\nMedian and range of three timing rounds")
    axes[1].set_xscale("log")
    axes[1].set_xlim(800, 25000)
    axes[0].set_title("(a) Original common-task scores")
    axes[1].set_title("(b) Deployment cost on one RTX 5090")
    for ax in axes:
        ax.grid(axis="x", alpha=.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(P / "figures/inference_cost.pdf")
    fig.savefig(P / "figures/inference_cost.png", dpi=180)
    plt.close(fig)
    findings = dict(
        gong_cnn_vs_bilstm_throughput_ratio=by["gong_cnn"]["batch32_records_per_second"]/by["gong_cnn_bilstm"]["batch32_records_per_second"],
        gong_bilstm_vs_cnn_latency_ratio=by["gong_cnn_bilstm"]["latency_p50_ms"]/by["gong_cnn"]["latency_p50_ms"],
        multifilter_vs_ag3_throughput_ratio=by["reference_multifilter"]["batch32_records_per_second"]/by["reference_ag3"]["batch32_records_per_second"],
        fcm_pool_seconds=by["xiong2025_fcm"]["pool_seconds"],
        maximum_replay_score_difference=max(x["maximum_score_difference"] for x in replay),
        total_replay_changed_decisions=sum(x["changed_decisions"] for x in replay))
    dump(O / "verification.json", dict(passed=True, completed=57, methods=19, timing_rounds=3,
         completion_hash_checks=len(expected), protocol_sha256=complete["protocol_sha256"],
         script_sha256=complete["script_sha256"], independent_records=512,
         common_sample_ids_sha256=hashlib.sha256("\n".join(shared_ids).encode()).hexdigest(),
         recomputed_from_raw_call_durations=True, findings=findings))
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
