"""Group 9, Block 4. Final integration of all four models and cross-model analysis.

Reads the delivered prediction files of Blocks 1-4 (never retrains anything),
recomputes every metric with the shared evaluation function and writes the
combined tables and figures used in the report to the FINAL folder .

Run from the FINAL/block4_fbert folder:
    python integrate_all.py
"""
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix

from shared_evaluation import evaluate_predictions

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT
FIG = OUT / "figures"
TAB = OUT / "tables"

# Prediction files of each block. The test order is checked below.
PREDICTIONS = {
    "Logistic Regression": {
        "In-domain": ROOT / "block1_logreg/results_logreg_in_domain.csv",
        "Cross-domain": ROOT / "block1_logreg/results_logreg_cross_domain.csv",
    },
    "BERT": {
        "In-domain": ROOT / "block2_bert/results_bert_in_domain.csv",
        "Cross-domain": ROOT / "block2_bert/results_bert_cross_domain.csv",
    },
    "HateBERT": {
        "In-domain": ROOT / "block3_hatebert/hatebert_results/results_hatebert_in_domain.csv",
        "Cross-domain": ROOT / "block3_hatebert/hatebert_results/results_hatebert_cross_domain.csv",
    },
    "fBERT": {
        "In-domain": ROOT / "block4_fbert/fbert_results/results_fbert_in_domain.csv",
        "Cross-domain": ROOT / "block4_fbert/fbert_results/results_fbert_cross_domain.csv",
    },
}
MODELS = list(PREDICTIONS)
SETTINGS = ["In-domain", "Cross-domain"]
# One fixed colour per model, the same in every figure.
COLORS = {"Logistic Regression": "#2a78d6", "BERT": "#eb6834", "HateBERT": "#1baf7a", "fBERT": "#eda100"}
INK, MUTED = "#0b0b0b", "#52514e"

# Short hand-made list of swear words and insults, only used to count them.
EXPLICIT = re.compile(
    r"\b(?:fuck\w*|shit\w*|bitch\w*|ass|asshole\w*|damn\w*|crap\w*|dick\w*|pussy|cunt\w*|"
    r"bastard\w*|nigg\w*|whore\w*|slut\w*|idiot\w*|stupid\w*|moron\w*|dumb\w*|retard\w*|"
    r"scum\w*|hell|piss\w*|motherf\w*|fag\w*|douche\w*|jackass\w*)\b",
    re.IGNORECASE,
)


def load_data():
    data = {n: pd.read_csv(ROOT / "block1_logreg" / f) for n, f in
            [("OLID train small", "olid-train-small.csv"), ("HASOC train", "hasoc-train.csv"), ("OLID test", "olid-test.csv")]}
    return data


def load_predictions(test):
    preds = {}
    for model, files in PREDICTIONS.items():
        for setting, path in files.items():
            r = pd.read_csv(path)
            assert len(r) == 860, path
            assert (r["id"].astype(str).values == test["id"].astype(str).values).all(), f"Test order differs: {path}"
            assert (r["labels"].values == test["labels"].values).all(), f"Gold labels differ: {path}"
            preds[(model, setting)] = r["prediction"].astype(int).values
    return preds


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def metrics_table(test, preds):
    rows = [evaluate_predictions(test["labels"], preds[(m, s)], m, s, show_matrix=False)
            for m in MODELS for s in SETTINGS]
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(OUT / "A4_all_model_metrics.csv", index=False)
    # Compare with the score files of each block (setting names written the same way).
    delivered = pd.concat([pd.read_csv(p) for p in [
        ROOT / "block1_logreg/metrics_logreg_comparison.csv", ROOT / "block2_bert/metrics_bert_comparison.csv",
        ROOT / "block3_hatebert/hatebert_results/metrics_hatebert_comparison.csv",
        ROOT / "block4_fbert/fbert_results/metrics_fbert_comparison.csv"]], ignore_index=True)
    delivered["setting"] = delivered["setting"].str.capitalize()
    merged = table.merge(delivered, on=["model", "setting"], suffixes=("", "_delivered"))
    assert len(merged) == 8
    for col in ["macro_precision", "macro_recall", "macro_f1", "f1_0", "f1_1"]:
        assert np.allclose(merged[col], merged[col + "_delivered"]), col
    return table


def drop_table(table):
    rows = []
    for m in MODELS:
        a = table[(table.model == m) & (table.setting == "In-domain")].iloc[0]
        b = table[(table.model == m) & (table.setting == "Cross-domain")].iloc[0]
        rows.append({
            "model": m, "macro_f1_in": a.macro_f1, "macro_f1_cross": b.macro_f1,
            "abs_drop_macro_f1": a.macro_f1 - b.macro_f1,
            "rel_drop_macro_f1_percent": 100 * (a.macro_f1 - b.macro_f1) / a.macro_f1,
            "drop_f1_0": a.f1_0 - b.f1_0, "drop_f1_1": a.f1_1 - b.f1_1,
            "change_precision_1": b.precision_1 - a.precision_1, "change_recall_1": b.recall_1 - a.recall_1,
        })
    d = pd.DataFrame(rows)
    d.to_csv(TAB / "domain_drop.csv", index=False)
    return d


def error_tables(test, preds):
    y = test["labels"].values
    words = test["text"].str.split().str.len()
    bins = pd.cut(words, bins=[-1, 15, 30, np.inf], labels=["0-15 words", "16-30 words", "31+ words"])
    explicit = test["text"].str.contains(EXPLICIT)
    err, length, expl, mcn = [], [], [], []
    for m in MODELS:
        for s in SETTINGS:
            p = preds[(m, s)]
            tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
            err.append({"model": m, "setting": s, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                        "false_positive_rate": fp / (tn + fp), "false_negative_rate": fn / (fn + tp),
                        "predicted_offensive_percent": 100 * p.mean()})
            for g in bins.cat.categories:
                k = (bins == g).values
                length.append({"model": m, "setting": s, "length_group": g, "n": int(k.sum()),
                               "error_rate": float((p[k] != y[k]).mean())})
            for flag, name in [(True, "with explicit word"), (False, "without explicit word")]:
                k = (explicit == flag).values
                off, non = k & (y == 1), k & (y == 0)
                expl.append({"model": m, "setting": s, "subset": name,
                             "n_offensive": int(off.sum()), "recall_offensive": float(p[off].mean()),
                             "n_non_offensive": int(non.sum()), "fpr_non_offensive": float(p[non].mean())})
        # Exact McNemar test on the same 860 tweets: in-domain vs cross-domain model.
        a, b = preds[(m, "In-domain")] == y, preds[(m, "Cross-domain")] == y
        only_in, only_cross = int((a & ~b).sum()), int((~a & b).sum())
        mcn.append({"model": m, "both_correct": int((a & b).sum()), "only_in_domain_correct": only_in,
                    "only_cross_domain_correct": only_cross, "both_wrong": int((~a & ~b).sum()),
                    "mcnemar_exact_p": binomtest(only_in, only_in + only_cross, 0.5).pvalue})
    pd.DataFrame(err).to_csv(TAB / "error_summary_all.csv", index=False)
    pd.DataFrame(length).to_csv(TAB / "errors_by_length_all.csv", index=False)
    pd.DataFrame(expl).to_csv(TAB / "explicit_word_analysis.csv", index=False)
    pd.DataFrame(mcn).to_csv(TAB / "paired_in_vs_cross.csv", index=False)

    # Pairwise exact McNemar between the two domain-adapted models, per setting.
    pair = []
    for s in SETTINGS:
        for m1, m2 in [("fBERT", "HateBERT"), ("fBERT", "BERT"), ("HateBERT", "BERT")]:
            a, b = preds[(m1, s)] == y, preds[(m2, s)] == y
            n1, n2 = int((a & ~b).sum()), int((~a & b).sum())
            pair.append({"setting": s, "model_a": m1, "model_b": m2, "only_a_correct": n1,
                         "only_b_correct": n2, "mcnemar_exact_p": binomtest(n1, n1 + n2, 0.5).pvalue})
    pd.DataFrame(pair).to_csv(TAB / "pairwise_model_tests.csv", index=False)

    # Test tweets that every transformer misclassifies in both settings.
    wrong = np.ones(len(y), bool)
    for m in ["BERT", "HateBERT", "fBERT"]:
        for s in SETTINGS:
            wrong &= preds[(m, s)] != y
    hard = test.loc[wrong, ["id", "text", "labels"]]
    hard.to_csv(TAB / "hard_examples_all_transformers_wrong.csv", index=False)
    return pd.DataFrame(err)


def domain_profile(data):
    """Simple properties of each dataset, used to discuss the domain shift."""
    tok = re.compile(r"[a-z0-9']+")
    rows, vocab = [], {}
    for name, df in data.items():
        t = df["text"].astype(str)
        vocab[name] = pd.Series([w for x in t.str.lower() for w in tok.findall(x)])
        rows.append({
            "dataset": name,
            "has_@USER_placeholder_%": 100 * t.str.contains("@USER", regex=False).mean(),
            "has_real_@mention_%": 100 * t.str.contains(r"@(?!USER\b)\w+").mean(),
            "has_URL_placeholder_%": 100 * t.str.contains(r"\bURL\b").mean(),
            "has_http_link_%": 100 * t.str.contains("http").mean(),
            "has_hashtag_%": 100 * t.str.contains(r"#\w+").mean(),
            "hashtags_per_message": t.str.count(r"#\w+").mean(),
            "has_explicit_word_%": 100 * t.str.contains(EXPLICIT).mean(),
            "explicit_word_%_offensive": 100 * t[df["labels"] == 1].str.contains(EXPLICIT).mean(),
            "explicit_word_%_non_offensive": 100 * t[df["labels"] == 0].str.contains(EXPLICIT).mean(),
        })
    prof = pd.DataFrame(rows)
    test_tokens = vocab["OLID test"]
    for name in ["OLID train small", "HASOC train"]:
        types = set(vocab[name])
        prof.loc[prof.dataset == name, "test_token_coverage_%"] = 100 * test_tokens.isin(types).mean()
        prof.loc[prof.dataset == name, "test_type_coverage_%"] = 100 * pd.Series(test_tokens.unique()).isin(types).mean()
    prof.to_csv(TAB / "domain_profile.csv", index=False)
    return prof


def fig_macro_f1(table):
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x, w = np.arange(len(SETTINGS)), 0.19
    for i, m in enumerate(MODELS):
        vals = table[table.model == m].set_index("setting").loc[SETTINGS, "macro_f1"].values
        bars = ax.bar(x + (i - 1.5) * (w + 0.01), vals, w, label=m, color=COLORS[m])
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8.5, color=INK)
    ax.set_xticks(x, SETTINGS, color=INK)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1", color=INK)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "macro_f1_all_models.png", dpi=200)
    plt.close(fig)


def fig_class_f1(table):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    x, w = np.arange(len(SETTINGS)), 0.19
    for ax, label, title in [(axes[0], 0, "Non-offensive class (0)"), (axes[1], 1, "Offensive class (1)")]:
        for i, m in enumerate(MODELS):
            vals = table[table.model == m].set_index("setting").loc[SETTINGS, f"f1_{label}"].values
            bars = ax.bar(x + (i - 1.5) * (w + 0.01), vals, w, label=m, color=COLORS[m])
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7.5, color=INK)
        ax.set_xticks(x, SETTINGS, color=INK)
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_ylim(0, 1)
        style(ax)
    axes[0].set_ylabel("F1", color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "class_f1_all_models.png", dpi=200)
    plt.close(fig)


def fig_confusions(test, preds):
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
    for j, m in enumerate(MODELS):
        for i, s in enumerate(SETTINGS):
            ax = axes[i, j]
            cm = confusion_matrix(test["labels"], preds[(m, s)], labels=[0, 1])
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=620)
            for (r, c), v in np.ndenumerate(cm):
                ax.text(c, r, str(v), ha="center", va="center", fontsize=11,
                        color="white" if v > 330 else INK)
            ax.set_xticks([0, 1], ["Pred 0", "Pred 1"], fontsize=8)
            ax.set_yticks([0, 1], ["True 0", "True 1"], fontsize=8)
            ax.set_title(f"{m}\n{s}", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrices_all.png", dpi=200)
    plt.close(fig)
    rows = []
    for m in MODELS:
        for s in SETTINGS:
            tn, fp, fn, tp = confusion_matrix(test["labels"], preds[(m, s)], labels=[0, 1]).ravel()
            rows.append({"model": m, "setting": s, "TN": tn, "FP": fp, "FN": fn, "TP": tp})
    pd.DataFrame(rows).to_csv(TAB / "confusion_matrices_all.csv", index=False)


def seed_table():
    """Seed check: seed 42 (main runs) plus seeds 1 and 2 (seed_check/)."""
    main_runs = {"HateBERT": ROOT / "block3_hatebert/hatebert_results/metrics_hatebert_comparison.csv",
                 "fBERT": ROOT / "block4_fbert/fbert_results/metrics_fbert_comparison.csv"}
    rows = []
    for model, tag in [("HateBERT", "hatebert"), ("fBERT", "fbert")]:
        files = {42: main_runs[model]}
        files.update({s: ROOT / f"block4_fbert/seed_check/{tag}_seed{s}/metrics_{tag}_comparison.csv" for s in (1, 2)})
        for seed, path in files.items():
            if not path.exists():
                continue
            m = pd.read_csv(path).set_index("setting")["macro_f1"]
            rows.append({"model": model, "seed": seed, "macro_f1_in": m["In-domain"],
                         "macro_f1_cross": m["Cross-domain"], "drop": m["In-domain"] - m["Cross-domain"]})
    seeds = pd.DataFrame(rows)
    seeds.to_csv(TAB / "seed_check.csv", index=False)
    summary = seeds.groupby("model")[["macro_f1_in", "macro_f1_cross", "drop"]].agg(["mean", "std"])
    summary.to_csv(TAB / "seed_check_summary.csv")
    return seeds, summary


def main():
    for d in (OUT, FIG, TAB):
        d.mkdir(parents=True, exist_ok=True)
    data = load_data()
    test = data["OLID test"]
    preds = load_predictions(test)
    table = metrics_table(test, preds)
    drops = drop_table(table)
    error_tables(test, preds)
    prof = domain_profile(data)
    fig_macro_f1(table)
    fig_class_f1(table)
    fig_confusions(test, preds)
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))
    print(drops.round(4).to_string(index=False))
    print(prof.round(2).T.to_string())
    seeds, summary = seed_table()
    print(seeds.round(4).to_string(index=False))
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
