"""Group 9, Block 4. Fine-tunes fBERT in-domain and cross-domain and saves the results.

Run from the directory containing the three supplied CSV files:
    python fbert_block4.py --setting all
Existing results are not overwritten, use a new output directory to train again.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import time
from datetime import datetime, timezone

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HOME", str(Path(".hf_cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mpl_cache").resolve()))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from shared_evaluation import evaluate_predictions

CHECKPOINT = "diptanu/fBERT"
REVISION = "c089daa35cb177238dd91ffe3010c2d8bbd069bf"
SEED = 42
TRAIN_FILES = {"in_domain": "olid-train-small.csv", "cross_domain": "hasoc-train.csv"}
SETTING_NAMES = {"in_domain": "In-domain", "cross_domain": "Cross-domain"}
# Same settings as the HateBERT script of Block 3, chosen before testing fBERT.
TRAINING = dict(
    num_train_epochs=3,
    learning_rate=2e-5,
    train_batch_size=8,
    gradient_accumulation_steps=2,
    eval_batch_size=32,
    max_seq_length=128,
    optimizer="AdamW",
    scheduler="linear_schedule_with_warmup",
    warmup_ratio=0.06,
    warmup_steps=0,
    weight_decay=0.01,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    manual_seed=SEED,
    fp16=True,
    do_lower_case=True,
    use_early_stopping=False,
    evaluate_during_training=False,
    save_model_every_epoch=False,
    save_eval_checkpoints=False,
    save_steps=-1,
    save_optimizer_and_scheduler=False,
    overwrite_output_dir=False,
    reprocess_input_data=True,
    no_cache=True,
    use_multiprocessing=False,
    use_multiprocessing_for_evaluation=False,
    dataloader_num_workers=0,
    thread_count=4,
    process_count=1,
    n_gpu=1,
    logging_steps=50,
    silent=False,
    wandb_project=None,
)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_data(data_dir):
    data = {}
    for filename in [*TRAIN_FILES.values(), "olid-test.csv"]:
        df = pd.read_csv(Path(data_dir) / filename)
        assert {"id", "text", "labels"}.issubset(df.columns)
        assert df.text.notna().all() and df.labels.isin([0, 1]).all()
        assert df.id.is_unique, f"Duplicate IDs within {filename}"
        data[filename] = df
    assert len(data["olid-train-small.csv"]) == len(data["hasoc-train.csv"]) == 5852
    assert len(data["olid-test.csv"]) == 860
    assert data["olid-train-small.csv"].labels.value_counts().to_dict() == data["hasoc-train.csv"].labels.value_counts().to_dict()
    return data


def audit_data(data_dir, output_dir, tokenizer):
    """Counts some statistics of the datasets. Nothing is removed or changed."""
    data = load_data(data_dir)
    test = data["olid-test.csv"]
    rows = []
    for filename, df in data.items():
        lengths = np.array([len(ids) for ids in tokenizer(df.text.tolist(), truncation=False, add_special_tokens=True)["input_ids"]])
        rows.append({
            "dataset": filename, "instances": len(df),
            "class_0": int((df.labels == 0).sum()), "class_1": int((df.labels == 1).sum()),
            "mean_words": float(df.text.str.split().str.len().mean()),
            "mean_characters": float(df.text.str.len().mean()),
            "mean_wordpieces_with_special_tokens": float(lengths.mean()),
            "max_wordpieces_with_special_tokens": int(lengths.max()),
            "truncated_at_128": int((lengths > 128).sum()),
            "truncated_percent": float(100 * (lengths > 128).mean()),
            "duplicate_text_rows": int(df.text.duplicated().sum()),
            "exact_text_overlap_with_test": None if filename == "olid-test.csv" else int(df.text.isin(test.text).sum()),
            "sha256": file_hash(Path(data_dir) / filename),
        })
    pd.DataFrame(rows).to_csv(Path(output_dir) / "data_audit_fbert.csv", index=False)
    return pd.DataFrame(rows)


def export_predictions(test, predictions, logits, setting, output_dir):
    output_dir = Path(output_dir)
    summary = evaluate_predictions(test.labels, predictions, "fBERT", SETTING_NAMES[setting], show_matrix=False)
    summary.to_csv(output_dir / f"metrics_fbert_{setting}.csv", index=False)
    results = test.copy()
    results["prediction"] = np.asarray(predictions, dtype=int)
    results["correct"] = results.labels == results.prediction
    logits = np.asarray(logits)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    results["logit_0"] = logits[:, 0]
    results["logit_1"] = logits[:, 1]
    results["probability_offensive"] = probs[:, 1]
    results.to_csv(output_dir / f"results_fbert_{setting}.csv", index=False)
    cm = confusion_matrix(test.labels, predictions, labels=[0, 1])
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(output_dir / f"confusion_fbert_{setting}.csv")
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ConfusionMatrixDisplay(cm, display_labels=["Non-offensive (0)", "Offensive (1)"]).plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
    ax.set_title(f"fBERT: {SETTING_NAMES[setting]}\nOLID test, n = 860")
    fig.tight_layout()
    fig.savefig(output_dir / f"confusion_fbert_{setting}.png", dpi=180)
    plt.close(fig)
    errors = results.loc[~results.correct].copy()
    errors["error_type"] = np.where(errors.labels == 0, "false_positive", "false_negative")
    errors.to_csv(output_dir / f"errors_fbert_{setting}.csv", index=False)
    return summary


def run_setting(setting, data_dir=".", output_dir="fbert_results", models_dir="fbert_models", cache_dir=".hf_cache"):
    import torch
    from huggingface_hub import snapshot_download
    from simpletransformers.classification import ClassificationModel, ClassificationArgs

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for these documented runs. In Colab select Runtime > Change runtime type > GPU.")
    output_dir, models_dir = Path(output_dir), Path(models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = models_dir / setting
    if (output_dir / f"run_fbert_{setting}.json").exists() or (model_dir.exists() and any(model_dir.iterdir())):
        raise FileExistsError("Run/model already exists. Choose new output and model directories to protect the delivered results.")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    data = load_data(data_dir)
    train = data[TRAIN_FILES[setting]]
    test = data["olid-test.csv"]
    snapshot = snapshot_download(
        CHECKPOINT, revision=REVISION, cache_dir=str(cache_dir),
        allow_patterns=["config.json", "model.safetensors", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"],
    )
    args = ClassificationArgs()
    args.update_from_dict(dict(TRAINING, output_dir=str(model_dir), cache_dir=str(output_dir / "feature_cache" / setting), tensorboard_dir=str(output_dir / "tensorboard" / setting), best_model_dir=str(model_dir / "unused_best_model")))
    print(f"Starting {setting}; independent initialization from {CHECKPOINT}@{REVISION}", flush=True)
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    model = ClassificationModel("bert", snapshot, num_labels=2, args=args, use_cuda=True)
    assert model.tokenizer.do_lower_case
    # fBERT is a language model, so the classification layer is new.
    model.model.config.id2label = {0: "non-offensive", 1: "offensive"}
    model.model.config.label2id = {"non-offensive": 0, "offensive": 1}
    audit_data(data_dir, output_dir, model.tokenizer)
    write_json(output_dir / f"planned_fbert_{setting}.json", {
        "checkpoint": CHECKPOINT, "revision": REVISION, "training": TRAINING,
        "train_file": TRAIN_FILES[setting], "test_file": "olid-test.csv",
        "validation": "None; fixed hyperparameters, no test-based selection", "started_utc": started,
    })
    # Only the training data is used for training.
    global_step, training_loss = model.train_model(train[["text", "labels"]], show_running_loss=False)
    train_seconds = time.perf_counter() - t0
    predictions, logits = model.predict(test.text.tolist())
    summary = export_predictions(test, predictions, logits, setting, output_dir)
    # Save the model and tokenizer so they can be loaded again later.
    model.model.save_pretrained(model_dir, safe_serialization=True)
    model.tokenizer.save_pretrained(model_dir)
    versions = {}
    for name in ["torch", "simpletransformers", "transformers", "huggingface-hub", "datasets", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib"]:
        versions[name] = importlib.metadata.version(name)
    manifest = {
        "status": "completed", "setting": SETTING_NAMES[setting], "checkpoint": CHECKPOINT,
        "revision": REVISION, "seed": SEED, "train_file": TRAIN_FILES[setting],
        "train_rows": len(train), "test_file": "olid-test.csv", "test_rows": len(test),
        "data_sha256": {name: file_hash(Path(data_dir) / name) for name in data},
        "training_parameters": TRAINING, "resolved_model_args": vars(model.args),
        "effective_batch_size": TRAINING["train_batch_size"] * TRAINING["gradient_accumulation_steps"],
        "global_optimizer_steps": int(global_step), "reported_training_loss": float(training_loss),
        "training_seconds_including_initialization": train_seconds,
        "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "python": platform.python_version(), "platform": platform.platform(), "versions": versions,
        "selection": "Final epoch; no validation, early stopping, threshold search or test-based tuning",
        "decision_rule": "argmax of two logits (class 1 if strictly greater; ties go to class 0)",
        "preprocessing": "Provided text unchanged; uncased WordPiece; pad/truncate to 128 including special tokens",
        "class_weights": None, "model_files_sha256": {
            p.name: file_hash(p) for p in model_dir.iterdir() if p.is_file()
        },
    }
    write_json(output_dir / f"run_fbert_{setting}.json", manifest)
    print(summary.to_string(index=False), flush=True)
    print(f"Completed {setting}: {global_step} optimizer steps, {train_seconds:.1f} seconds", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def compare_results(output_dir="fbert_results"):
    output_dir = Path(output_dir)
    frames = [pd.read_csv(output_dir / f"metrics_fbert_{s}.csv") for s in TRAIN_FILES]
    comparison = pd.concat(frames, ignore_index=True)
    comparison.to_csv(output_dir / "metrics_fbert_comparison.csv", index=False)
    a, b = comparison.macro_f1
    write_json(output_dir / "comparison_fbert.json", {
        "macro_f1_in_domain": a, "macro_f1_cross_domain": b,
        "absolute_drop": a - b, "percentage_point_drop": 100 * (a-b),
        "relative_drop_percent": 100 * (a-b) / a if a else None,
    })
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=["all", *TRAIN_FILES], default="all")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--output-dir", default="fbert_results")
    parser.add_argument("--models-dir", default="fbert_models")
    parser.add_argument("--cache-dir", default=".hf_cache")
    cli = parser.parse_args()
    for setting in TRAIN_FILES if cli.setting == "all" else [cli.setting]:
        run_setting(setting, cli.data_dir, cli.output_dir, cli.models_dir, cli.cache_dir)
    if all((Path(cli.output_dir) / f"metrics_fbert_{s}.csv").exists() for s in TRAIN_FILES):
        print(compare_results(cli.output_dir).to_string(index=False))
