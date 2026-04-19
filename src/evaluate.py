import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.cnn_baseline import CNNBaseline  # noqa: E402
from models.clip_standalone import CLIPStandalone  # noqa: E402
from models.cnn_clip_hybrid import CNNCLIPHybrid  # noqa: E402
from src.data_loader import load_fashionpedia_data  # noqa: E402

PLOTS_DIR = ROOT / "outputs" / "plots"
METRICS_PATH = ROOT / "outputs" / "metrics.json"
LOGS_DIR = ROOT / "logs"

CLASS_DIMS = ["formal", "season", "gender", "time", "frequency"]
CLASS_LABELS = {
    "formal": ["casual", "smart casual", "formal"],
    "season": ["spring", "summer", "fall", "winter"],
    "gender": ["male", "female", "neutral"],
    "time": ["day", "night"],
    "frequency": ["occasional", "everyday"],
}


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(model_name: str, ckpt_path: Path, device: torch.device):
    if model_name == "cnn":
        model = CNNBaseline()
    elif model_name == "clip":
        model = CLIPStandalone()
    else:
        model = CNNCLIPHybrid()

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_inference(model, loader, device) -> dict:
    """Collect all predictions and ground truth labels."""
    preds = {k: [] for k in ["formal", "season", "gender", "time", "frequency"]}
    gts = {k: [] for k in ["formal", "season", "gender", "time", "frequency"]}

    for images, labels in loader:
        images = images.to(device)
        output = model(images)

        # all dims are classification: argmax
        for dim in CLASS_DIMS:
            preds[dim].extend(output[dim].argmax(dim=1).cpu().tolist())
            gts[dim].extend(labels[dim].tolist())

    return {"preds": preds, "gts": gts}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(results: dict) -> dict:
    preds, gts = results["preds"], results["gts"]
    metrics = {}

    for dim in CLASS_DIMS:
        p = np.array(preds[dim])
        g = np.array(gts[dim])
        metrics[dim] = {
            "accuracy": float(accuracy_score(g, p)),
            "macro_f1": float(f1_score(g, p, average="macro", zero_division=0)),
        }

    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_accuracy_bars(all_metrics: dict, model_names: list):
    """Bar chart: accuracy per classification attribute across models."""
    dims = CLASS_DIMS
    x = np.arange(len(dims))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, name in enumerate(model_names):
        accs = [all_metrics[name][d]["accuracy"] for d in dims]
        ax.bar(x + i * width, accs, width, label=name)

    ax.set_xticks(x + width)
    ax.set_xticklabels(dims)
    ax.set_ylabel("Accuracy")
    ax.set_title("Classification Accuracy per Attribute")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "accuracy_comparison.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrices(results_by_model: dict):
    """One confusion matrix per (model, classification dim)."""
    for model_name, results in results_by_model.items():
        preds, gts = results["preds"], results["gts"]
        for dim in CLASS_DIMS:
            labels_list = CLASS_LABELS[dim]
            cm = confusion_matrix(
                gts[dim], preds[dim], labels=list(range(len(labels_list)))
            )
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(labels_list)))
            ax.set_yticks(range(len(labels_list)))
            ax.set_xticklabels(labels_list, rotation=45, ha="right")
            ax.set_yticklabels(labels_list)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{model_name} — {dim}")
            fig.colorbar(im, ax=ax)
            # Annotate cells
            for r in range(cm.shape[0]):
                for c in range(cm.shape[1]):
                    ax.text(
                        c,
                        r,
                        str(cm[r, c]),
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if cm[r, c] > cm.max() / 2 else "black",
                    )
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / f"cm_{model_name}_{dim}.png", dpi=150)
            plt.close(fig)


def plot_loss_curves(model_names: list):
    """Training and val loss curves from logs/{model}_losses.json."""
    for model_name in model_names:
        log_path = LOGS_DIR / f"{model_name}_losses.json"
        if not log_path.exists():
            print(f"  [warn] No loss log found for {model_name}, skipping curve.")
            continue

        with open(log_path) as f:
            history = json.load(f)

        epochs = range(1, len(history["train"]) + 1)
        train_total = [e["total_loss"] for e in history["train"]]
        val_total = [e["total_loss"] for e in history["val"]]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, train_total, label="train")
        ax.plot(epochs, val_total, label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Loss")
        ax.set_title(f"{model_name} — Training Loss Curves")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"loss_curve_{model_name}.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------


def print_summary(all_metrics: dict):
    col_w = 14
    model_names = list(all_metrics.keys())
    print("\n" + "=" * (14 + col_w * len(model_names)))
    print(f"{'Attribute':<14}" + "".join(f"{m:>{col_w}}" for m in model_names))
    print("=" * (14 + col_w * len(model_names)))

    for dim in CLASS_DIMS:
        for metric in ("accuracy", "macro_f1"):
            row = f"{dim + ' ' + metric:<14}"
            for m in model_names:
                row += f"{all_metrics[m][dim][metric]:>{col_w}.4f}"
            print(row)

    print("=" * (14 + col_w * len(model_names)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate vibe models")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["cnn", "clip", "hybrid"],
        default=["cnn", "clip", "hybrid"],
    )
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load val split once
    print("Loading dataset ...")
    _, val_dataset = load_fashionpedia_data(args.data_dir)
    num_workers = 0 if device.type == "mps" else 4
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type != "mps"),
    )
    print(f"Val samples: {len(val_dataset)}")

    all_metrics = {}
    results_by_model = {}

    for model_name in args.models:
        ckpt_path = ROOT / args.checkpoints_dir / model_name / "best.pt"
        if not ckpt_path.exists():
            print(f"  [warn] Checkpoint not found for {model_name}: {ckpt_path}")
            continue

        print(f"\nEvaluating {model_name} ...")
        model = load_model(model_name, ckpt_path, device)
        results = run_inference(model, val_loader, device)
        results_by_model[model_name] = results
        all_metrics[model_name] = compute_metrics(results)

    if not all_metrics:
        print("No models evaluated — check checkpoint paths.")
        return

    # Plots
    print("\nGenerating plots ...")
    plot_accuracy_bars(all_metrics, list(all_metrics.keys()))
    plot_confusion_matrices(results_by_model)

    plot_loss_curves(args.models)
    print(f"Plots saved → {PLOTS_DIR}")

    # Metrics JSON
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved → {METRICS_PATH}")

    # Terminal summary
    print_summary(all_metrics)


if __name__ == "__main__":
    main()
