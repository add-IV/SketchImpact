import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_loader import CollisionDataset
from model import CollisionCNN

ROOT = Path(__file__).parent


def _curve(logits: np.ndarray, labels: np.ndarray, n_points: int = 300):
    """Return (thresholds, recalls, precisions, fracs_skipped) arrays."""
    thresholds = np.linspace(logits.min(), logits.max(), n_points)
    n_pos = labels.sum()
    recalls, precisions, fracs_skipped = [], [], []
    for t in thresholds:
        preds = (logits > t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        recalls.append(tp / n_pos if n_pos > 0 else 0.0)
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        fracs_skipped.append((preds == 0).sum() / len(labels))
    return thresholds, np.array(recalls), np.array(precisions), np.array(fracs_skipped)


def _threshold_for_recall(thresholds, recalls, target: float) -> int:
    """Index of the highest threshold that still achieves >= target recall."""
    candidates = np.where(recalls >= target)[0]
    return int(candidates[-1]) if len(candidates) > 0 else -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "model_best.pt")
    parser.add_argument("--dataset",    type=Path, default=ROOT / "dataset.npz")
    parser.add_argument("--workers",    type=int,  default=4)
    parser.add_argument("--no-export",  action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = CollisionCNN().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Checkpoint: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    val_ds = CollisionDataset(args.dataset, split="val")
    val_dl = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=args.workers)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, label in val_dl:
            all_logits.append(model(x.to(device)).cpu())
            all_labels.append(label)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    n, n_pos = len(labels), int(labels.sum())
    print(f"Val set: {n:,} samples, {n_pos:,} positive ({n_pos/n*100:.1f}%)\n")

    thresholds, recalls, precisions, fracs_skipped = _curve(logits, labels)

    # --- operating point table ---
    print(f"{'Target recall':>14} {'Threshold':>10} {'Actual recall':>14} {'Precision':>10} {'Skipped':>8}")
    print("-" * 62)
    for target in [0.999, 0.995, 0.99, 0.98, 0.95, 0.90]:
        idx = _threshold_for_recall(thresholds, recalls, target)
        if idx < 0:
            print(f"{target:>14.1%}  {'—':>10}  not achievable")
            continue
        print(f"{target:>14.1%} {thresholds[idx]:>10.3f} {recalls[idx]:>14.4f} "
              f"{precisions[idx]:>10.4f} {fracs_skipped[idx]:>8.3f}")

    # --- baselines ---
    print(f"\nBaselines (reference):")
    print(f"  Always-predict-collision : recall=1.000  precision={n_pos/n:.3f}  skipped=0.000")
    print(f"  Never-predict-collision  : recall=0.000  skipped=1.000")
    print(f"  Exact numpy check        : recall=1.000  precision=1.000  skipped=0.000  (no CNN needed in Python)")

    # --- confusion matrix at 99.9% recall operating point ---
    idx = _threshold_for_recall(thresholds, recalls, 0.999)
    if idx >= 0:
        t = thresholds[idx]
        preds = (logits > t).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        print(f"\nConfusion matrix at 99.9% recall (threshold={t:.3f}):")
        print(f"  TP={tp:>8,}  FP={fp:>8,}")
        print(f"  FN={fn:>8,}  TN={tn:>8,}")
        print(f"  Checks skipped: {tn+fn:,} / {n:,} ({(tn+fn)/n*100:.1f}%)")

    # --- ONNX export ---
    if not args.no_export:
        out = ROOT / "model.onnx"
        dummy = torch.zeros(1, 6, 32, 32, device=device)
        torch.onnx.export(
            model, dummy, str(out),
            opset_version=17,
            input_names=["input"],
            output_names=["logit"],
            dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
        )
        print(f"\nONNX model saved → {out}")


if __name__ == "__main__":
    main()
