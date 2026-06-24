import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_loader import CollisionDataset
from model import CollisionCNN

ROOT = Path(__file__).parent


def _metrics(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, float, float]:
    preds = (logits > 0).long()
    labels = labels.long()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    acc  = (tp + tn) / len(labels)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return acc, prec, rec, f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--workers",    type=int,   default=4)
    parser.add_argument("--dataset",    type=Path,  default=ROOT / "dataset.npz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = CollisionDataset(args.dataset, split="train")
    val_ds   = CollisionDataset(args.dataset, split="val")
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    model     = CollisionCNN().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn   = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    ckpt_path = ROOT / "model_best.pt"

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        tr_loss = 0.0
        tr_logits, tr_labels = [], []
        for x, label in train_dl:
            x, label = x.to(device), label.to(device)
            optimiser.zero_grad()
            logit = model(x)
            loss  = loss_fn(logit, label)
            loss.backward()
            optimiser.step()
            tr_loss += loss.item() * len(label)
            tr_logits.append(logit.detach().cpu())
            tr_labels.append(label.cpu())
        tr_loss /= len(train_ds)
        tr_acc, tr_p, tr_r, tr_f1 = _metrics(torch.cat(tr_logits), torch.cat(tr_labels))

        # --- val ---
        model.eval()
        vl_loss = 0.0
        vl_logits, vl_labels = [], []
        with torch.no_grad():
            for x, label in val_dl:
                x, label = x.to(device), label.to(device)
                logit = model(x)
                vl_loss += loss_fn(logit, label).item() * len(label)
                vl_logits.append(logit.cpu())
                vl_labels.append(label.cpu())
        vl_loss /= len(val_ds)
        vl_acc, vl_p, vl_r, vl_f1 = _metrics(torch.cat(vl_logits), torch.cat(vl_labels))

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={tr_loss:.4f} acc={tr_acc:.3f} P={tr_p:.3f} R={tr_r:.3f} F1={tr_f1:.3f} | "
            f"val   loss={vl_loss:.4f} acc={vl_acc:.3f} P={vl_p:.3f} R={vl_r:.3f} F1={vl_f1:.3f}"
        )

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_loss": vl_loss}, ckpt_path)
            print(f"  → checkpoint saved (val_loss={vl_loss:.4f})")

    print(f"\nBest val loss: {best_val_loss:.4f}  checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
