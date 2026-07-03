# Collision Classifier — Model Design

## Task

Binary classification: given two 32×32 binary alpha masks and a relative offset (dx, dy),
predict whether the sprites collide under the GDScript floor/ceil pixel collision logic.

## Dataset

- `dataset.npz`: 713 sprites (32×32 uint8), ~1.9M pairs, 25% positive rate
- Labels from exact FFT cross-correlation (`collision_label_grid` in `pixel_collision.py`)
- Train/val split by pair key `min(i, j)` — same pair never appears in both splits

## Input Representation

Shape: **(6, 32, 32)** float32

| Channel | Content | Range |
|---------|---------|-------|
| 0 | alpha_i / 255 | {0, 1} |
| 1 | alpha_j / 255 | {0, 1} |
| 2 | dx / 32 broadcast | ~[−1, 1] |
| 3 | dy / 32 broadcast | ~[−1, 1] |
| 4 | x coordinate grid (linspace 0→1) | [0, 1] |
| 5 | y coordinate grid (linspace 0→1) | [0, 1] |

Channels 2–3 give the network the relative offset at every layer (CoordConv-style constant
injection). Channels 4–5 give spatial context so the network can reason about *which* pixels
in alpha_i are geometrically near *which* pixels in alpha_j given the offset — e.g.
`x_grid − dx_const` approximates where a pixel in sprite_i maps to in sprite_j's frame.

No bilinear shift is applied. The network must learn collision geometry from first principles,
which is the genuine learning contribution — a bilinear composite would pre-encode the answer.

## Data Augmentation

`dataset_loader.py` applies a random dihedral (D4) transform — 0/90/180/270° rotation, each
with an optional mirror — to `alpha_i`, `alpha_j`, and `(dx, dy)` together, on the fly in
`__getitem__`. Rotating/reflecting both sprites and their offset by the same transform is just
a change of coordinates, so the collision label is unchanged (verified empirically against the
floor/ceil rule in `pixel_collision.py`, 0 mismatches over 2000 random trials). This gives an
effective 8x increase in training variety without inflating `dataset.npz`. Enabled by default
for `split="train"`, disabled for `split="val"` so validation metrics stay stable across runs.

## Architecture

```
Input (B, 6, 32, 32)
        │
Conv(6→16, 3×3, pad=1) + BN + ReLU        → (B, 16, 32, 32)
        │
Conv(16→32, 3×3, pad=1) + BN + ReLU
+ MaxPool(2)                               → (B, 32, 16, 16)
        │
Conv(32→64, 3×3, pad=1) + BN + ReLU
+ MaxPool(2)                               → (B, 64, 8, 8)
        │
Flatten                                    → (B, 4096)
        │
Linear(4096 → 64) + ReLU
        │
Linear(64 → 1)
        │
     logit
```

**Output:** raw logit. No sigmoid in `forward` — use `BCEWithLogitsLoss`.

**~286K parameters**, dominated by `Linear(4096→64)`. If this is too large, add
`MaxPool(2)` after the third conv block to reduce flatten dim to 1024 (~90K parameters).

### Design decisions

- **Flatten, not Global Average Pool** — collision is a sparse local signal. GAP averages
  the feature map across all spatial locations, diluting a corner-touch to near zero.
  Flatten lets the FC layer detect any locally strong activation.
- **BN after every conv** — stabilises training; needed with binary input (low variance).
- **3 conv blocks** — sufficient receptive field for 32×32 input; deeper adds no benefit.

## Training

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam, lr=1e-3, weight_decay=1e-4 |
| Loss | BCEWithLogitsLoss |
| Batch size | 512 |
| Epochs | 20 |
| Checkpoint | best val-loss → `model_best.pt` |

Log per epoch: loss, accuracy, precision, recall, F1.

## Evaluation

The decision threshold on the logit is the primary operating parameter — lowering it
increases recall at the cost of fewer skipped checks.

Report:
1. **Recall vs fraction-skipped curve** across thresholds — pick operating point at ≥99.9% recall
2. **F1 at threshold=0** for reference
3. **Exact numpy baseline** — `(alpha_i * alpha_j_shifted).any()` — trivially fast in Python;
   shows the CNN's value is in *what it learned*, not in raw speed vs numpy
4. **Confusion matrix** at chosen operating point

Export best checkpoint to ONNX (opset 17) for potential Godot integration:
```python
torch.onnx.export(model, torch.zeros(1, 6, 32, 32), "model.onnx", opset_version=17)
```
