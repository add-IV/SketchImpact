"""
Generate the SketchImpact collision dataset.

For each sampled sprite pair the script:
  1. Computes all collision labels in one FFT cross-correlation.
  2. Keeps all positive samples + neg_ratio x as many randomly chosen negatives,
     capped at max_samples_per_pair total.
  3. Records sprite indices + offsets + labels (no composites computed here).

The bilinear shift composite is computed on-the-fly in the DataLoader at
training time, keeping this file small and generation fast.

Output dataset.npz keys
  sprites  (M, S, S)  uint8   -- all unique sprites
  i        (N,)        uint16  -- index into sprites for sprite i
  j        (N,)        uint16  -- index into sprites for sprite j
  dx       (N,)        float32 -- offset in sprite-pixel space
  dy       (N,)        float32
  label    (N,)        uint8   -- 1 = collision, 0 = no collision

Usage
  python dataset/generate_dataset.py
  python dataset/generate_dataset.py --max_pairs 500 --max_samples_per_pair 200
"""

import argparse
import sys
from functools import partial
from pathlib import Path

import builtins

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from pixel_collision import SPRITE_SIZE, aabb_overlap_range, collision_label_grid
from sprite_generators import generate_synthetic_sprites

print = partial(builtins.print, flush=True)

STEP = 0.5
_GRID_CENTER = 2 * SPRITE_SIZE  # index 64 -> offset (0, 0) in the 129x129 label grid


# ---------------------------------------------------------------------------
# Sprite loading
# ---------------------------------------------------------------------------

def load_real_sprites(sprites_dir: Path) -> list[np.ndarray]:
    sprites = []
    for p in sorted(sprites_dir.glob("*.png")):
        img = Image.open(p).convert("RGBA")
        if img.size != (SPRITE_SIZE, SPRITE_SIZE):
            img = img.resize((SPRITE_SIZE, SPRITE_SIZE), Image.NEAREST)
        alpha = np.array(img)[:, :, 3]
        if alpha.max() > 0:
            sprites.append(alpha)
    return sprites


# ---------------------------------------------------------------------------
# Per-pair: fast label lookup via FFT grid
# ---------------------------------------------------------------------------

def offsets_and_labels(
    alpha_i: np.ndarray, alpha_j: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Returns (offsets, labels) covering all 0.5-step (dx, dy) within the
    AABB-overlap range. Labels come from a single FFT cross-correlation.
    offsets : (K, 2) float32
    labels  : (K,)  bool
    """
    r = aabb_overlap_range(alpha_i, alpha_j)
    if r is None:
        return None

    dx_lo, dx_hi, dy_lo, dy_hi = r
    dxs = np.arange(dx_lo + STEP, dx_hi, STEP, dtype=np.float32)
    dys = np.arange(dy_lo + STEP, dy_hi, STEP, dtype=np.float32)
    if len(dxs) == 0 or len(dys) == 0:
        return None

    grid_dx, grid_dy = np.meshgrid(dxs, dys)
    offsets = np.column_stack([grid_dx.ravel(), grid_dy.ravel()]).astype(np.float32)

    cx = np.round(offsets[:, 0] / STEP).astype(int) + _GRID_CENTER
    cy = np.round(offsets[:, 1] / STEP).astype(int) + _GRID_CENTER

    grid = collision_label_grid(alpha_i, alpha_j)
    labels = grid[cy, cx]
    return offsets, labels


def process_pair(
    alpha_i: np.ndarray,
    alpha_j: np.ndarray,
    idx_i: int,
    idx_j: int,
    rng: np.random.Generator,
    neg_ratio: int,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Returns (i_arr, j_arr, dxs, dys, labels) or None if no positives exist.
    """
    result = offsets_and_labels(alpha_i, alpha_j)
    if result is None:
        return None
    offsets, labels = result

    pos_idx = np.where(labels)[0]
    neg_idx = np.where(~labels)[0]
    if len(pos_idx) == 0:
        return None

    n_pos = min(len(pos_idx), max_samples // (1 + neg_ratio))
    n_neg = min(len(neg_idx), n_pos * neg_ratio)

    chosen_pos = rng.choice(pos_idx, size=n_pos, replace=False)
    chosen_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    chosen = np.concatenate([chosen_pos, chosen_neg])

    n = len(chosen)
    return (
        np.full(n, idx_i, dtype=np.uint16),
        np.full(n, idx_j, dtype=np.uint16),
        offsets[chosen, 0],
        offsets[chosen, 1],
        np.where(labels[chosen], np.uint8(1), np.uint8(0)),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SketchImpact collision dataset")
    parser.add_argument("--sprites_dir", type=Path, default=ROOT / "sprites")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset" / "dataset.npz")
    parser.add_argument("--n_synthetic", type=int, default=500)
    parser.add_argument("--neg_ratio", type=int, default=3)
    parser.add_argument("--max_pairs", type=int, default=5000)
    parser.add_argument("--max_samples_per_pair", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("Loading real sprites...")
    real = load_real_sprites(args.sprites_dir)
    print(f"  {len(real)} real sprites loaded")

    print(f"Generating {args.n_synthetic} synthetic sprites...")
    synthetic = generate_synthetic_sprites(args.n_synthetic, seed=args.seed)
    print(f"  {len(synthetic)} synthetic sprites generated")

    sprites = real + synthetic
    n_sprites = len(sprites)
    sprites_arr = np.stack(sprites).astype(np.uint8)  # (M, S, S)
    print(f"Total sprites: {n_sprites}  |  Possible pairs: {n_sprites*(n_sprites-1)//2}")

    pairs = [(i, j) for i in range(n_sprites) for j in range(i + 1, n_sprites)]
    rng.shuffle(pairs)
    pairs = pairs[: args.max_pairs]
    print(f"Processing {len(pairs)} pairs  "
          f"(max {args.max_samples_per_pair} samples/pair, neg_ratio={args.neg_ratio})...\n")

    all_i, all_j, all_dx, all_dy, all_labels = [], [], [], [], []
    pairs_with_hits = 0
    n_samples_so_far = 0

    bar = tqdm(pairs, unit="pair", dynamic_ncols=True)
    for idx_i, idx_j in bar:
        result = process_pair(
            sprites[idx_i], sprites[idx_j],
            idx_i, idx_j,
            rng, args.neg_ratio, args.max_samples_per_pair,
        )
        if result is None:
            continue

        i_arr, j_arr, dxs, dys, labels = result
        all_i.append(i_arr)
        all_j.append(j_arr)
        all_dx.append(dxs)
        all_dy.append(dys)
        all_labels.append(labels)
        pairs_with_hits += 1
        n_samples_so_far += len(labels)
        bar.set_postfix(samples=n_samples_so_far, hits=pairs_with_hits)

    if not all_labels:
        print("No collision samples found.")
        return

    i_arr = np.concatenate(all_i)
    j_arr = np.concatenate(all_j)
    dx_arr = np.concatenate(all_dx)
    dy_arr = np.concatenate(all_dy)
    labels_arr = np.concatenate(all_labels)

    n_pos = int(labels_arr.sum())
    n_total = len(labels_arr)
    size_mb = (sprites_arr.nbytes + i_arr.nbytes + j_arr.nbytes
               + dx_arr.nbytes + dy_arr.nbytes + labels_arr.nbytes) / 1e6

    print(f"\nDataset summary")
    print(f"  Sprites  : {n_sprites}  shape={sprites_arr.shape}")
    print(f"  Samples  : {n_total:,}  ({n_pos/n_total*100:.1f}% positive)")
    print(f"  Est. size: {size_mb:.1f} MB")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sprites=sprites_arr,
        i=i_arr, j=j_arr,
        dx=dx_arr, dy=dy_arr,
        label=labels_arr,
    )
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
