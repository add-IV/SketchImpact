import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

SPRITE_SIZE = 32

# Precompute coordinate grids once — shared across all Dataset instances
_xs = np.linspace(0, 1, SPRITE_SIZE, dtype=np.float32)
_ys = np.linspace(0, 1, SPRITE_SIZE, dtype=np.float32)
_X_GRID, _Y_GRID = np.meshgrid(_xs, _ys)  # (32, 32) each; _X_GRID[i,j]=col, _Y_GRID[i,j]=row


class CollisionDataset(Dataset):
    def __init__(self, npz_path: Path, split: str = "train", val_fraction: float = 0.1, seed: int = 42):
        data = np.load(npz_path)
        self.sprites = data["sprites"]   # (M, 32, 32) uint8
        i       = data["i"]              # (N,) uint16
        j       = data["j"]
        dx      = data["dx"]             # (N,) float32
        dy      = data["dy"]
        label   = data["label"]          # (N,) uint8

        # Split by pair key so the same sprite pair never appears in both splits
        pair_keys = np.minimum(i, j)
        unique_keys = np.unique(pair_keys)
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_keys)
        n_val = int(len(unique_keys) * val_fraction)
        val_keys = unique_keys[:n_val]

        mask = np.isin(pair_keys, val_keys)
        idx = np.where(~mask if split == "train" else mask)[0]

        self.i     = i[idx]
        self.j     = j[idx]
        self.dx    = dx[idx]
        self.dy    = dy[idx]
        self.label = label[idx].astype(np.float32)

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, n: int):
        alpha_i = self.sprites[self.i[n]]   # (32, 32) uint8
        alpha_j = self.sprites[self.j[n]]

        x = np.empty((6, SPRITE_SIZE, SPRITE_SIZE), dtype=np.float32)
        x[0] = alpha_i / 255.0
        x[1] = alpha_j / 255.0
        x[2] = self.dx[n] / SPRITE_SIZE    # constant broadcast — offset x
        x[3] = self.dy[n] / SPRITE_SIZE    # constant broadcast — offset y
        x[4] = _X_GRID                     # column coord grid [0, 1]
        x[5] = _Y_GRID                     # row coord grid    [0, 1]

        return torch.from_numpy(x), torch.tensor(self.label[n])
