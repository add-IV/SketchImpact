import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

SPRITE_SIZE = 32

# Precompute coordinate grids once — shared across all Dataset instances
_xs = np.linspace(0, 1, SPRITE_SIZE, dtype=np.float32)
_ys = np.linspace(0, 1, SPRITE_SIZE, dtype=np.float32)
_X_GRID, _Y_GRID = np.meshgrid(_xs, _ys)  # (32, 32) each; _X_GRID[i,j]=col, _Y_GRID[i,j]=row


def _random_symmetry(alpha_i: np.ndarray, alpha_j: np.ndarray, dx: float, dy: float):
    """
    Applies a random element of the dihedral group D4 (4 rotations x optional
    mirror) to both sprites and their offset vector.

    Rotating/reflecting both sprites and (dx, dy) by the same transform is
    just a change of coordinates, so the collision label is unchanged —
    verified empirically against pixel_collision.py's floor/ceil rule.
    """
    if np.random.randint(2):
        alpha_i = np.fliplr(alpha_i)
        alpha_j = np.fliplr(alpha_j)
        dx = -dx
    k = np.random.randint(4)
    if k:
        alpha_i = np.rot90(alpha_i, k)
        alpha_j = np.rot90(alpha_j, k)
        for _ in range(k):
            dx, dy = dy, -dx
    return alpha_i, alpha_j, dx, dy


class CollisionDataset(Dataset):
    def __init__(self, npz_path: Path, split: str = "train", val_fraction: float = 0.1,
                 seed: int = 42, augment: bool | None = None):
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

        # Default: augment during training only, not for validation
        self.augment = (split == "train") if augment is None else augment

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, n: int):
        alpha_i = self.sprites[self.i[n]]   # (32, 32) uint8
        alpha_j = self.sprites[self.j[n]]
        dx, dy = self.dx[n], self.dy[n]

        if self.augment:
            alpha_i, alpha_j, dx, dy = _random_symmetry(alpha_i, alpha_j, dx, dy)

        x = np.empty((6, SPRITE_SIZE, SPRITE_SIZE), dtype=np.float32)
        x[0] = alpha_i / 255.0
        x[1] = alpha_j / 255.0
        x[2] = dx / SPRITE_SIZE            # constant broadcast — offset x
        x[3] = dy / SPRITE_SIZE            # constant broadcast — offset y
        x[4] = _X_GRID                     # column coord grid [0, 1]
        x[5] = _Y_GRID                     # row coord grid    [0, 1]

        return torch.from_numpy(x), torch.tensor(self.label[n])
