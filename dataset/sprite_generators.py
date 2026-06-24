"""
Synthetic 32×32 binary sprite generators.

Each generator returns a (SPRITE_SIZE, SPRITE_SIZE) uint8 array where
non-zero pixels are opaque (255) and zero pixels are transparent, matching
the alpha-channel convention used in billiard_table.gd.
"""

import numpy as np
from PIL import Image, ImageDraw

SPRITE_SIZE = 32


def _blob(size: int, rng: np.random.Generator) -> np.ndarray:
    """Organic irregular blob grown from a random seed via random-walk expansion."""
    alpha = np.zeros((size, size), dtype=np.uint8)
    cx = int(rng.integers(size // 4, 3 * size // 4))
    cy = int(rng.integers(size // 4, 3 * size // 4))
    alpha[cy, cx] = 255

    target = int(rng.integers(size * size // 8, size * size // 2))
    filled = 1
    while filled < target:
        ys, xs = np.where(alpha > 0)
        idx = int(rng.integers(len(xs)))
        nx = int(xs[idx]) + int(rng.integers(-1, 2))
        ny = int(ys[idx]) + int(rng.integers(-1, 2))
        if 0 <= nx < size and 0 <= ny < size and alpha[ny, nx] == 0:
            alpha[ny, nx] = 255
            filled += 1

    return alpha


def _convex_polygon(size: int, rng: np.random.Generator) -> np.ndarray:
    """Random convex polygon with 3–8 vertices placed in polar coordinates."""
    n = int(rng.integers(3, 9))
    angles = np.sort(rng.uniform(0, 2 * np.pi, n))
    radii = rng.uniform(size * 0.1, size * 0.4, n)
    cx = size / 2 + rng.uniform(-size * 0.1, size * 0.1)
    cy = size / 2 + rng.uniform(-size * 0.1, size * 0.1)
    pts = [
        (float(cx + r * np.cos(a)), float(cy + r * np.sin(a)))
        for a, r in zip(angles, radii)
    ]
    img = Image.new("L", (size, size), 0)
    ImageDraw.Draw(img).polygon(pts, fill=255)
    return np.array(img, dtype=np.uint8)


def _primitive(size: int, rng: np.random.Generator) -> np.ndarray:
    """Simple geometric shape: filled circle, rectangle, cross, or L."""
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    m = size // 6
    shape = rng.choice(["circle", "rect", "cross", "L"])

    if shape == "circle":
        r = int(rng.integers(size // 6, size // 3))
        cx = int(rng.integers(r + m, size - r - m))
        cy = int(rng.integers(r + m, size - r - m))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)

    elif shape == "rect":
        x0, y0 = int(rng.integers(m, size // 2)), int(rng.integers(m, size // 2))
        x1, y1 = int(rng.integers(size // 2, size - m)), int(rng.integers(size // 2, size - m))
        draw.rectangle([x0, y0, x1, y1], fill=255)

    elif shape == "cross":
        w = int(rng.integers(2, 6))
        cx, cy = size // 2, size // 2
        arm = int(rng.integers(size // 5, size // 3))
        draw.rectangle([cx - w, cy - arm, cx + w, cy + arm], fill=255)
        draw.rectangle([cx - arm, cy - w, cx + arm, cy + w], fill=255)

    else:  # L
        w = int(rng.integers(2, 6))
        arm = int(rng.integers(size // 4, size // 2))
        x0, y0 = int(rng.integers(m, size // 2)), int(rng.integers(m, size // 2))
        draw.rectangle([x0, y0, x0 + w, y0 + arm], fill=255)
        draw.rectangle([x0, y0 + arm - w, x0 + arm, y0 + arm], fill=255)

    return np.array(img, dtype=np.uint8)


def _noise(size: int, rng: np.random.Generator) -> np.ndarray:
    """Smoothed random noise thresholded into a binary sprite."""
    raw = rng.random((size, size)).astype(np.float32)
    sigma = rng.uniform(1.0, 3.0)

    # Simple repeated box-blur approximates a Gaussian (no scipy needed)
    kernel = int(np.ceil(sigma * 2)) | 1  # odd kernel size
    pad = kernel // 2
    blurred = raw
    for _ in range(3):
        padded = np.pad(blurred, pad, mode="edge")
        blurred = np.lib.stride_tricks.sliding_window_view(
            padded, (kernel, kernel)
        ).mean(axis=(-2, -1))

    lo, hi = blurred.min(), blurred.max()
    if hi - lo < 1e-6:
        return np.zeros((size, size), dtype=np.uint8)

    normalized = (blurred - lo) / (hi - lo)
    threshold = float(rng.uniform(0.3, 0.7))
    return (normalized > threshold).astype(np.uint8) * 255


_GENERATORS = [_blob, _convex_polygon, _primitive, _noise]


def generate_synthetic_sprites(
    n: int, size: int = SPRITE_SIZE, seed: int = 42
) -> list[np.ndarray]:
    """
    Generate n synthetic sprites cycling evenly through all four generator types.
    Invisible sprites (all-transparent) are skipped and regenerated.
    """
    rng = np.random.default_rng(seed)
    sprites: list[np.ndarray] = []
    attempts = 0
    while len(sprites) < n and attempts < n * 4:
        gen = _GENERATORS[attempts % len(_GENERATORS)]
        alpha = gen(size, rng)
        if alpha.max() > 0:
            sprites.append(alpha)
        attempts += 1
    return sprites
