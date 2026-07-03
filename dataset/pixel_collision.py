"""
Numpy reimplementation of billiard_table.gd collision logic.

All functions work in sprite-pixel space (not screen space).
dx = (pos_j.x - pos_i.x) / display_scale
dy = (pos_j.y - pos_i.y) / display_scale
"""

import numpy as np

SPRITE_SIZE = 32


def tight_aabb(alpha: np.ndarray) -> tuple[int, int, int, int] | None:
    """Returns (min_x, min_y, max_x, max_y), or None if sprite is empty."""
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def pixel_collision(alpha_i: np.ndarray, alpha_j: np.ndarray, dx: float, dy: float) -> bool:
    """
    Returns True if the two sprites collide at offset (dx, dy).
    Matches _pixel_collision / _find_overlapping_j_pixel in GDScript exactly,
    including the floor/ceil sub-pixel candidate logic.
    Vectorised over opaque pixels of alpha_i for speed.
    """
    s = SPRITE_SIZE
    ys_i, xs_i = np.where(alpha_i > 0)
    if len(xs_i) == 0:
        return False

    pxb_f = xs_i.astype(np.float64) - dx
    pyb_f = ys_i.astype(np.float64) - dy

    for rx in (np.floor(pxb_f).astype(np.int32), np.ceil(pxb_f).astype(np.int32)):
        for ry in (np.floor(pyb_f).astype(np.int32), np.ceil(pyb_f).astype(np.int32)):
            mask = (rx >= 0) & (rx < s) & (ry >= 0) & (ry < s)
            if mask.any() and (alpha_j[ry[mask], rx[mask]] > 0).any():
                return True
    return False


def bilinear_shift(alpha: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Returns alpha shifted by (dx, dy) using bilinear interpolation.
    Output[y, x] = interpolated alpha at source (y - dy, x - dx).
    Out-of-bounds samples are 0. Returns float32 in the same range as input.
    """
    return bilinear_shift_batch(alpha, np.array([[dx, dy]], dtype=np.float32))[0]


def bilinear_shift_batch(alpha: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """
    Vectorised bilinear shift for N offsets at once.
    alpha   : (S, S) uint8
    offsets : (N, 2) float32  — columns are [dx, dy]
    returns : (N, S, S) float32
    """
    s = alpha.shape[0]
    a = alpha.astype(np.float32)           # (S, S)
    n = len(offsets)

    # Pixel coordinate grids: (1, S, S) broadcast with (N, 1, 1)
    ys, xs = np.mgrid[0:s, 0:s]           # (S, S) each
    xs = xs[None].astype(np.float32)      # (1, S, S)
    ys = ys[None].astype(np.float32)

    dx = offsets[:, 0].reshape(n, 1, 1)   # (N, 1, 1)
    dy = offsets[:, 1].reshape(n, 1, 1)

    src_x = xs - dx                        # (N, S, S)
    src_y = ys - dy

    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    fx = (src_x - x0).astype(np.float32)
    fy = (src_y - y0).astype(np.float32)

    def fetch(r, c):
        valid = (r >= 0) & (r < s) & (c >= 0) & (c < s)
        r_s = np.clip(r, 0, s - 1)
        c_s = np.clip(c, 0, s - 1)
        return np.where(valid, a[r_s, c_s], 0.0)

    return (
        fetch(y0, x0) * (1 - fx) * (1 - fy)
        + fetch(y0, x1) * fx       * (1 - fy)
        + fetch(y1, x0) * (1 - fx) * fy
        + fetch(y1, x1) * fx       * fy
    ).astype(np.float32)


def collision_label_grid(alpha_i: np.ndarray, alpha_j: np.ndarray) -> np.ndarray:
    """
    Compute collision labels for ALL offsets at 0.5-pixel step in one shot.

    Uses FFT cross-correlation to get integer-offset labels in O(S² log S),
    then derives half-pixel labels from the floor/ceil rule:
      collision at (dx+0.5, dy) = C[dy, dx] OR C[dy, dx+1]
      collision at (dx, dy+0.5) = C[dy, dx] OR C[dy+1, dx]
      collision at (dx+0.5, dy+0.5) = OR of all four neighbours

    Returns a (2*(2S-1)-1, 2*(2S-1)-1) = (4S-3, 4S-3) bool array — shape (125, 125)
    for S=32 — where entry [oy, ox] corresponds to offset:
      dx = (ox - (2S-2)) * 0.5
      dy = (oy - (2S-2)) * 0.5
    i.e. the centre index (2S-2, 2S-2) = (62, 62) maps to (dx=0, dy=0).
    """
    s = SPRITE_SIZE
    mask_i = (alpha_i > 0).astype(np.float32)
    mask_j = (alpha_j > 0).astype(np.float32)

    # FFT cross-correlation: C_int[oy, ox] = number of overlapping pixel pairs
    # at integer offset (dx, dy) = (ox-(s-1), oy-(s-1)).  Shape (2s-1, 2s-1).
    # irfft2 returns zero-lag at index [0, 0] with negative lags wrapped to the
    # end of the array; fftshift re-centers zero-lag at [s-1, s-1] to match
    # that indexing convention (verified against pixel_collision: 0 mismatches
    # over 30 random real-sprite pairs x 33x33 offsets).
    fft_s = 2 * s - 1
    Fi = np.fft.rfft2(mask_i, s=(fft_s, fft_s))
    Fj = np.fft.rfft2(mask_j, s=(fft_s, fft_s))
    C_int = np.fft.irfft2(Fi * np.conj(Fj), s=(fft_s, fft_s))
    C_int = np.fft.fftshift(C_int)
    C_bool = np.round(C_int) > 0  # shape (2s-1, 2s-1) = (63, 63)

    # Pad by 1 so that boundary half-pixel offsets (e.g. dx=±31.5) map to valid
    # indices even though their ceil/floor candidate falls outside the sprite grid.
    C_p = np.pad(C_bool, 1, constant_values=False)  # shape (65, 65)

    # Expand to half-pixel grid: shape (2*65-1, 2*65-1) = (129, 129)
    # Centre index = 64, offset dx maps to col = round(dx/0.5) + 64
    H, W = C_p.shape
    out = np.zeros((2 * H - 1, 2 * W - 1), dtype=bool)
    out[::2, ::2] = C_p                                       # integer offsets
    out[::2, 1::2] = C_p[:, :-1] | C_p[:, 1:]               # dx half-step
    out[1::2, ::2] = C_p[:-1, :] | C_p[1:, :]               # dy half-step
    out[1::2, 1::2] = (                                       # both half-step
        C_p[:-1, :-1] | C_p[:-1, 1:] | C_p[1:, :-1] | C_p[1:, 1:]
    )
    return out


def aabb_overlap_range(
    alpha_i: np.ndarray, alpha_j: np.ndarray
) -> tuple[float, float, float, float] | None:
    """
    Returns (dx_lo, dx_hi, dy_lo, dy_hi) — the open interval of offsets where
    the tight AABBs of the two sprites overlap, matching GDScript Rect2.intersects.
    Returns None if either sprite is empty or the AABBs can never overlap.
    """
    bb_i = tight_aabb(alpha_i)
    bb_j = tight_aabb(alpha_j)
    if bb_i is None or bb_j is None:
        return None

    half = SPRITE_SIZE / 2.0

    # AABB edges in sprite-pixel space, relative to sprite centre
    ix0, iy0 = bb_i[0] - half, bb_i[1] - half
    ix1, iy1 = bb_i[2] + 1 - half, bb_i[3] + 1 - half
    jx0, jy0 = bb_j[0] - half, bb_j[1] - half
    jx1, jy1 = bb_j[2] + 1 - half, bb_j[3] + 1 - half

    # Overlap iff ix0 < dx+jx1 AND dx+jx0 < ix1 (strict — no border touching)
    dx_lo, dx_hi = ix0 - jx1, ix1 - jx0
    dy_lo, dy_hi = iy0 - jy1, iy1 - jy0

    if dx_lo >= dx_hi or dy_lo >= dy_hi:
        return None

    return dx_lo, dx_hi, dy_lo, dy_hi
