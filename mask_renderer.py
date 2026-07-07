"""
mask_renderer.py — Spider-Man mask warping and compositing.

render_mask() is the single public entry point.  It aligns the mask PNG to
the detected face via a similarity transform, applies an optional filter to
the mask colours, feathers the alpha edge, and alpha-composites the result
onto the base frame.
"""

import cv2
import numpy as np

from config import (
    SMOOTH,
    SCALE_FACTOR,
    HAIR_SHIFT,
    FEATHER,
    MASK_EYE_L,
    MASK_EYE_R,
    EYE_L,
    EYE_R,
    IDX_FOREHEAD,
    IDX_CHIN,
    ERODE_KERNEL,
)


# EMA-smoothed affine transform — persists across frames to damp landmark jitter
_smoothed_M: np.ndarray | None = None


def reset_transform() -> None:
    """Reset the EMA transform (call when the face is lost between frames)."""
    global _smoothed_M
    _smoothed_M = None


def render_mask(
    base: np.ndarray,
    lm: list,
    mask_img: np.ndarray,
    filter_fn=None,
) -> np.ndarray:
    """
    Warp the Spider-Man mask onto *base*, aligned to the detected face.

    Steps
    -----
    1. Compute a similarity transform from mask eye positions → real eye positions.
    2. Scale up to compensate for the mask's wide eye-lens proportions.
    3. Shift upward along the face axis to cover the hairline.
    4. EMA-smooth the transform to damp per-frame landmark jitter.
    5. Warp the mask; optionally apply *filter_fn* to only the non-transparent
       bounding box (avoids running an expensive filter over empty pixels).
    6. Feather the alpha edge (erode → Gaussian blur) for a natural skin blend.

    Parameters
    ----------
    base      : BGR uint8 frame to composite the mask onto.
    lm        : 478-point face landmark list (NormalizedLandmark objects).
    mask_img  : BGRA uint8 Spider-Man mask image.
    filter_fn : Optional callable ``(np.ndarray) -> np.ndarray`` applied to
                the mask colours; receives and returns a BGR uint8 crop.

    Returns
    -------
    np.ndarray  BGR uint8 — *base* with the mask composited on top.
    """
    global _smoothed_M
    h, w = base.shape[:2]
    pts  = np.array([[p.x * w, p.y * h] for p in lm], np.float32)

    # Average inner/outer corner landmarks for a stable eye centre
    eye_l = (pts[EYE_L[0]] + pts[EYE_L[1]]) / 2
    eye_r = (pts[EYE_R[0]] + pts[EYE_R[1]]) / 2

    # Similarity transform: align mask eye-holes to real eye positions
    M, _ = cv2.estimateAffinePartial2D(
        np.float32([MASK_EYE_L, MASK_EYE_R]),
        np.float32([eye_l, eye_r]),
    )
    if M is None:
        return base

    # Scale up around the eye midpoint so the mask fills the full face.
    # Spider-Man's lenses are wider than real eyes, so the naive eye-to-eye
    # alignment leaves the mask about half the correct size.
    cx, cy = (eye_l + eye_r) / 2
    M[0, 2] += (1 - SCALE_FACTOR) * (M[0, 0] * cx + M[0, 1] * cy)
    M[1, 2] += (1 - SCALE_FACTOR) * (M[1, 0] * cx + M[1, 1] * cy)
    M[:, :2] *= SCALE_FACTOR

    # Shift the mask upward along the face axis to cover the hairline
    up     = pts[IDX_FOREHEAD] - pts[IDX_CHIN]
    face_h = np.linalg.norm(up)
    up    /= face_h + 1e-6
    M[:, 2] += up * face_h * HAIR_SHIFT

    # EMA-smooth the transform to reduce jitter from noisy landmarks
    if _smoothed_M is None:
        _smoothed_M = M.copy()
    else:
        _smoothed_M[:] = SMOOTH * M + (1 - SMOOTH) * _smoothed_M

    # Warp the mask to the current frame dimensions
    warped = cv2.warpAffine(
        mask_img, _smoothed_M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    mask_bgr = warped[:, :, :3].copy()

    if filter_fn is not None:
        # Filter only the tight bounding box of non-transparent pixels to avoid
        # running an expensive operation over ~80 % black/empty pixels.
        nz = cv2.findNonZero(warped[:, :, 3])
        if nz is not None:
            x, y, bw, bh = cv2.boundingRect(nz)
            # Small padding so filter edge effects don't clip at the mask boundary
            x1, y1 = max(0, x - 4),       max(0, y - 4)
            x2, y2 = min(w, x + bw + 4),  min(h, y + bh + 4)
            mask_bgr[y1:y2, x1:x2] = filter_fn(mask_bgr[y1:y2, x1:x2])

    # Feather the alpha: erode (pulls edge inward) then Gaussian-blur (smooth falloff)
    soft_a = cv2.erode(warped[:, :, 3], ERODE_KERNEL, iterations=2)
    soft_a = cv2.GaussianBlur(soft_a, (FEATHER | 1, FEATHER | 1), 0)
    alpha  = soft_a.astype(np.float32)[:, :, None] / 255.0

    return (mask_bgr * alpha + base * (1 - alpha)).astype(np.uint8)
