"""
filters.py — Image filter functions and the finger-count → filter map.

Each function takes a BGR uint8 ndarray and returns a transformed BGR uint8
ndarray of the same shape.  They are applied to the full camera frame (inside
the hand window) and also to the Spider-Man mask colours for a unified look.

Performance notes:
  - Oil Painting and Sketch run at half resolution internally to cut pixel
    count by 4×, then upscale back.  Visual quality is virtually unchanged
    because both filters destroy fine detail anyway.
  - Thermal and Negative are already O(n) pixel ops and stay full-res.
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Filter implementations
# ─────────────────────────────────────────────────────────────────────────────

def oil_painting(img: np.ndarray) -> np.ndarray:
    """Bilateral-smoothed frame with edge outlines and boosted saturation/brightness.

    Internally downscales to half resolution before running the 3 bilateral
    passes and edge detection, then upscales.  This is 4× faster with no
    perceptible quality loss (bilateral + edges destroy fine detail anyway).
    """
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)

    s = small
    for _ in range(3):
        s = cv2.bilateralFilter(s, d=9, sigmaColor=150, sigmaSpace=150)
    gray  = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=7, C=3
    )
    s   = cv2.bitwise_and(s, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.5, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.4, 0, 255)
    out = cv2.convertScaleAbs(
        cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), alpha=1.3, beta=15
    )
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)


def thermal(img: np.ndarray) -> np.ndarray:
    """JET colormap on grayscale luminance — thermal-camera look."""
    return cv2.applyColorMap(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)


def sketch(img: np.ndarray) -> np.ndarray:
    """Colored pencil sketch with boosted saturation and contrast.

    Runs pencilSketch at half resolution to cut cost ~4×, then upscales.
    The sketch effect is inherently low-frequency, so this has minimal
    visible impact.
    """
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)

    _, color = cv2.pencilSketch(small, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.8, 0, 255)
    out = cv2.convertScaleAbs(
        cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), alpha=1.2, beta=10
    )
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)


def negative(img: np.ndarray) -> np.ndarray:
    """Invert every channel — classic photographic negative."""
    return 255 - img


# ─────────────────────────────────────────────────────────────────────────────
# Filter registry  (finger count → (function, display label))
# ─────────────────────────────────────────────────────────────────────────────

FILTERS: dict = {
    1: (oil_painting, "Oil Painting"),
    2: (thermal,      "Thermal"),
    3: (sketch,       "Sketch"),
    4: (negative,     "Negative"),
}
