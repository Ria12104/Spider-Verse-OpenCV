"""
filters.py — Image filter functions and the finger-count → filter map.

Each function takes a BGR uint8 ndarray and returns a transformed BGR uint8
ndarray of the same shape.  They are applied to the full camera frame (inside
the hand window) and also to the Spider-Man mask colours for a unified look.
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Filter implementations
# ─────────────────────────────────────────────────────────────────────────────

def oil_painting(img: np.ndarray) -> np.ndarray:
    """Bilateral-smoothed frame with edge outlines and boosted saturation/brightness."""
    s = img
    for _ in range(3):      # 3 passes: good quality without killing frame rate
        s = cv2.bilateralFilter(s, d=9, sigmaColor=150, sigmaSpace=150)
    gray  = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=7, C=3
    )
    s   = cv2.bitwise_and(s, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.5, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.4, 0, 255)
    return cv2.convertScaleAbs(
        cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), alpha=1.3, beta=15
    )


def thermal(img: np.ndarray) -> np.ndarray:
    """JET colormap on grayscale luminance — thermal-camera look."""
    return cv2.applyColorMap(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)


def sketch(img: np.ndarray) -> np.ndarray:
    """Colored pencil sketch with boosted saturation and contrast."""
    _, color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.8, 0, 255)
    return cv2.convertScaleAbs(
        cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), alpha=1.2, beta=10
    )


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
