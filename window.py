"""
window.py — Hand-window mask builders.

These functions produce a single-channel (grayscale) mask that defines the
region of the frame where the active filter is revealed.

  circle_mask  — one hand  → circular window centred between thumb & index tip
  quad_mask    — two hands → convex quad spanning all four fingertips
"""

import cv2
import numpy as np

from config import MIN_RADIUS, MAX_RADIUS, WIN_PADDING


def circle_mask(shape: tuple, pa: tuple, pb: tuple) -> np.ndarray:
    """
    Return a filled-circle mask centred between *pa* and *pb*.

    Parameters
    ----------
    shape : (height, width[, channels])
        Shape of the target frame — only height and width are used.
    pa, pb : (int, int)
        Pixel coordinates of the two reference points (e.g. thumb tip /
        index tip).  The circle centre is their midpoint; the radius is
        their distance plus WIN_PADDING, clamped to [MIN_RADIUS, MAX_RADIUS].

    Returns
    -------
    np.ndarray  uint8 mask — 255 inside the circle, 0 outside.
    """
    h, w = shape[:2]
    m    = np.zeros((h, w), np.uint8)
    cx   = int(np.clip((pa[0] + pb[0]) / 2, 0, w))
    cy   = int(np.clip((pa[1] + pb[1]) / 2, 0, h))
    r    = int(np.clip(np.hypot(*np.subtract(pb, pa)) + WIN_PADDING, MIN_RADIUS, MAX_RADIUS))
    cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def quad_mask(shape: tuple, points: list) -> np.ndarray:
    """
    Return a filled convex-quad mask from four *points*.

    Points are sorted by polar angle around their centroid before drawing so
    the polygon always winds correctly (no bowtie artefacts regardless of
    input order).

    Parameters
    ----------
    shape : (height, width[, channels])
    points : list of (int, int)
        Four pixel-coordinate pairs (typically two thumb tips + two index tips
        from the two visible hands).

    Returns
    -------
    np.ndarray  uint8 mask — 255 inside the quad, 0 outside.
    """
    h, w = shape[:2]
    m    = np.zeros((h, w), np.uint8)
    pts  = np.array(points, np.float32)
    c    = pts.mean(axis=0)
    ordered = pts[
        np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    ].astype(np.int32)
    cv2.fillPoly(m, [ordered], 255)
    return m
