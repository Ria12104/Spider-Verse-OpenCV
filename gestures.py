"""
gestures.py — Hand-gesture helper functions.

Works with a single hand's landmark list as returned by MediaPipe's
HandLandmarker (a list of NormalizedLandmark objects with .x / .y fields).
"""

import numpy as np

from config import PINCH_THR


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dist(a, b) -> float:
    """Euclidean distance between two NormalizedLandmark points (x, y only)."""
    return np.hypot(a.x - b.x, a.y - b.y)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def count_fingers(lm) -> int:
    """
    Count extended fingers (index through pinky) for one hand.

    A finger is considered extended when its tip is farther from the wrist
    (landmark 0) than the corresponding MCP knuckle — a simple but robust
    heuristic that works well for frontal hand poses.

    Parameters
    ----------
    lm : list[NormalizedLandmark]
        21-point hand landmark list from MediaPipe.

    Returns
    -------
    int  0–4
    """
    return sum(
        _dist(lm[tip], lm[0]) > _dist(lm[knuckle], lm[0])
        for tip, knuckle in [(8, 5), (12, 9), (16, 13), (20, 17)]
    )


def is_pinching(lm) -> bool:
    """
    Detect a pinch gesture (thumb tip close to index tip).

    The raw distance is normalised by the wrist-to-middle-MCP span so the
    threshold is scale-independent.

    Parameters
    ----------
    lm : list[NormalizedLandmark]

    Returns
    -------
    bool  True when a pinch is detected.
    """
    size = _dist(lm[0], lm[9])
    return size > 0 and _dist(lm[4], lm[8]) / size < PINCH_THR
