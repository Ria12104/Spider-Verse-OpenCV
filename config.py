"""
config.py — All tunable constants for the Spider-verse filter app.
Edit values here to tune the experience without touching logic files.
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Asset paths & model URLs
# ─────────────────────────────────────────────────────────────────────────────

MASK_PATH  = "assets/spider-man-mask.png"
FACE_MODEL = "face_landmarker.task"
HAND_MODEL = "hand_landmarker.task"
FACE_URL   = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
HAND_URL   = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# ─────────────────────────────────────────────────────────────────────────────
# Mask appearance
# ─────────────────────────────────────────────────────────────────────────────

SMOOTH       = 0.35   # EMA factor for transform smoothing (lower = smoother but laggier)
SCALE_FACTOR = 2.1    # Compensates for Spider-Man's wide eye lenses vs a real face
HAIR_SHIFT   = 0.20   # Fraction of face height to shift mask upward (covers hair)
FEATHER      = 31     # Alpha-blur width in px — higher = softer skin-blend edge

# ─────────────────────────────────────────────────────────────────────────────
# Hand window geometry
# ─────────────────────────────────────────────────────────────────────────────

MIN_RADIUS  = 40
MAX_RADIUS  = 300
WIN_PADDING = 20
PINCH_THR   = 0.55    # thumb-to-index / hand_size threshold for pinch detection

# ─────────────────────────────────────────────────────────────────────────────
# Sparkle particles
# ─────────────────────────────────────────────────────────────────────────────

SPARKLE_LIFE   = 18
SPARKLE_N      = 10
SPARKLE_RADIUS = 8

# ─────────────────────────────────────────────────────────────────────────────
# Mask eye-hole centers (measured in the 640×671 mask PNG)
# ─────────────────────────────────────────────────────────────────────────────

MASK_EYE_L = np.float32([185, 310])
MASK_EYE_R = np.float32([455, 310])

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe face landmark indices (478-point model)
# ─────────────────────────────────────────────────────────────────────────────

EYE_L        = (33,  133)   # left eye inner / outer corners
EYE_R        = (362, 263)   # right eye inner / outer corners
IDX_FOREHEAD = 10            # top-of-scalp anchor
IDX_CHIN     = 152           # chin anchor

# ─────────────────────────────────────────────────────────────────────────────
# Pre-allocated morphology kernel (avoids rebuilding every frame)
# ─────────────────────────────────────────────────────────────────────────────

ERODE_KERNEL = np.ones((5, 5), np.uint8)
