"""
models.py — Loads MediaPipe models and the Spider-Man mask image.
Downloads model files on first run if they are not already present.
"""

import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from config import MASK_PATH, FACE_MODEL, HAND_MODEL, FACE_URL, HAND_URL


def _ensure_model(path: str, url: str) -> None:
    """Download *path* from *url* if the file does not already exist."""
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)} (first run only)…")
        urllib.request.urlretrieve(url, path)
        print("Done.")


def load_models():
    """
    Download (if needed) and initialise the MediaPipe face and hand landmarkers.

    Returns
    -------
    face_lmk : FaceLandmarker  — VIDEO-mode face landmarker
    hand_lmk : HandLandmarker  — VIDEO-mode hand landmarker (up to 2 hands)
    """
    _ensure_model(FACE_MODEL, FACE_URL)
    _ensure_model(HAND_MODEL, HAND_URL)

    face_lmk = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=FACE_MODEL),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
        )
    )
    hand_lmk = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
        )
    )
    return face_lmk, hand_lmk


def load_mask() -> "np.ndarray":
    """
    Load the Spider-Man mask PNG with an alpha channel.
    Raises FileNotFoundError if the asset is missing.
    """
    import numpy as np  # local import keeps module-level imports minimal

    img = cv2.imread(MASK_PATH, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Mask not found: {MASK_PATH}")
    if img.shape[2] == 3:                        # add alpha channel if missing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img
