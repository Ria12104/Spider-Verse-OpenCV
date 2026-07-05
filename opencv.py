"""
Finger-Gesture Filter Window
-----------------------------
Gesture -> Filter:
  1 finger  -> Oil Painting   |  2 fingers -> Thermal
  3 fingers -> Sketch         |  4 fingers -> Negative
  Fist      -> No filter      |  2 hands   -> Quad window
  Pinch & release             -> Sparkle burst
  'q' to quit

First run downloads hand_landmarker.task (~8 MB) automatically.
"""

import os
import time
import random
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# --- Model setup ---

MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_model_downloaded():
    if not os.path.exists(MODEL_FILENAME):
        print("Downloading hand-tracking model (first run only)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILENAME)
        print("Done.")


ensure_model_downloaded()

hand_landmarker = mp_vision.HandLandmarker.create_from_options(
    mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_FILENAME),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
    )
)

# --- Constants ---

MIN_RADIUS       = 40
MAX_RADIUS       = 300
WINDOW_PADDING   = 20
PINCH_THRESHOLD  = 0.55   # thumb-to-index / hand_size ratio

SPARKLE_LIFETIME      = 18   # shorter life = fewer active particles
SPARKLE_COUNT_PER_TAP = 10   # fewer particles per tap
SPARKLE_MAX_RADIUS    = 8

# --- Landmark indices (MediaPipe hand model) ---
# 0=wrist  4=thumb tip  5/8=index MCP/tip  9/12=middle  13/16=ring  17/20=pinky

def _dist(a, b):
    return np.hypot(a.x - b.x, a.y - b.y)

def _finger_up(lm, tip, knuckle):
    """Tip farther from wrist than knuckle = finger is extended."""
    return _dist(lm[tip], lm[0]) > _dist(lm[knuckle], lm[0])

def count_fingers_up(lm):
    """Return number of extended fingers (index–pinky), 0–4."""
    return sum(_finger_up(lm, t, k) for t, k in [(8,5),(12,9),(16,13),(20,17)])

def is_pinching(lm):
    """True when thumb and index tips are close relative to hand size."""
    size = _dist(lm[0], lm[9])
    return size > 0 and _dist(lm[4], lm[8]) / size < PINCH_THRESHOLD

# --- Filters ---

def oil_painting_filter(frame):
    """
    Bright oil-paint look:
      1. bilateralFilter ×7 — smooths texture, keeps hard edges (paint strokes)
      2. adaptiveThreshold  — burns in dark outlines at colour boundaries
      3. HSV boost          — cranks saturation and brightness for vivid pigment
    """
    smooth = frame
    for _ in range(4):   # 4 passes is the sweet spot: fast + still paint-like
        smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=150, sigmaSpace=150)

    gray  = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, blockSize=7, C=3)
    outlined = cv2.bitwise_and(smooth, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

    hsv = cv2.cvtColor(outlined, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.5, 0, 255)  # saturation
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.4, 0, 255)  # brightness
    return cv2.convertScaleAbs(
        cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), alpha=1.3, beta=15
    )

def thermal_filter(frame):
    """Luminance mapped to JET colormap — thermal camera look."""
    return cv2.applyColorMap(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                             cv2.COLORMAP_JET)

def sketch_filter(frame):
    """
    Vibrant colored pencil sketch:
      pencilSketch base -> HSV saturation ×1.8 -> contrast/brightness lift
    """
    _, color = cv2.pencilSketch(frame, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.8, 0, 255)
    return cv2.convertScaleAbs(
        cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), alpha=1.2, beta=10
    )

def negative_filter(frame):
    """Invert every pixel — classic photographic negative."""
    return 255 - frame

# finger count -> (function, label)
FILTER_MAP = {
    1: (oil_painting_filter, "Oil Painting"),
    2: (thermal_filter,      "Thermal"),
    3: (sketch_filter,       "Sketch"),
    4: (negative_filter,     "Negative"),
}

# --- Mask helpers ---

def make_circle_mask(shape, pt_a, pt_b):
    """Filled circle centred between pt_a and pt_b (single-hand mode)."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx = int(np.clip((pt_a[0] + pt_b[0]) / 2, 0, w))
    cy = int(np.clip((pt_a[1] + pt_b[1]) / 2, 0, h))
    r  = int(np.clip(int(np.hypot(*np.subtract(pt_b, pt_a))) + WINDOW_PADDING,
                     MIN_RADIUS, MAX_RADIUS))
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask

def make_polygon_mask(shape, points):
    """Filled quadrilateral from 4 points, sorted to avoid a bowtie (two-hand mode)."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(points, dtype=np.float32)
    c   = pts.mean(axis=0)
    ordered = pts[np.argsort(np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0]))].astype(np.int32)
    cv2.fillPoly(mask, [ordered], 255)
    return mask

def apply_masked_filter(frame, filtered, mask):
    """Show filtered inside the mask, original frame outside."""
    return cv2.add(
        cv2.bitwise_and(filtered, filtered, mask=mask),
        cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask)),
    )

# --- Sparkle particles ---

class Sparkle:
    _COLORS = [(0, 215, 255), (0, 255, 255), (0, 180, 255)]  # yellow / amber (BGR)

    def __init__(self, x, y):
        self.x, self.y  = float(x), float(y)
        self.vx         = random.uniform(-3, 3)
        self.vy         = random.uniform(-4, -1)   # drift upward
        self.life = self.max_life = SPARKLE_LIFETIME
        self.color      = random.choice(self._COLORS)

    def update(self):
        self.x += self.vx; self.y += self.vy; self.life -= 1

    def is_alive(self):
        return self.life > 0

    def draw(self, frame):
        fade  = self.life / self.max_life
        color = tuple(int(c * fade) for c in self.color)
        cv2.circle(frame, (int(self.x), int(self.y)),
                   max(2, int(SPARKLE_MAX_RADIUS * fade)), color, -1)

def spawn_sparkle_burst(particles, x, y):
    particles.extend(Sparkle(x, y) for _ in range(SPARKLE_COUNT_PER_TAP))

# --- Main loop ---

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    t0       = time.time()
    sparkles = []
    was_pinching = {}

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame  = cv2.flip(frame, 1)
            h, w   = frame.shape[:2]
            output = frame.copy()
            label  = "No hand detected"

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts_ms  = int((time.time() - t0) * 1000)
            hands  = hand_landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms
            ).hand_landmarks

            if hands:
                fingers = max(count_fingers_up(lm) for lm in hands)
                choice  = FILTER_MAP.get(fingers)

                # Build window mask: quad (2 hands) or circle (1 hand)
                if len(hands) >= 2:
                    pts  = [(int(lm[t].x*w), int(lm[t].y*h))
                            for lm in hands[:2] for t in (4, 8)]
                    mask = make_polygon_mask(frame.shape, pts)
                    label = "Two-hand quad window"
                else:
                    lm   = hands[0]
                    pa   = (int(lm[4].x*w), int(lm[4].y*h))
                    pb   = (int(lm[8].x*w), int(lm[8].y*h))
                    mask = make_circle_mask(frame.shape, pa, pb)

                if choice:
                    fn, name = choice
                    label  = f"Filter: {name} ({fingers} finger(s)) | {label}"
                    output = apply_masked_filter(frame, fn(frame), mask)
                else:
                    label = f"Fist / no filter | {label}"

                # Sparkle on pinch rising edge
                for i, lm in enumerate(hands):
                    now = is_pinching(lm)
                    if now and not was_pinching.get(i, False):
                        tx = int((lm[4].x + lm[8].x) / 2 * w)
                        ty = int((lm[4].y + lm[8].y) / 2 * h)
                        spawn_sparkle_burst(sparkles, tx, ty)
                    was_pinching[i] = now
            else:
                was_pinching.clear()

            for s in sparkles:
                s.update(); s.draw(output)
            sparkles = [s for s in sparkles if s.is_alive()]

            cv2.putText(output, label, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Gesture Filters", output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_landmarker.close()


if __name__ == "__main__":
    main()
