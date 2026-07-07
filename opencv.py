"""
Spider-Man Mask + Gesture Filter Window
========================================
By default: plain camera feed, no overlay.

Show a hand to open a filter window:
  • 1 hand   → circular window (between thumb & index tips)
  • 2 hands  → quad window (across all four fingertips)

Inside the window:
  Finger count selects filter (applied to camera + mask):
    1 finger  → Oil Painting
    2 fingers → Thermal
    3 fingers → Sketch
    4 fingers → Negative
    Fist      → No filter (Spider-Man mask still shows)

  Pinch (thumb + index close) → Sparkle burst

Press 'q' to quit.

Performance notes:
  - Face detection only runs when a hand is visible (skips ML inference otherwise)
  - Mask filter is applied to the bounding-box crop only (not the full frame)
  - Oil painting uses 3 bilateral passes (down from 4) — still looks good, faster
"""

import os, time, random, urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit these to tune the experience
# ─────────────────────────────────────────────────────────────────────────────

MASK_PATH  = "assets/spider-man-mask.png"
FACE_MODEL = "face_landmarker.task"
HAND_MODEL = "hand_landmarker.task"
FACE_URL   = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")
HAND_URL   = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/1/hand_landmarker.task")

# Mask appearance
SMOOTH       = 0.35   # EMA factor for transform smoothing (lower = smoother but laggier)
SCALE_FACTOR = 2.1    # Compensates for Spider-Man's wide eye lenses vs a real face
HAIR_SHIFT   = 0.20   # Fraction of face height to shift mask upward (covers hair)
FEATHER      = 31     # Alpha-blur width in px — higher = softer skin-blend edge

# Hand window geometry
MIN_RADIUS  = 40
MAX_RADIUS  = 300
WIN_PADDING = 20
PINCH_THR   = 0.55    # thumb-to-index / hand_size threshold for pinch detection

# Sparkle particles
SPARKLE_LIFE   = 18
SPARKLE_N      = 10
SPARKLE_RADIUS = 8

# Eye-hole centers measured in the 640×671 mask PNG (viewer's left / right)
MASK_EYE_L = np.float32([185, 310])
MASK_EYE_R = np.float32([455, 310])

# MediaPipe face landmark indices (478-point model)
EYE_L        = (33,  133)   # left eye inner / outer corners
EYE_R        = (362, 263)   # right eye inner / outer corners
IDX_FOREHEAD = 10            # top-of-scalp anchor
IDX_CHIN     = 152           # chin anchor

# Pre-allocate morphology kernel so it isn't rebuilt every frame
_ERODE_K = np.ones((5, 5), np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
# Model & asset loading
# ─────────────────────────────────────────────────────────────────────────────

for path, url in [(FACE_MODEL, FACE_URL), (HAND_MODEL, HAND_URL)]:
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)} (first run only)…")
        urllib.request.urlretrieve(url, path)
        print("Done.")

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

mask_img = cv2.imread(MASK_PATH, cv2.IMREAD_UNCHANGED)
if mask_img is None:
    raise FileNotFoundError(f"Mask not found: {MASK_PATH}")
if mask_img.shape[2] == 3:              # add alpha channel if missing
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2BGRA)

# ─────────────────────────────────────────────────────────────────────────────
# Image filters
# ─────────────────────────────────────────────────────────────────────────────

def oil_painting(img):
    """Bilateral-smoothed frame with edge outlines and boosted saturation/brightness."""
    s = img
    for _ in range(3):      # 3 passes: good quality without killing frame rate
        s = cv2.bilateralFilter(s, d=9, sigmaColor=150, sigmaSpace=150)
    gray  = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, blockSize=7, C=3)
    s = cv2.bitwise_and(s, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.5, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.4, 0, 255)
    return cv2.convertScaleAbs(
        cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), alpha=1.3, beta=15)

def thermal(img):
    """JET colormap on grayscale luminance — thermal-camera look."""
    return cv2.applyColorMap(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)

def sketch(img):
    """Colored pencil sketch with boosted saturation and contrast."""
    _, color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.8, 0, 255)
    return cv2.convertScaleAbs(
        cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), alpha=1.2, beta=10)

def negative(img):
    """Invert every channel — classic photographic negative."""
    return 255 - img

# Maps finger count → (filter function, display label)
FILTERS = {
    1: (oil_painting, "Oil Painting"),
    2: (thermal,      "Thermal"),
    3: (sketch,       "Sketch"),
    4: (negative,     "Negative"),
}

# ─────────────────────────────────────────────────────────────────────────────
# Hand gesture helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dist(a, b):
    return np.hypot(a.x - b.x, a.y - b.y)

def count_fingers(lm):
    """Count extended fingers (index–pinky) by comparing tip-to-wrist vs knuckle-to-wrist."""
    return sum(_dist(lm[tip], lm[0]) > _dist(lm[knuckle], lm[0])
               for tip, knuckle in [(8, 5), (12, 9), (16, 13), (20, 17)])

def is_pinching(lm):
    """True when thumb and index tips are closer than PINCH_THR × hand size."""
    size = _dist(lm[0], lm[9])
    return size > 0 and _dist(lm[4], lm[8]) / size < PINCH_THR

# ─────────────────────────────────────────────────────────────────────────────
# Hand window mask builders
# ─────────────────────────────────────────────────────────────────────────────

def circle_mask(shape, pa, pb):
    """Filled circle centred between thumb and index tips (single-hand window)."""
    h, w = shape[:2]
    m    = np.zeros((h, w), np.uint8)
    cx   = int(np.clip((pa[0] + pb[0]) / 2, 0, w))
    cy   = int(np.clip((pa[1] + pb[1]) / 2, 0, h))
    r    = int(np.clip(np.hypot(*np.subtract(pb, pa)) + WIN_PADDING, MIN_RADIUS, MAX_RADIUS))
    cv2.circle(m, (cx, cy), r, 255, -1)
    return m

def quad_mask(shape, points):
    """Filled convex quad from 4 fingertip points, sorted by angle to avoid bowtie."""
    h, w = shape[:2]
    m    = np.zeros((h, w), np.uint8)
    pts  = np.array(points, np.float32)
    c    = pts.mean(axis=0)
    ordered = pts[np.argsort(np.arctan2(pts[:, 1] - c[1],
                                        pts[:, 0] - c[0]))].astype(np.int32)
    cv2.fillPoly(m, [ordered], 255)
    return m

# ─────────────────────────────────────────────────────────────────────────────
# Sparkle particles
# ─────────────────────────────────────────────────────────────────────────────

class Sparkle:
    _COLORS = [(0, 215, 255), (0, 255, 255), (0, 180, 255)]  # amber / gold tones

    def __init__(self, x, y):
        self.x, self.y = float(x), float(y)
        self.vx        = random.uniform(-3, 3)
        self.vy        = random.uniform(-4, -1)   # drift upward
        self.life      = self.max_life = SPARKLE_LIFE
        self.color     = random.choice(self._COLORS)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1

    def is_alive(self):
        return self.life > 0

    def draw(self, frame):
        fade = self.life / self.max_life
        cv2.circle(frame, (int(self.x), int(self.y)),
                   max(2, int(SPARKLE_RADIUS * fade)),
                   tuple(int(c * fade) for c in self.color), -1)

# ─────────────────────────────────────────────────────────────────────────────
# Spider-Man mask renderer
# ─────────────────────────────────────────────────────────────────────────────

smoothed_M = None   # EMA-smoothed affine transform (persists across frames)

def render_mask(base, lm, filter_fn=None):
    """
    Warp the Spider-Man mask onto `base` aligned to the detected face.

    Steps:
      1. Compute similarity transform from mask eye positions → real eye positions
      2. Scale up to compensate for the mask's wide eye-lens proportions
      3. Shift upward to cover hair
      4. EMA-smooth the transform to damp per-frame landmark jitter
      5. Warp mask, optionally apply filter (to bounding-box crop only for speed)
      6. Feather alpha edge (erode → blur) for a natural skin blend

    `filter_fn` is applied to only the non-transparent bounding box of the
    warped mask — not the full frame — to avoid running an expensive filter
    on mostly-empty pixels.
    """
    global smoothed_M
    h, w = base.shape[:2]
    pts  = np.array([[p.x * w, p.y * h] for p in lm], np.float32)

    # Average inner/outer corner landmarks to get a stable eye center
    eye_l = (pts[EYE_L[0]] + pts[EYE_L[1]]) / 2
    eye_r = (pts[EYE_R[0]] + pts[EYE_R[1]]) / 2

    # Similarity transform: align mask eye-holes to real eye positions
    # This gives the correct rotation and base scale in one step.
    M, _ = cv2.estimateAffinePartial2D(
        np.float32([MASK_EYE_L, MASK_EYE_R]),
        np.float32([eye_l, eye_r]),
    )
    if M is None:
        return base

    # Scale up around the eye midpoint so the mask fills the full face.
    # Spider-Man's eye lenses are proportionally wider than real eyes, so the
    # naive eye-to-eye scale makes the mask appear about half the correct size.
    cx, cy = (eye_l + eye_r) / 2
    M[0, 2] += (1 - SCALE_FACTOR) * (M[0, 0] * cx + M[0, 1] * cy)
    M[1, 2] += (1 - SCALE_FACTOR) * (M[1, 0] * cx + M[1, 1] * cy)
    M[:, :2] *= SCALE_FACTOR

    # Shift the whole mask upward along the face axis to cover the hairline
    up     = pts[IDX_FOREHEAD] - pts[IDX_CHIN]
    face_h = np.linalg.norm(up)
    up    /= face_h + 1e-6
    M[:, 2] += up * face_h * HAIR_SHIFT

    # EMA-smooth the transform matrix to reduce jitter from noisy landmarks
    if smoothed_M is None:
        smoothed_M = M.copy()
    else:
        smoothed_M[:] = SMOOTH * M + (1 - SMOOTH) * smoothed_M

    # Warp the mask image to the current frame dimensions
    warped = cv2.warpAffine(mask_img, smoothed_M, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    mask_bgr = warped[:, :, :3].copy()

    if filter_fn is not None:
        # Find the tight bounding box of non-transparent pixels and filter only that.
        # This avoids running the expensive filter over the ~80% of pixels that are black.
        nz = cv2.findNonZero(warped[:, :, 3])
        if nz is not None:
            x, y, bw, bh = cv2.boundingRect(nz)
            # Small padding so filter edge effects don't clip the mask boundary
            x1, y1 = max(0, x - 4),      max(0, y - 4)
            x2, y2 = min(w, x + bw + 4), min(h, y + bh + 4)
            mask_bgr[y1:y2, x1:x2] = filter_fn(mask_bgr[y1:y2, x1:x2])

    # Feather the alpha channel so the mask blends into skin rather than cutting off hard.
    # Erode first (pulls the visible edge inward to hide the hard boundary),
    # then Gaussian-blur (creates a smooth gradient falloff at the perimeter).
    soft_a = cv2.erode(warped[:, :, 3], _ERODE_K, iterations=2)
    soft_a = cv2.GaussianBlur(soft_a, (FEATHER | 1, FEATHER | 1), 0)
    alpha  = soft_a.astype(np.float32)[:, :, None] / 255.0

    return (mask_bgr * alpha + base * (1 - alpha)).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global smoothed_M
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    t0           = time.time()
    sparkles     = []        # active Sparkle instances
    was_pinching = {}        # per-hand pinch state for rising-edge detection

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame  = cv2.flip(frame, 1)            # mirror so it feels like a selfie
            h, w   = frame.shape[:2]
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts     = int((time.time() - t0) * 1000)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Hand detection drives the whole interaction; always runs
            hands  = hand_lmk.detect_for_video(mp_img, ts).hand_landmarks

            output = frame.copy()   # default output: plain camera
            label  = ""

            if hands:
                # ── 1. Determine filter from finger count ─────────────────
                fingers   = max(count_fingers(lm) for lm in hands)
                choice    = FILTERS.get(fingers)
                filter_fn = choice[0] if choice else None
                label     = choice[1] if choice else "No filter (fist)"

                # ── 2. Build hand window (circle for 1 hand, quad for 2) ──
                if len(hands) >= 2:
                    tips     = [(int(lm[t].x*w), int(lm[t].y*h))
                                for lm in hands[:2] for t in (4, 8)]
                    win_mask = quad_mask(frame.shape, tips)
                    label    = f"Quad | {label}"
                else:
                    lm0      = hands[0]
                    pa       = (int(lm0[4].x*w), int(lm0[4].y*h))   # thumb tip
                    pb       = (int(lm0[8].x*w), int(lm0[8].y*h))   # index tip
                    win_mask = circle_mask(frame.shape, pa, pb)

                # ── 3. Compose inner content: filter + mask ───────────────
                # Apply filter to the full camera frame (background inside window)
                inner = filter_fn(frame) if filter_fn else frame.copy()

                # Face detection only runs when a hand is visible.
                # Skipping it when no hand is up saves one full ML inference per frame.
                face_res = face_lmk.detect_for_video(mp_img, ts)
                if face_res.face_landmarks:
                    # Overlay Spider-Man mask; same filter applied to mask colors too
                    inner = render_mask(inner, face_res.face_landmarks[0], filter_fn)
                else:
                    smoothed_M = None   # reset EMA when face is lost

                # ── 4. Clip: inner inside window, plain camera outside ─────
                win3   = win_mask[:, :, None].astype(np.float32) / 255.0
                output = (inner * win3 + frame * (1 - win3)).astype(np.uint8)

                # ── 5. Sparkle burst on pinch rising edge ─────────────────
                for i, lm in enumerate(hands):
                    pinching = is_pinching(lm)
                    if pinching and not was_pinching.get(i, False):
                        tx = int((lm[4].x + lm[8].x) / 2 * w)
                        ty = int((lm[4].y + lm[8].y) / 2 * h)
                        sparkles.extend(Sparkle(tx, ty) for _ in range(SPARKLE_N))
                    was_pinching[i] = pinching

            else:
                # No hand: reset all per-hand state
                was_pinching.clear()
                smoothed_M = None

            # ── Advance and draw sparkles over the final output ───────────
            for s in sparkles:
                s.update()
                s.draw(output)
            sparkles = [s for s in sparkles if s.is_alive()]

            # ── HUD: show active filter name in top-left ──────────────────
            if label:
                cv2.putText(output, label, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Spider-Man", output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_lmk.close()
        hand_lmk.close()


if __name__ == "__main__":
    main()
