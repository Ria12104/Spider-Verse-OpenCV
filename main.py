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

import time

import cv2
import mediapipe as mp
import numpy as np

from filters import FILTERS
from gestures import count_fingers, is_pinching
from mask_renderer import render_mask, reset_transform
from models import load_mask, load_models
from sparkle import Sparkle
from config import SPARKLE_N
from window import circle_mask, quad_mask


def main() -> None:
    # ── Initialise models and assets ──────────────────────────────────────────
    face_lmk, hand_lmk = load_models()
    mask_img            = load_mask()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    t0           = time.time()
    sparkles: list[Sparkle] = []
    was_pinching: dict[int, bool] = {}   # per-hand pinch state for rising-edge detection
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame  = cv2.flip(frame, 1)                    # mirror → selfie feel
            h, w   = frame.shape[:2]
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts     = int((time.time() - t0) * 1000)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Hand detection drives the whole interaction; always runs
            hands  = hand_lmk.detect_for_video(mp_img, ts).hand_landmarks

            output = frame.copy()   # default: plain camera
            label  = ""

            if hands:
                # ── 1. Determine filter from finger count ─────────────────
                fingers   = max(count_fingers(lm) for lm in hands)
                choice    = FILTERS.get(fingers)
                filter_fn = choice[0] if choice else None
                label     = choice[1] if choice else "No filter (fist)"

                # ── 2. Build hand window (circle for 1 hand, quad for 2) ──
                if len(hands) >= 2:
                    tips = [
                        (int(lm[t].x * w), int(lm[t].y * h))
                        for lm in hands[:2]
                        for t in (4, 8)
                    ]
                    win_mask = quad_mask(frame.shape, tips)
                    label    = f"Quad | {label}"
                else:
                    lm0      = hands[0]
                    pa       = (int(lm0[4].x * w), int(lm0[4].y * h))   # thumb tip
                    pb       = (int(lm0[8].x * w), int(lm0[8].y * h))   # index tip
                    win_mask = circle_mask(frame.shape, pa, pb)

                # ── 3. Compose inner content: filter + mask ───────────────
                # Apply filter to the full camera frame (background inside window)
                inner = filter_fn(frame) if filter_fn else frame.copy()

                # Face detection only runs when a hand is visible.
                # Skipping it when no hand is up saves one full ML inference per frame.
                face_res = face_lmk.detect_for_video(mp_img, ts)
                if face_res.face_landmarks:
                    inner = render_mask(inner, face_res.face_landmarks[0], mask_img, filter_fn)
                else:
                    reset_transform()   # reset EMA when face is lost

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
                # No hand visible: reset all per-hand state
                was_pinching.clear()
                reset_transform()

            # ── Advance and draw sparkles over the final output ───────────
            for s in sparkles:
                s.update()
                s.draw(output)
            sparkles = [s for s in sparkles if s.is_alive()]

            # ── HUD: active filter name in top-left ───────────────────────
            if label:
                cv2.putText(
                    output, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
                )

            cv2.imshow("Spider-Man", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_lmk.close()
        hand_lmk.close()


if __name__ == "__main__":
    main()
