"""
demo.py -- MediaPipe raw landmark visualiser
Shows how MediaPipe maps faces (478 pts) and hands (21 pts) in real time.
Uses the Tasks API (mediapipe >= 0.10) with the .task files already on disk.
Press q to quit.
"""

import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# --- MediaPipe 0.10+ (Tasks-only) stripped out mp.solutions entirely.      ---
# --- The hand skeleton topology is fixed at 21 landmarks — it's just       ---
# --- anatomy (4 bones per finger + palm edge).  MediaPipe documents these  ---
# --- at: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker ---
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm edge (wrist → pinky MCP)
]

# --- Load Hand Landmarker (Tasks API) ---
hand_lmk = mp_vision.HandLandmarker.create_from_options(
    mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
)

# --- Load Face Landmarker (Tasks API) ---
face_lmk = mp_vision.FaceLandmarker.create_from_options(
    mp_vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path="face_landmarker.task"),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)

cap = cv2.VideoCapture(0)
t0  = time.time()

FONT = cv2.FONT_HERSHEY_SIMPLEX

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frame   = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ts      = int((time.time() - t0) * 1000)
    mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    hand_result = hand_lmk.detect_for_video(mp_img, ts)
    face_result = face_lmk.detect_for_video(mp_img, ts)

    num_hands  = len(hand_result.hand_landmarks)
    face_found = len(face_result.face_landmarks) > 0

    # --- Hand landmarks + skeleton (first hand only) ---
    for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        # Skeleton connections
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, pts[start], pts[end], (0, 200, 0), 2)

        # Dots + index labels
        for idx, (cx, cy) in enumerate(pts):
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame, str(idx), (cx + 5, cy - 4), FONT, 0.3, (0, 220, 0), 1)

        # Left / Right label from MediaPipe's handedness output
        if hand_result.handedness and i < len(hand_result.handedness):
            side = hand_result.handedness[i][0].display_name
            wx, wy = pts[0]
            cv2.putText(frame, side, (wx - 10, wy + 20), FONT, 0.6, (0, 255, 255), 2)

    # --- 478 face landmarks as dots ---
    # (FACEMESH_TESSELATION was also in the removed mp.solutions; the tessellation
    #  has 900+ edges which would need the same kind of table, so dots are cleaner)
    for face_landmarks in face_result.face_landmarks:
        for lm in face_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)


    # --- Status HUD (top-left) ---
    hud = [
        ("Face",  face_found,  f"Face: {'Detected (' + str(len(face_result.face_landmarks)) + ')' if face_found else 'Not detected'}"),
        ("Hands", num_hands > 0, f"Hands: {num_hands} detected" if num_hands else "Hands: Not detected"),
    ]
    for row, (_, active, text) in enumerate(hud):
        color = (0, 255, 0) if active else (0, 0, 255)
        cv2.putText(frame, text, (10, 30 + row * 28), FONT, 0.65, color, 2, cv2.LINE_AA)

    cv2.imshow("MediaPipe: raw landmark map  (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hand_lmk.close()
face_lmk.close()

