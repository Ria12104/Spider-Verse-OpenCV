# 🕷️ Gesture Filter Window — Spider-verse Edition

A real-time webcam demo where **hand gestures trigger live visual filters**, built for a workshop on computer vision with OpenCV and MediaPipe.

Hold up fingers in front of your webcam and watch your video feed transform.

---

## ✋ Gesture → Filter Map

| Gesture | Filter |
|---|---|
| ☝️ 1 finger (index) | 🎨 **Oil Painting** — bilateral smoothing + vivid colour boost |
| ✌️ 2 fingers (index + middle) | 🌡️ **Thermal** — JET colormap heat-camera look |
| 🤟 3 fingers (index + middle + ring) | ✏️ **Sketch** — vibrant colored pencil drawing |
| 🖐️ 4 fingers (index–pinky) | 🔄 **Negative** — inverted colors |
| ✊ Fist / no gesture | No filter — plain camera |
| 🤲 Two hands forming a shape | Filter appears **inside the quadrilateral** your hands define |
| 🤌 Pinch then release | ✨ **Sparkle burst** at pinch point |
| `q` key | Quit |

---

## 🔧 How the filters work

### 🎨 Oil Painting
Uses `cv2.bilateralFilter` (×4 passes) to merge colors into broad brush strokes while preserving hard edges, then `cv2.adaptiveThreshold` to add inked outlines. Saturation and brightness are cranked up in HSV space for that thick pigment look.

### 🌡️ Thermal
Converts to grayscale and applies OpenCV's built-in `COLORMAP_JET` — warm colors for bright areas, cool for dark, just like a real thermal camera.

### ✏️ Sketch
`cv2.pencilSketch()` generates a colored pencil drawing. We then boost saturation ×1.8 in HSV space and lift contrast slightly so the colors pop.

### 🔄 Negative
One line: `255 - frame`. Classic photographic negative.

---

## 📦 Requirements

```
python >= 3.9
opencv-contrib-python
mediapipe
numpy
```

Install everything:
```bash
pip install opencv-contrib-python mediapipe numpy
```

> **Note:** `opencv-contrib-python` is required (not plain `opencv-python`) for `cv2.pencilSketch`.

---

## 🚀 Run

```bash
python opencv.py
```

On **first run** the MediaPipe hand-tracking model (`hand_landmarker.task`, ~8 MB) is downloaded automatically. After that it runs offline.

---

## 🗂️ Project structure

```
opencv.py              # main script — all filters + gesture logic in one file
hand_landmarker.task   # auto-downloaded on first run, not committed to git
```

---

## 🧠 Workshop concepts covered

| Concept | Where it appears |
|---|---|
| Reading & flipping webcam frames | `cv2.VideoCapture`, `cv2.flip` |
| Color space conversions | `COLOR_BGR2GRAY`, `COLOR_BGR2HSV`, `COLOR_BGR2RGB` |
| Edge-preserving blur | `cv2.bilateralFilter` |
| Thresholding | `cv2.adaptiveThreshold` |
| Colormaps | `cv2.applyColorMap` |
| Alpha blending / masking | `cv2.bitwise_and`, `cv2.add`, HSV channel math |
| Drawing primitives | `cv2.circle`, `cv2.fillPoly`, `cv2.putText` |
| Hand landmark tracking | MediaPipe `HandLandmarker` (Tasks API) |
| Gesture recognition | Distance-based finger-up detection |
| Particle systems | `Sparkle` class — position, velocity, fade, lifetime |

---

## ⚡ Performance tips

If the oil painting filter feels slow, reduce the bilateral pass count:

```python
for _ in range(4):   # try 2 or 3 if needed
    smooth = cv2.bilateralFilter(...)
```

Sparkle density is tunable at the top of the file:
```python
SPARKLE_COUNT_PER_TAP = 10   # lower = fewer particles
SPARKLE_LIFETIME      = 18   # lower = particles die sooner
```

---

## 📄 License

MIT — free to use, modify, and share.
