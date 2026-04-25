# Flipo Flip Rolling Tracker — v6.0 Engineering Workflow

> **Goal:** Simultaneously locate the centroid of a rolling Flipo Flip disc, determine its rotational angle (via ellipse fitting), and track the orientation of a hand-drawn white marker line — all in real time using OpenCV.

---

## Table of Contents

1. [Problem Statement & Constraints](#1-problem-statement--constraints)
2. [Environment & Dependencies](#2-environment--dependencies)
3. [Algorithm Overview](#3-algorithm-overview)
4. [Step-by-Step Pipeline](#4-step-by-step-pipeline)
   - [Step 1 — CLAHE Contrast Enhancement](#step-1--clahe-contrast-enhancement)
   - [Step 2 — HSV Blue Mask](#step-2--hsv-blue-mask)
   - [Step 3 — Morphological Solidification & Erosion](#step-3--morphological-solidification--erosion)
   - [Step 4 — Ground-Line Calibration & CC Rejection](#step-4--ground-line-calibration--cc-rejection)
   - [Step 5 — Column / Row Thickness Filter *(Key Fix)*](#step-5--column--row-thickness-filter-key-fix)
   - [Step 6 — Ellipse Fitting & Centroid](#step-6--ellipse-fitting--centroid)
   - [Step 7 — White Marker Line Detection](#step-7--white-marker-line-detection)
5. [Complete Source Code](#5-complete-source-code)
6. [Parameter Reference](#6-parameter-reference)
7. [Validation Methods](#7-validation-methods)
8. [Known Limitations & Future Work](#8-known-limitations--future-work)

---

## 1. Problem Statement & Constraints

### Target object
- **Flipo Flip**: a small blue elliptical disc, rolling on a flat surface.
- A short **white marker line** is drawn through the disc center with a marker pen; its angle encodes the disc's instantaneous rotational phase.

### Interference sources

| Source | Symptom | Severity |
|--------|---------|----------|
| Backlight / shadow on disc | Dark side drops in HSV saturation → partial blue mask | High |
| Ground surface similar in hue to disc | Ground shadow detected as Flipo | High |
| Ground shadow **fused** with disc blob | Single connected component spans disc + ground strip | High |
| Floor cracks / dark seams | Additional small spurious contours | Medium |
| Window frame / curtain reflection | Tall thin vertical blue strips (aspect ≈ 0.1) | Medium |

### Invariants we can rely on
- The **ground line is horizontal** (parallel to X axis) and fixed per recording session.
- Flipo Flip always has **meaningful thickness** in every direction (it is a disc, not a line).
- The white marker line **passes through the disc center**.

---

## 2. Environment & Dependencies

```bash
pip install opencv-python-headless numpy matplotlib
```

| Library | Version tested | Role |
|---------|---------------|------|
| `opencv-python-headless` | 4.x | All image processing |
| `numpy` | 1.x / 2.x | Array operations |
| `matplotlib` | 3.x | Debug visualization only |

---

## 3. Algorithm Overview

```
Raw frame (BGR)
      │
      ▼
[Step 1] CLAHE enhancement          ← fix backlight
      │
      ▼
[Step 2] HSV blue mask              ← isolate disc color
      │
      ▼
[Step 3] Morphology                 ← fill holes, cut thin connections
         close (7×7) → erode 3× (5×5) → dilate 3× (5×5)
      │
      ▼
[Step 4] Ground-line CC rejection   ← drop independent thin ground strips
         auto-detect ground_y via Canny edge density
         reject CC where: near_ground AND thin AND wide
      │
      ▼
[Step 5] Column / row thickness filter  ← ★ KEY FIX for fused blobs
         for each column: if fill < MIN_THICKNESS → zero it out
         for each row:    if fill < MIN_THICKNESS → zero it out
      │
      ▼
[Step 6] Ellipse fit on largest contour
         → centroid (cx, cy), ellipse angle
      │
      ▼
[Step 7] White line detection inside body ROI
         HSV white threshold → nearest CC to centroid → fitLine
         → line angle
      │
      ▼
Annotated output frame
```

---

## 4. Step-by-Step Pipeline

### Step 1 — CLAHE Contrast Enhancement

**Why:** Motion blur and backlighting cause the shadowed side of the disc to have low HSV saturation and value, making it nearly invisible to a plain color threshold. CLAHE (Contrast Limited Adaptive Histogram Equalization) applied to the **L channel in LAB space** boosts local contrast without over-brightening bright areas.

```python
lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l_enhanced = clahe.apply(l)

img_enhanced = cv2.cvtColor(cv2.merge([l_enhanced, a, b]),
                            cv2.COLOR_LAB2BGR)
```

**Key parameters:**

| Parameter | Value | Effect |
|-----------|-------|--------|
| `clipLimit` | 3.0 | Max amplification per tile — higher → more contrast, more noise |
| `tileGridSize` | (8, 8) | Tile count — finer grid adapts more locally |

---

### Step 2 — HSV Blue Mask

**Why:** HSV separates chromatic information (H) from lighting intensity (V), making color thresholding far more robust to illumination changes than RGB thresholding.

```python
hsv = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2HSV)

lower_blue = np.array([85,  40,  40])
upper_blue = np.array([140, 255, 255])

blue_raw = cv2.inRange(hsv, lower_blue, upper_blue)
```

**Threshold rationale:**

| Channel | Range | Rationale |
|---------|-------|-----------|
| H (Hue) | 85–140 | Covers blue spectrum; extended toward cyan (85) and indigo (140) to capture shadow-side color shift |
| S (Saturation) | 40–255 | Low floor (40) captures desaturated shadow side without pulling in grey background |
| V (Value) | 40–255 | Low floor captures dark underlit pixels |

> **Tuning tip:** If the background (e.g. a grey wall) bleeds into the mask, raise the S lower bound from 40 toward 60–80. If shadow side is missed, lower it toward 25.

---

### Step 3 — Morphological Solidification & Erosion

Two distinct purposes addressed by two separate operations:

#### 3a. Close (fill holes)
Motion blur creates a hollow center inside the disc mask. A large closing kernel fills these holes to produce a solid body.

```python
k7 = np.ones((7, 7), np.uint8)
solid = cv2.morphologyEx(blue_raw, cv2.MORPH_CLOSE, k7)
```

#### 3b. Erode → Dilate (sever thin connections)
Any region thinner than `kernel_size × iterations` pixels is physically destroyed by erosion and cannot be recovered by the subsequent dilation. This severs thin ground-contact strips **before** they fuse with the disc body.

```python
k5 = np.ones((5, 5), np.uint8)
eroded = cv2.erode(solid, k5, iterations=3)   # severs regions < ~15px thick
morph  = cv2.dilate(eroded, k5, iterations=3) # restores thicker regions
```

**Effective cut thickness:** `kernel_half × iterations = 2.5 × 3 ≈ 7–15 px` — anything thinner is permanently removed.

> **Tuning tip:** Increase `iterations` to cut thicker ground connections; decrease if the disc itself is being eroded.

---

### Step 4 — Ground-Line Calibration & CC Rejection

#### 4a. Auto-detect ground line

Scan the bottom 40% of the frame for the row with the highest Canny edge density — this is the ground surface boundary.

```python
def detect_ground_line(img_bgr):
    H, W = img_bgr.shape[:2]
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)

    search_top = int(H * 0.60)          # only look in bottom 40%
    row_sums   = np.sum(edges[search_top:, :], axis=1)
    return search_top + int(np.argmax(row_sums))
```

> **Alternative:** Set `ground_y` manually once per recording session for maximum stability.

#### 4b. Connected-component geometric filter

After morphology, label all connected components and apply three simultaneous conditions to identify and reject independent ground shadow strips:

```python
GROUND_TOL  = 15    # px — how close to ground_y counts as "touching"
THICK_LIMIT = 30    # px — maximum height of a shadow strip
ASPECT_MAX  = 6.0   # width/height ratio above which a blob is "strip-like"

n, labels, stats, _ = cv2.connectedComponentsWithStats(morph)

for i in range(1, n):
    x_s, y_s, w_s, h_s, area = stats[i, :5]
    if area < 100:
        continue

    bottom = y_s + h_s
    aspect = w_s / max(h_s, 1)

    near_ground = bottom >= ground_y - GROUND_TOL
    is_thin     = h_s < THICK_LIMIT
    is_wide     = aspect > ASPECT_MAX

    if near_ground and is_thin and is_wide:
        morph[labels == i] = 0   # reject
```

**Logic (the `___皿__` pattern):**

```
___皿__
```

- `皿` = Flipo disc: high bbox height, moderate aspect → **KEPT**
- `___` = ground shadow: bbox height < 30px, aspect > 6, bottom touches ground → **REJECTED**

**Why three conditions together?** Any single condition would cause false rejections. The disc itself may be near the ground or have large width; requiring all three simultaneously ensures only genuine thin horizontal strips are removed.

---

### Step 5 — Column / Row Thickness Filter *(Key Fix)*

**Root cause of the remaining problem:** After Step 4, there are cases where the ground shadow strip is *fused* (connected) to the disc blob as one single connected component. The CC filter sees one large blob with sufficient height and rejects nothing.

**Diagnostic evidence:** Frame 1's largest CC had `bbox = (x=0, w=977, h=263)` — the 977px width spanning from the left edge to beyond the disc revealed that the left-side ground extension was attached.

**Solution:** For every column within the blob, count how many rows are filled. If fewer than `MIN_THICKNESS` rows are filled in that column, the column belongs to a thin extension (ground contact or edge artifact) rather than the solid disc body — zero it out. Repeat symmetrically for rows.

```python
def strip_thin_extensions(blob_mask, min_thickness):
    """
    Remove columns and rows whose pixel fill is below min_thickness.
    This surgically detaches thin horizontal/vertical extensions
    from the main disc body without affecting the body itself.
    """
    cleaned = blob_mask.copy()

    # --- Column pass ---
    col_fills = np.sum(blob_mask > 0, axis=0)   # shape: (W,)
    cleaned[:, col_fills < min_thickness] = 0

    # --- Row pass (on already column-cleaned mask) ---
    row_fills = np.sum(cleaned > 0, axis=1)      # shape: (H,)
    cleaned[row_fills < min_thickness, :] = 0

    return cleaned
```

**Adaptive threshold (resolution-independent):**

```python
MIN_THICKNESS = max(28, int(min(H, W) * 0.06))
# 676px  image → 41px
# 1416px image → 85px
```

**Visualization:** The pipeline figure's Column ④ shows stripped pixels in blue and kept pixels in green, allowing immediate visual verification.

After stripping, close small holes introduced by the operation:

```python
stripped = strip_thin_extensions(single_blob_mask, MIN_THICKNESS)
stripped = cv2.morphologyEx(stripped, cv2.MORPH_CLOSE, k5, iterations=2)
```

---

### Step 6 — Ellipse Fitting & Centroid

`cv2.fitEllipse` fits a minimum-area ellipse to the contour points. For a rolling disc viewed at an oblique angle, this directly encodes the **disc tilt angle**.

```python
contours, _ = cv2.findContours(kept_mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
flipo_cnt = max(contours, key=cv2.contourArea)

# Centroid from image moments
M  = cv2.moments(flipo_cnt)
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])

# Ellipse: returns ((cx, cy), (major_axis, minor_axis), angle)
ellipse       = cv2.fitEllipse(flipo_cnt)
angle_ellipse = float(ellipse[2])   # degrees, 0–180
```

**Output interpretation:**

| Output | Description |
|--------|-------------|
| `(cx, cy)` | Disc centroid in pixel coordinates |
| `ellipse[1]` | `(major_axis_length, minor_axis_length)` in pixels |
| `angle_ellipse` | Rotation of ellipse major axis, 0°–180°, measured from vertical |

> **Why `fitEllipse` over `minAreaRect`?** `fitEllipse` is fitted to the full contour point set with least-squares, making it more robust to motion-blur asymmetry and partial occlusion by the ground.

---

### Step 7 — White Marker Line Detection

The marker line is always centered on the disc. Detection is therefore restricted to the disc body mask (ROI), eliminating all background interference.

```python
# 1. Create filled body ROI mask
body_roi = np.zeros((H, W), np.uint8)
cv2.drawContours(body_roi, [flipo_cnt], -1, 255, -1)  # filled

# 2. White threshold on ORIGINAL (non-CLAHE) HSV
#    — CLAHE can over-brighten background into the white range
hsv_orig  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
white_all = cv2.inRange(hsv_orig,
                        np.array([0,   0,  130]),
                        np.array([180, 110, 255]))

# 3. Mask to body only
white_in_body = cv2.bitwise_and(white_all, body_roi)
white_in_body = cv2.dilate(white_in_body, np.ones((3,3), np.uint8), iterations=1)

# 4. Among all white blobs inside body, pick the one closest to centroid
#    → eliminates edge highlights / reflection noise
nw, lw, sw, cw = cv2.connectedComponentsWithStats(white_in_body)
best_idx, min_dist = -1, float('inf')
for i in range(1, nw):
    if sw[i, cv2.CC_STAT_AREA] < 5:
        continue
    dist = np.hypot(cw[i][0] - cx, cw[i][1] - cy)
    if dist < min_dist:
        min_dist, best_idx = dist, i

# 5. Fit line to winning blob pixels
pts = np.column_stack(np.where(lw == best_idx)[::-1]).astype(np.float32)
[vx, vy, _, _] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
vx = float(vx.flat[0]); vy = float(vy.flat[0])
line_angle = float(np.degrees(np.arctan2(vy, vx)))   # degrees

# 6. Draw line through centroid
L = 50   # half-length in pixels
line_pts = (
    (int(cx - vx * L), int(cy - vy * L)),
    (int(cx + vx * L), int(cy + vy * L))
)
```

**Why `fitLine` over `HoughLinesP`?**

| | `fitLine` | `HoughLinesP` |
|---|---|---|
| Pixel coverage needed | Low (works on sparse blobs) | Needs connected line pixels |
| Robustness to blur | High | Medium |
| Sub-pixel accuracy | Yes (L2 regression) | No |
| Output | Direction vector | Two endpoints |

---

## 5. Complete Source Code

```python
import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────
#  Helper functions
# ──────────────────────────────────────────────────────────────

def detect_ground_line(img_bgr: np.ndarray) -> int:
    """
    Auto-detect the horizontal ground line by finding the row
    with peak Canny edge density in the bottom 40% of the frame.
    For maximum stability, call once on a reference frame and
    pass the result as ground_y to process_frame().
    """
    H = img_bgr.shape[0]
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    top   = int(H * 0.60)
    return top + int(np.argmax(np.sum(edges[top:, :], axis=1)))


def strip_thin_extensions(blob_mask: np.ndarray,
                          min_thickness: int) -> np.ndarray:
    """
    Remove columns and rows whose filled-pixel count is below
    min_thickness. Detaches thin ground-contact extensions from
    the main disc body without affecting the disc itself.

    Parameters
    ----------
    blob_mask     : single-channel binary mask (uint8)
    min_thickness : minimum acceptable fill per column/row (pixels)

    Returns
    -------
    cleaned binary mask (same shape and dtype as input)
    """
    cleaned = blob_mask.copy()

    col_fills = np.sum(cleaned > 0, axis=0)
    cleaned[:, col_fills < min_thickness] = 0

    row_fills = np.sum(cleaned > 0, axis=1)
    cleaned[row_fills < min_thickness, :] = 0

    return cleaned


# ──────────────────────────────────────────────────────────────
#  Main processing function
# ──────────────────────────────────────────────────────────────

def process_frame(img_bgr: np.ndarray,
                  ground_y: int | None = None) -> dict:
    """
    Full Flipo Flip tracking pipeline (v6.0).

    Parameters
    ----------
    img_bgr  : BGR frame from cv2.imread / VideoCapture
    ground_y : pre-calibrated ground line Y coordinate.
               If None, auto-detected via Canny edge scan.

    Returns
    -------
    dict with keys:
        ground_y        – int
        kept_mask       – binary mask of final Flipo body
        rejected_mask   – binary mask of CC-rejected regions
        strip_debug_mask– binary mask of thickness-stripped pixels
        white_mask      – binary mask of detected white marker
        flipo_cnt       – numpy contour array (or None)
        ellipse         – cv2.fitEllipse result tuple (or None)
        cx, cy          – centroid in pixels (or None)
        angle_ellipse   – ellipse major-axis angle in degrees (or None)
        line_pts        – ((x1,y1),(x2,y2)) of marker line (or None)
        line_angle      – marker line angle in degrees (or None)
        min_thickness   – adaptive thickness threshold used
    """
    H, W = img_bgr.shape[:2]

    # ── Step 1: CLAHE ─────────────────────────────────────────
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_enh  = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]),
                            cv2.COLOR_LAB2BGR)
    hsv      = cv2.cvtColor(img_enh, cv2.COLOR_BGR2HSV)

    # ── Step 2: Blue mask ─────────────────────────────────────
    blue_raw = cv2.inRange(hsv,
                           np.array([85,  40,  40]),
                           np.array([140, 255, 255]))

    # ── Step 3: Morphology ────────────────────────────────────
    k7     = np.ones((7, 7), np.uint8)
    k5     = np.ones((5, 5), np.uint8)
    solid  = cv2.morphologyEx(blue_raw, cv2.MORPH_CLOSE, k7)
    eroded = cv2.erode(solid, k5, iterations=3)
    morph  = cv2.dilate(eroded, k5, iterations=3)

    # ── Step 4: Ground-line CC rejection ──────────────────────
    if ground_y is None:
        ground_y = detect_ground_line(img_bgr)

    GROUND_TOL  = 15
    THICK_LIMIT = 30
    ASPECT_MAX  = 6.0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(morph)
    after_cc      = np.zeros_like(morph)
    rejected_mask = np.zeros_like(morph)

    for i in range(1, n):
        x_s, y_s, w_s, h_s, area = stats[i, :5]
        if area < 100:
            continue
        bottom = y_s + h_s
        aspect = w_s / max(h_s, 1)
        if (bottom >= ground_y - GROUND_TOL and
                h_s < THICK_LIMIT and
                aspect > ASPECT_MAX):
            rejected_mask[labels == i] = 255
        else:
            after_cc[labels == i] = 255

    # ── Step 5: Column / row thickness filter ─────────────────
    MIN_THICKNESS = max(28, int(min(H, W) * 0.06))

    contours_tmp, _ = cv2.findContours(after_cc, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
    contours_tmp = [c for c in contours_tmp if cv2.contourArea(c) > 200]

    kept_mask        = np.zeros_like(morph)
    strip_debug_mask = np.zeros_like(morph)

    if contours_tmp:
        largest_c = max(contours_tmp, key=cv2.contourArea)
        single    = np.zeros_like(morph)
        cv2.drawContours(single, [largest_c], -1, 255, -1)

        stripped         = strip_thin_extensions(single, MIN_THICKNESS)
        strip_debug_mask = cv2.subtract(single, stripped)
        stripped         = cv2.morphologyEx(stripped, cv2.MORPH_CLOSE,
                                            k5, iterations=2)
        kept_mask        = stripped

    # ── Step 6: Ellipse fit & centroid ────────────────────────
    contours_f, _ = cv2.findContours(kept_mask, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    contours_f = [c for c in contours_f if cv2.contourArea(c) > 200]

    cx, cy = None, None
    angle_ellipse, ellipse, flipo_cnt = None, None, None

    if contours_f:
        flipo_cnt = max(contours_f, key=cv2.contourArea)
        M = cv2.moments(flipo_cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        if len(flipo_cnt) >= 5:
            ellipse       = cv2.fitEllipse(flipo_cnt)
            angle_ellipse = float(ellipse[2])

    # ── Step 7: White marker line ─────────────────────────────
    white_mask = np.zeros((H, W), np.uint8)
    line_pts, line_angle = None, None

    if flipo_cnt is not None and cx is not None:
        body_roi = np.zeros((H, W), np.uint8)
        cv2.drawContours(body_roi, [flipo_cnt], -1, 255, -1)

        hsv_orig      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        white_all     = cv2.inRange(hsv_orig,
                                    np.array([0,   0,  130]),
                                    np.array([180, 110, 255]))
        white_in_body = cv2.bitwise_and(white_all, body_roi)
        white_in_body = cv2.dilate(white_in_body,
                                   np.ones((3, 3), np.uint8), iterations=1)
        white_mask    = white_in_body

        nw, lw, sw, cw = cv2.connectedComponentsWithStats(white_in_body)
        best_idx, min_dist = -1, float('inf')
        for i in range(1, nw):
            if sw[i, cv2.CC_STAT_AREA] < 5:
                continue
            dist = np.hypot(cw[i][0] - cx, cw[i][1] - cy)
            if dist < min_dist:
                min_dist, best_idx = dist, i

        if best_idx != -1:
            pts = np.column_stack(
                np.where(lw == best_idx)[::-1]).astype(np.float32)
            if len(pts) >= 2:
                [vx, vy, _, _] = cv2.fitLine(pts, cv2.DIST_L2,
                                              0, 0.01, 0.01)
                lvx = float(vx.flat[0])
                lvy = float(vy.flat[0])
                line_angle = float(np.degrees(np.arctan2(lvy, lvx)))
                L = 50
                line_pts = (
                    (int(cx - lvx * L), int(cy - lvy * L)),
                    (int(cx + lvx * L), int(cy + lvy * L))
                )

    return {
        'ground_y'        : ground_y,
        'kept_mask'       : kept_mask,
        'rejected_mask'   : rejected_mask,
        'strip_debug_mask': strip_debug_mask,
        'white_mask'      : white_mask,
        'flipo_cnt'       : flipo_cnt,
        'ellipse'         : ellipse,
        'cx'              : cx,
        'cy'              : cy,
        'angle_ellipse'   : angle_ellipse,
        'line_pts'        : line_pts,
        'line_angle'      : line_angle,
        'min_thickness'   : MIN_THICKNESS,
    }


# ──────────────────────────────────────────────────────────────
#  Annotation helper
# ──────────────────────────────────────────────────────────────

def draw_result(img_bgr: np.ndarray, res: dict) -> np.ndarray:
    """Compose annotated BGR frame from process_frame() result."""
    out = img_bgr.copy()
    H, W = out.shape[:2]
    gy   = res['ground_y']

    # Ground line
    cv2.line(out, (0, gy), (W, gy), (50, 50, 255), 2)
    cv2.putText(out, f"Ground y={gy}", (8, gy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 255), 1)

    # Body overlay (orange tint)
    ov = np.zeros_like(out)
    ov[res['kept_mask'] > 0] = [255, 140, 0]
    out = cv2.addWeighted(out, 0.75, ov, 0.40, 0)

    # Stripped-away overlay (blue tint)
    ov2 = np.zeros_like(out)
    ov2[res['strip_debug_mask'] > 0] = [0, 0, 200]
    out = cv2.addWeighted(out, 0.85, ov2, 0.50, 0)

    # White marker pixels
    out[res['white_mask'] > 0] = [0, 220, 220]

    if res['flipo_cnt'] is not None:
        cv2.drawContours(out, [res['flipo_cnt']], -1, (0, 255, 80), 2)
    if res['ellipse'] is not None:
        cv2.ellipse(out, res['ellipse'], (0, 210, 255), 2)
    if res['cx'] is not None:
        cv2.circle(out, (res['cx'], res['cy']), 8, (0, 0, 255), -1)
        cv2.putText(out, f"C({res['cx']},{res['cy']})",
                    (res['cx'] + 10, res['cy'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if res['angle_ellipse'] is not None:
            ar = np.radians(res['angle_ellipse'])
            ae = (int(res['cx'] + 50 * np.cos(ar)),
                  int(res['cy'] + 50 * np.sin(ar)))
            cv2.arrowedLine(out, (res['cx'], res['cy']),
                            ae, (0, 210, 255), 2, tipLength=0.3)
    if res['line_pts'] is not None:
        cv2.line(out, res['line_pts'][0], res['line_pts'][1],
                 (0, 255, 255), 3)
        mid = ((res['line_pts'][0][0] + res['line_pts'][1][0]) // 2,
               (res['line_pts'][0][1] + res['line_pts'][1][1]) // 2)
        cv2.putText(out, f"L:{res['line_angle']:.1f}deg",
                    (mid[0] + 6, mid[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return out


# ──────────────────────────────────────────────────────────────
#  Example: process a video file
# ──────────────────────────────────────────────────────────────

def process_video(input_path: str,
                  output_path: str,
                  ground_y: int | None = None) -> None:
    """
    Process an entire video and write annotated output.
    Calibrate ground_y from the first frame if not provided.
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (W, H))

    # Auto-calibrate from first frame
    ret, first = cap.read()
    if not ret:
        return
    if ground_y is None:
        ground_y = detect_ground_line(first)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res         = process_frame(frame, ground_y=ground_y)
        annotated   = draw_result(frame, res)
        out.write(annotated)
        frame_idx  += 1
        if frame_idx % 30 == 0:
            print(f"Frame {frame_idx}: "
                  f"centroid=({res['cx']},{res['cy']})  "
                  f"ellipse={res['angle_ellipse']:.1f}°  "
                  f"line={res['line_angle']:.1f}°")

    cap.release()
    out.release()
    print(f"Done — {frame_idx} frames written to {output_path}")
```

---

## 6. Parameter Reference

| Parameter | Location | Default | Raise to… | Lower to… |
|-----------|----------|---------|-----------|-----------|
| `clipLimit` | Step 1 | `3.0` | More contrast (noisy backgrounds) | Less amplification |
| `tileGridSize` | Step 1 | `(8,8)` | Finer local adaptation | Coarser (faster) |
| HSV H lower | Step 2 | `85` | Exclude more cyan | Include more cyan |
| HSV S lower | Step 2 | `40` | Reject more grey background | Include darker shadows |
| Close kernel | Step 3 | `7×7` | Fill larger holes | Smaller footprint |
| Erode iterations | Step 3 | `3` | Sever thicker connections | Preserve thinner links |
| `GROUND_TOL` | Step 4 | `15` | Include blobs further from ground | Stricter proximity |
| `THICK_LIMIT` | Step 4 | `30` | Reject taller shadow strips | Only reject very thin |
| `ASPECT_MAX` | Step 4 | `6.0` | Reject less extreme strips | Reject more blobs |
| `MIN_THICKNESS` scale | Step 5 | `0.035` | Remove thicker extensions | Keep thinner extensions |
| `MIN_THICKNESS` floor | Step 5 | `20` | Larger minimum | Smaller minimum |
| White V lower | Step 7 | `130` | Only pure white | Include grey regions |
| White S upper | Step 7 | `110` | Include slightly saturated | Only desaturated |

---

## 7. Validation Methods

### Frame-level visual checks
Use the 5-column debug figure:

| Column | What to inspect |
|--------|----------------|
| ① Original + ground | Ground line positioned at actual surface edge |
| ② Raw blue mask | Disc covered; no massive background bleed |
| ③ After CC rejection | Independent ground strips shown in red; disc in green |
| ④ Thickness strip | Fused extensions shown in blue; disc body in green — verify no disc pixels removed |
| ⑤ Final result | Ellipse fits disc outline; white line passes through centre |

### Inter-frame consistency checks

```python
# In your video loop, maintain a short history:
history_cx, history_cy = [], []

res = process_frame(frame, ground_y)
if res['cx'] is not None:
    history_cx.append(res['cx'])
    history_cy.append(res['cy'])

    if len(history_cx) > 2:
        # Flag sudden jumps (likely misdetection)
        jump = np.hypot(history_cx[-1] - history_cx[-2],
                        history_cy[-1] - history_cy[-2])
        if jump > 100:  # pixels — adjust to your frame rate / disc speed
            print(f"WARNING: centroid jump {jump:.0f}px at frame {frame_idx}")
```

### Cross-validation: ellipse vs. line angle

The white marker line angle and the ellipse major axis encode the same physical orientation. Their difference should remain stable across a rolling cycle:

```python
angle_diff = abs(res['angle_ellipse'] - res['line_angle']) % 180
# Expect a roughly constant offset (depends on how you drew the line).
# Sudden changes indicate a failed detection.
```

---

## 8. Known Limitations & Future Work

| Limitation | Impact | Suggested fix |
|------------|--------|---------------|
| `MIN_THICKNESS` is a global threshold | May clip disc at extreme tilt angles (disc appears very thin) | Predict expected thickness from prior frame's ellipse minor-axis |
| White line detection fails when disc is face-on (line foreshortened to a point) | `line_angle` returns None | Fall back to ellipse angle; flag frame |
| Ground line assumed horizontal | Fails if camera is tilted | Fit a line to the detected ground edge instead of using a row index |
| Single largest blob assumed to be Flipo | Fails if a large blue object enters frame | Add colour-histogram or size-continuity filter across frames |
| No temporal smoothing | Jitter in angle output | Apply a Kalman filter or rolling-median on `cx`, `cy`, `angle_ellipse` |
