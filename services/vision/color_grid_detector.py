
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import cv2

# ---- Tunable parameters ----
WARP_SIZE = 450           # Output warped face size (square)
SAMPLE_RATIO = 0.4        # Fraction of each grid cell used for center sampling
# HSV ranges (OpenCV: H∈[0,179], S∈[0,255], V∈[0,255])
COLOR_RANGES = {
    "W": [((0, 0, 180), (179, 60, 255))],               # low S, high V
    "Y": [((18, 80, 140), (35, 255, 255))],
    "O": [((8, 100, 120), (18, 255, 255))],
    "R": [((0, 100, 100), (6, 255, 255)), ((170, 100, 100), (179, 255, 255))],
    "G": [((45, 60, 80), (85, 255, 255))],
    "B": [((95, 80, 80), (130, 255, 255))],
}
# Priority when multiple ranges hit (helps resolve borderline conflicts)
COLOR_PRIORITY = ["R", "O", "Y", "G", "B", "W"]


@dataclass
class DetectedFace:
    """Result container for a single-face detection."""
    labels: List[List[str]]       # 3x3 color labels
    warped_bgr: np.ndarray        # warped face (BGR)
    grid_overlay: np.ndarray      # warped face with grid & sample boxes drawn (BGR)


def detect_face_3x3_from_bgr(frame_bgr: np.ndarray) -> Optional[DetectedFace]:
    """
    Detect a single cube face in a BGR frame and classify a 3x3 color grid.
    Returns None if no face-like quadrilateral is found.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    quad = _find_largest_square_quad(frame_bgr)
    if quad is None:
        return None

    warped = _warp_perspective(frame_bgr, quad, WARP_SIZE)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    h_step, w_step = warped.shape[0] // 3, warped.shape[1] // 3
    labels: List[List[str]] = []
    overlay = warped.copy()

    for r in range(3):
        row = []
        for c in range(3):
            # Compute center sampling box within each grid cell
            y0, x0 = r * h_step, c * w_step
            cy = y0 + int(h_step * (1 - SAMPLE_RATIO) / 2)
            cx = x0 + int(w_step * (1 - SAMPLE_RATIO) / 2)
            ch = int(h_step * SAMPLE_RATIO)
            cw = int(w_step * SAMPLE_RATIO)

            roi = hsv[cy:cy + ch, cx:cx + cw]
            mean_hsv = roi.reshape(-1, 3).mean(axis=0)  # [H, S, V]
            label = _classify_hsv(mean_hsv)
            row.append(label)

            # Draw sample box & put label for visual debugging
            cv2.rectangle(overlay, (cx, cy), (cx + cw, cy + ch), (0, 255, 255), 1)
            cv2.putText(overlay, label, (x0 + 10, y0 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay, label, (x0 + 10, y0 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1, cv2.LINE_AA)
        labels.append(row)

    # Draw 3x3 grid lines
    for i in range(1, 3):
        cv2.line(overlay, (0, i * h_step), (WARP_SIZE, i * h_step), (0, 255, 0), 1)
        cv2.line(overlay, (i * w_step, 0), (i * w_step, WARP_SIZE), (0, 255, 0), 1)

    return DetectedFace(labels=labels, warped_bgr=warped, grid_overlay=overlay)


# ------------------ helpers ------------------

def _find_largest_square_quad(bgr: np.ndarray) -> Optional[np.ndarray]:
    """Find the largest convex 4-pt contour (square-like); return 4x2 float32 points ordered TL,TR,BR,BL in original scale."""
    img = bgr.copy()
    ratio = 720.0 / max(1, max(img.shape[:2]))
    if ratio < 1.0:
        img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_area = 0
    img_area = img.shape[0] * img.shape[1]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > best_area and area > 0.05 * img_area:
                best = approx
                best_area = area

    if best is None:
        return None

    quad = best.reshape(-1, 2).astype(np.float32) / ratio
    return _order_points_tl_tr_br_bl(quad)


def _order_points_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL using sum/diff heuristics."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    bl = pts[np.argmin(d)]
    tr = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _warp_perspective(bgr: np.ndarray, quad: np.ndarray, size: int) -> np.ndarray:
    """Perspective-warp the quadrilateral region to a size×size square image."""
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(bgr, M, (size, size))


def _classify_hsv(mean_hsv: np.ndarray) -> str:
    """Classify HSV by rule ranges first; fall back to nearest prototype if no range hits."""
    H, S, V = float(mean_hsv[0]), float(mean_hsv[1]), float(mean_hsv[2])
    hits = []
    for label, ranges in COLOR_RANGES.items():
        for (lo, hi) in ranges:
            (h1, s1, v1), (h2, s2, v2) = lo, hi
            if h1 <= H <= h2 and s1 <= S <= s2 and v1 <= V <= v2:
                hits.append(label)
                break
    if hits:
        for p in COLOR_PRIORITY:
            if p in hits:
                return p

    # Nearest-prototype fallback (tolerant to lighting)
    proto = {
        "W": (0, 10, 245),
        "Y": (28, 200, 200),
        "O": (14, 200, 200),
        "R": (0, 200, 200),
        "G": (65, 200, 180),
        "B": (110, 200, 180),
    }
    best, best_d = "W", 1e9
    for label, (h0, s0, v0) in proto.items():
        dh = min(abs(H - h0), 180 - abs(H - h0)) / 20.0
        ds = abs(S - s0) / 60.0
        dv = abs(V - v0) / 60.0
        d = (dh ** 2 + ds ** 2 + dv ** 2) ** 0.5
        if d < best_d:
            best, best_d = label, d
    return best