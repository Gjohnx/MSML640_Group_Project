# services/vision/color_grid_detector.py
from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, Tuple, Optional

__all__ = ["detect_color_grid"]

# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------

def detect_color_grid(
    frame_bgr: np.ndarray,
    debug: bool = True,
    expected_size: int = 300,
    ksize: int = 3,
    min_area_ratio: float = 0.08,          # prefer big faces
    approx_epsilon_ratio: float = 0.02,    # RDP epsilon ratio
    sample_ratio: float = 0.36,            # central sampling square (fraction of a cell side)
    ref_lab: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> Dict:
    """
    Detect a Rubik's cube face in a BGR frame, rectify to 3x3, robustly sample
    per-cell colors, and classify into {W, Y, R, O, B, G}.

    Returns a dict:
        ok: bool
        grid: (3,3) array of str labels ('W','Y','R','O','B','G') or None
        grid_dists: (3,3) array of float LAB distances (smaller = better) or None
        face_lab: (3,3,3) array of LAB medians after correction (float32) or None
        overlay_bgr: BGR image with overlays (same size as input) or None
        warp_bgr: rectified face (expected_size x expected_size) or None
        quad: (4,2) float32 corner points or None
        msg: str message
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return _res(False, "empty frame", None, None, None, None, None, None)

    H, W = frame_bgr.shape[:2]
    overlay = frame_bgr.copy() if debug else None

    # --- 0) light pre-processing (denoise a bit) ---
    img = cv2.GaussianBlur(frame_bgr, (ksize | 1, ksize | 1), 0)

    # --- 1) edges & contours ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    v_low, v_high = _auto_canny_thresholds(gray)
    edges = cv2.Canny(gray, v_low, v_high)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return _res(False, "no contours", overlay, None, None, None, None, None)

    # --- 2) enumerate convex quads, score each by geometry + grid-edge energy ---
    area_min = min_area_ratio * (H * W)
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < area_min:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, approx_epsilon_ratio * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        pts = approx.reshape(-1, 2).astype(np.float32)

        # loose aspect ratio pre-filter (avoid very skinny faces)
        x, y, w, h = cv2.boundingRect(approx)
        aspect = w / float(h + 1e-6)
        if not (0.70 <= aspect <= 1.35):
            continue

        # score this quad
        score, meta = _score_quad(frame_bgr, pts, expected_size)
        candidates.append((score, pts, meta))

    if not candidates:
        return _res(False, "no valid quad candidates", overlay, None, None, None, None, None)

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_quad, _ = candidates[0]
    quad = _order_corners(best_quad)

    if debug and overlay is not None:
        cv2.polylines(overlay, [quad.astype(np.int32)], True, (0, 255, 255), 2)
        center = tuple(np.int32(quad.mean(axis=0)))
        cv2.putText(overlay, f"Detected 3x3 face  score={best_score:.0f}",
                    (max(10, center[0]-140), max(25, center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 3)

    # --- 3) perspective warp ---
    warp_bgr, _ = _warp_face(frame_bgr, quad, expected_size)

    # --- 4) robust sampling per cell (LAB median after masking) ---
    cell = expected_size // 3
    lab = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2LAB)
    face_lab = np.zeros((3, 3, 3), dtype=np.float32)
    for r in range(3):
        for c_ in range(3):
            cx = c_ * cell + cell // 2
            cy = r * cell + cell // 2
            face_lab[r, c_, :] = _robust_cell_lab(lab, warp_bgr, cx, cy, cell, sample_ratio)
            if debug:
                pad = int(cell * sample_ratio / 2.0)
                x0, x1 = max(0, cx - pad), min(expected_size, cx + pad)
                y0, y1 = max(0, cy - pad), min(expected_size, cy + pad)
                cv2.rectangle(warp_bgr, (x0, y0), (x1, y1), (0, 255, 0), 1)

    # --- 4.5) illumination neutrality on a/b (face-wise bias removal) ---
    face_lab_corr = _neutralize_ab(face_lab, chroma_thresh=30.0)

    # --- 4.6) dynamic white reference from this face ---
    dyn_white = _compute_dynamic_white_ref(face_lab_corr)
    refs = _default_refs_lab() if ref_lab is None else ref_lab
    if dyn_white is not None:
        refs = dict(refs)        # copy to avoid mutating caller dict
        refs["W"] = dyn_white    # replace static white with dynamic white

    # --- 5) classify (HSV coarse -> LAB refine) + strong white protection ---
    grid_labels = np.empty((3, 3), dtype="<U1")
    grid_dists  = np.zeros((3, 3), dtype=np.float32)

    for r in range(3):
        for c_ in range(3):
            cx = c_ * cell + cell // 2
            cy = r * cell + cell // 2
            bgr_center = warp_bgr[cy, cx, :]
            lab_vec = face_lab_corr[r, c_, :]  # corrected LAB

            # ---- strong white protection: bright & nearly achromatic
            L, a, b = float(lab_vec[0]), float(lab_vec[1]), float(lab_vec[2])
            chroma = (a*a + b*b) ** 0.5
            forced_white = (L > 98.0 and chroma < 14.0)

            if forced_white and "W" in refs:
                best = "W"
                best_d = float(np.linalg.norm(lab_vec - np.array(refs["W"], dtype=np.float32)))
            else:
                # HSV coarse family
                coarse = _hsv_coarse_label(bgr_center)
                if coarse in ("R","O","Y","G","B","W"):
                    candidates = {k:v for k,v in refs.items() if (
                        (coarse == "R" and k in ("R",)) or
                        (coarse == "O" and k in ("O","R","Y")) or
                        (coarse == "Y" and k in ("Y","O","W")) or
                        (coarse == "G" and k in ("G",)) or
                        (coarse == "B" and k in ("B",)) or
                        (coarse == "W" and k in ("W","Y"))
                    )}
                    if not candidates:
                        candidates = refs
                else:
                    candidates = refs

                # nearest in LAB among candidates
                best, best_d = None, 1e9
                for k, ref in candidates.items():
                    d = float(np.linalg.norm(lab_vec - np.array(ref, dtype=np.float32)))
                    if d < best_d:
                        best, best_d = k, d

                # ---- Yellow -> White fallback (distance margin + L/chroma gate)
                if "W" in refs and "Y" in refs:
                    dW = float(np.linalg.norm(lab_vec - np.array(refs["W"], dtype=np.float32)))
                    dY = float(np.linalg.norm(lab_vec - np.array(refs["Y"], dtype=np.float32)))
                    # margin avoids jitter; also allow in ultra-bright low-chroma area
                    if (dW + 4.0 < dY) or (L > max(90.0, refs["W"][0] - 2.0) and chroma < 18.0):
                        best, best_d = "W", dW

            grid_labels[r, c_] = best
            grid_dists[r, c_]  = best_d

    # --- 6) debug overlays on warp (grid lines + labels) ---
    if debug:
        for i in range(1, 3):
            x = i * cell
            y = i * cell
            cv2.line(warp_bgr, (x, 0), (x, expected_size - 1), (255, 255, 255), 1)
            cv2.line(warp_bgr, (0, y), (expected_size - 1, y), (255, 255, 255), 1)

        for r in range(3):
            for c_ in range(3):
                cx = c_ * cell + cell // 2
                cy = r * cell + cell // 2
                cv2.putText(
                    warp_bgr, str(grid_labels[r, c_]),
                    (cx - 10, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )

        # small warp preview on overlay
        if overlay is not None:
            h_small = 180
            ratio = h_small / warp_bgr.shape[0]
            w_small = int(warp_bgr.shape[1] * ratio)
            warp_small = cv2.resize(warp_bgr, (w_small, h_small))
            overlay[10:10 + h_small, 10:10 + w_small] = warp_small

    return {
        "ok": True,
        "grid": grid_labels,
        "grid_dists": grid_dists,
        "face_lab": face_lab_corr,   # return corrected LAB for transparency
        "overlay_bgr": overlay,
        "warp_bgr": warp_bgr,
        "quad": quad,
        "msg": "ok",
    }

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _auto_canny_thresholds(gray: np.ndarray) -> Tuple[int, int]:
    """Choose Canny thresholds from median intensity (robust to lighting)."""
    v = float(np.median(gray))
    low = int(max(0, 0.66 * v))
    high = int(min(255, 1.33 * v + 40))
    if low >= high:
        low = max(0, high - 50)
    return low, high


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [tl, tr, br, bl] for perspective transform."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _warp_face(frame_bgr: np.ndarray, quad: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Perspective warp the quad to a frontal square of `size`."""
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    warp = cv2.warpPerspective(frame_bgr, M, (size, size), flags=cv2.INTER_LINEAR)
    return warp, M


def _angle_deviation_from_right(quad: np.ndarray) -> float:
    """
    Total deviation (degrees) of the 4 interior angles from 90°.
    Smaller is better (more rectangular / front-facing).
    """
    pts = _order_corners(quad).astype(np.float32)
    total = 0.0
    for i in range(4):
        a = pts[i] - pts[(i - 1) % 4]
        b = pts[(i + 1) % 4] - pts[i]
        na = a / (np.linalg.norm(a) + 1e-6)
        nb = b / (np.linalg.norm(b) + 1e-6)
        cosang = np.clip(np.dot(na, nb), -1.0, 1.0)
        ang = np.degrees(np.arccos(cosang))
        total += abs(90.0 - ang)
    return total


def _score_quad(frame_bgr: np.ndarray, quad: np.ndarray, size: int) -> tuple[float, dict]:
    """
    Project 'quad' to a frontal square, then measure:
      - grid-edge energy along theoretical 3x3 lines (stickers have black seams),
      - geometric right-angle fitness,
      - color variance (single face has diverse colors).
    Return a combined score (higher is better) and debug meta.
    """
    # geometry
    angle_dev = _angle_deviation_from_right(quad)  # smaller is better
    geom_score = max(0.0, 120.0 - angle_dev)

    # warp
    warp, _ = _warp_face(frame_bgr, _order_corners(quad), size)
    gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # grid-edge energy
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    cell = size // 3
    line_w = max(1, cell // 20)
    edge_energy = 0.0

    for i in (1, 2):
        x = i * cell
        band = mag[:, max(0, x - line_w):min(size, x + line_w)]
        edge_energy += float(band.mean())

    for i in (1, 2):
        y = i * cell
        band = mag[max(0, y - line_w):min(size, y + line_w), :]
        edge_energy += float(band.mean())

    # color variance (per face colors differ)
    lab = cv2.cvtColor(warp, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    color_var = float(lab.var(axis=0).mean())

    score = edge_energy * 3.0 + geom_score * 1.5 + color_var * 0.5
    meta = {
        "edge_energy": edge_energy,
        "geom_score": geom_score,
        "color_var": color_var,
        "angle_dev": angle_dev,
    }
    return score, meta


def _robust_cell_lab(lab_img: np.ndarray, bgr_img: np.ndarray, cx: int, cy: int,
                     cell: int, sample_ratio: float) -> np.ndarray:
    """
    Robust cell sampling:
      - sample a smaller inner box,
      - mask out very dark pixels (black borders) and specular highlights,
      - return median LAB (more robust than mean).
    """
    h, w = lab_img.shape[:2]
    pad = int(cell * sample_ratio / 2.0)
    x0, x1 = max(0, cx - pad), min(w, cx + pad)
    y0, y1 = max(0, cy - pad), min(h, cy + pad)

    lab_patch = lab_img[y0:y1, x0:x1].reshape(-1, 3)
    if lab_patch.size == 0:
        return np.array([0., 0., 0.], dtype=np.float32)

    bgr_patch = bgr_img[y0:y1, x0:x1].reshape(-1, 3).astype(np.uint8)

    # HSV-based masking (remove black border & highlights)
    hsv_patch = cv2.cvtColor(bgr_patch.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    V = hsv_patch[:, 2].astype(np.float32)
    v_low  = np.percentile(V, 5)     # near-black borders
    v_high = np.percentile(V, 96)    # stricter highlight removal
    good = (V >= v_low) & (V <= v_high)

    filtered = lab_patch[good]
    if filtered.size == 0:
        filtered = lab_patch  # fallback

    med = np.median(filtered, axis=0).astype(np.float32)
    return med


def _hsv_coarse_label(bgr_color: np.ndarray) -> Optional[str]:
    """
    Coarse label in HSV space to separate families (R/O/Y/G/B/W).
    Returns 'R','O','Y','G','B','W' or None if uncertain.
    """
    hsv = cv2.cvtColor(bgr_color.reshape(1, 1, 3).astype(np.uint8),
                       cv2.COLOR_BGR2HSV).reshape(3,)
    h, s, v = float(hsv[0]), float(hsv[1]), float(hsv[2])

    # tighter white: very low saturation & very high value
    if s < 30 and v > 200:
        return "W"

    # tighten saturation/value to avoid warm whites being classified as yellow
    if s > 60 and v > 80:
        if (h <= 10) or (h >= 170):
            return "R"
        if 10 < h <= 22:
            return "O"
        if 22 < h <= 38:   # narrower yellow range
            return "Y"
        if 38 < h <= 85:
            return "G"
        if 85 < h <= 140:
            return "B"
    return None


def _default_refs_lab() -> Dict[str, Tuple[float, float, float]]:
    """
    Reference LAB centers for cube stickers (coarse defaults).
    You can override via `ref_lab` or run a quick calibration.
    """
    # (L, a, b) approximate centers under D65
    return {
        "W": (96, 0, 0),
        "Y": (88, -6, 100),
        "R": (53, 80, 67),
        "O": (66, 40, 78),
        "B": (32, 12, -55),
        "G": (87, -80, 75),
    }


def _neutralize_ab(face_lab: np.ndarray, chroma_thresh: float = 30.0) -> np.ndarray:
    """
    Estimate warm/cool illumination bias from low-chroma cells on this face,
    then subtract that bias from *all* cells' a/b. (Keep L unchanged.)
    This makes whites truly achromatic and reduces W<->Y confusion.
    """
    corr = face_lab.copy().astype(np.float32)
    a = corr[..., 1]
    b = corr[..., 2]
    chroma = np.sqrt(a*a + b*b)

    mask = chroma < chroma_thresh  # "near-achromatic" cells
    if np.any(mask):
        bias_a = float(a[mask].mean())
        bias_b = float(b[mask].mean())
        corr[..., 1] -= bias_a
        corr[..., 2] -= bias_b
    return corr


def _compute_dynamic_white_ref(face_lab_corr: np.ndarray) -> Optional[tuple[float,float,float]]:
    """
    Pick a dynamic white reference from this face:
    - choose cells with low chroma
    - among them take the HIGH-L group (top-2~3 by L) and median as white ref
    Returns (L,a,b) or None if not enough candidates.
    """
    L = face_lab_corr[..., 0].astype(np.float32)
    a = face_lab_corr[..., 1].astype(np.float32)
    b = face_lab_corr[..., 2].astype(np.float32)
    chroma = np.sqrt(a*a + b*b)

    low_chroma = chroma < 20.0  # “near-achromatic”
    if not np.any(low_chroma):
        return None
    cand_idx = np.where(low_chroma)
    Lc = L[cand_idx]
    ac = a[cand_idx]
    bc = b[cand_idx]

    # take top-L subset to avoid picking shaded tiles
    if Lc.size >= 2:
        top_k = min(3, Lc.size)
        top_sel = np.argsort(-Lc)[:top_k]
        Lc, ac, bc = Lc[top_sel], ac[top_sel], bc[top_sel]

    return float(np.median(Lc)), float(np.median(ac)), float(np.median(bc))


def _res(ok: bool, msg: str, overlay, warp, quad, grid, face_lab, grid_dists):
    return {
        "ok": ok,
        "grid": grid,
        "grid_dists": grid_dists,
        "face_lab": face_lab,
        "overlay_bgr": overlay,
        "warp_bgr": warp,
        "quad": quad,
        "msg": msg,
    }
