# integrated/services/detection_methods/qbr_simple.py


from __future__ import annotations
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import cv2

from .base import DetectionMethod



def _bgr_to_lab_list(bgr: Tuple[int, int, int]) -> List[float]:
    """Convert a BGR color to CIE L*a*b* as [L, a, b] without cv2.cvtColor."""
    b, g, r = bgr
    R, G, B = (r / 255.0), (g / 255.0), (b / 255.0)

    def _to_linear(c: float) -> float:
        return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92

    R, G, B = _to_linear(R), _to_linear(G), _to_linear(B)

    # Linear RGB -> XYZ (D65)
    X = R * 0.4124 + G * 0.3576 + B * 0.1805
    Y = R * 0.2126 + G * 0.7152 + B * 0.0722
    Z = R * 0.0193 + G * 0.1192 + B * 0.9505

    # Normalize by D65 white point
    X /= 0.95047; Y /= 1.00000; Z /= 1.08883

    def _f(t: float) -> float:
        return t ** (1.0 / 3.0) if t > 0.008856 else (7.787 * t) + (16.0 / 116.0)

    fX, fY, fZ = _f(X), _f(Y), _f(Z)
    L = (116.0 * fY) - 16.0
    a = 500.0 * (fX - fY)
    b_ = 200.0 * (fY - fZ)
    return [L, a, b_]


def _ciede2000(Lab1: List[float], Lab2: List[float]) -> float:
    """Compute the CIEDE2000 color difference between two Lab colors."""
    import math
    L1, a1, b1 = Lab1
    L2, a2, b2 = Lab2

    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_ave = (C1 + C2) / 2.0

    G = 0.5 * (1.0 - math.sqrt((C_ave**7) / (C_ave**7 + 25.0**7)))
    a1p, a2p = (1.0 + G) * a1, (1.0 + G) * a2
    C1p = math.sqrt(a1p**2 + b1**2)
    C2p = math.sqrt(a2p**2 + b2**2)

    def _hp(ap: float, bp: float) -> float:
        if ap == 0.0 and bp == 0.0:
            return 0.0
        h = math.atan2(bp, ap)
        return h + 2.0 * math.pi if h < 0.0 else h

    h1p = _hp(a1p, b1)
    h2p = _hp(a2p, b2)

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = 0.0
    if C1p * C2p != 0.0:
        dh = h2p - h1p
        if dh > math.pi: dh -= 2.0 * math.pi
        elif dh < -math.pi: dh += 2.0 * math.pi
        dhp = 2.0 * math.sqrt(C1p * C2p) * math.sin(dh / 2.0)

    Lp_ave = (L1 + L2) / 2.0
    Cp_ave = (C1p + C2p) / 2.0

    if C1p * C2p == 0.0:
        hp_ave = h1p + h2p
    else:
        if abs(h1p - h2p) > math.pi:
            hp_ave = (h1p + h2p + 2.0 * math.pi) / 2.0 if h1p + h2p < 2.0 * math.pi else (h1p + h2p - 2.0 * math.pi) / 2.0
        else:
            hp_ave = (h1p + h2p) / 2.0

    T = (1.0
         - 0.17 * math.cos(hp_ave - math.pi / 6.0)
         + 0.24 * math.cos(2.0 * hp_ave)
         + 0.32 * math.cos(3.0 * hp_ave + math.pi / 30.0)
         - 0.20 * math.cos(4.0 * hp_ave - 63.0 * math.pi / 180.0))

    dTheta = 30.0 * math.exp(-((hp_ave * 180.0 / math.pi - 275.0) / 25.0)**2)
    R_C = 2.0 * math.sqrt((Cp_ave**7) / (Cp_ave**7 + 25.0**7))
    S_L = 1.0 + (0.015 * (Lp_ave - 50.0)**2 / math.sqrt(20.0 + (Lp_ave - 50.0)**2))
    S_C = 1.0 + 0.045 * Cp_ave
    S_H = 1.0 + 0.015 * Cp_ave * T
    R_T = -math.sin(2.0 * dTheta * math.pi / 180.0) * R_C

    return math.sqrt((dLp / S_L)**2 + (dCp / S_C)**2 + (dhp / S_H)**2 + R_T * (dCp / S_C) * (dhp / S_H))



class QbrDetectionMethod(DetectionMethod):
    """
    QBR-inspired detector that identifies a 3×3 Rubik's cube face from a frame
    and classifies colors using a camera-realistic palette. Implements the
    DetectionMethod API: returns a processed BGR frame, a 6×3×3 array of letters,
    and a placeholder rotation (x,y,z).
    """

    name = "QBR (Simple)"

    def __init__(self) -> None:
      super().__init__()

      # Camera-realistic initial palette (BGR). Can be replaced via calibration.
      self._prominent_palette: Dict[str, Tuple[int, int, int]] = {
          "red":    (40,  40, 210),
          "orange": (0,  120, 255),
          "blue":   (180, 80,  30),
          "green":  (40, 180,  40),
          "white":  (225, 225, 225),
          "yellow": (40, 210, 210),
      }
      self.cube_color_palette: Dict[str, Tuple[int, int, int]] = dict(self._prominent_palette)

      # Map color names to cube notation
    #   self._color_to_notation: Dict[str, str] = {
    #       "white":  "U",
    #       "red":    "R",
    #       "green":  "F",
    #       "yellow": "D",
    #       "orange": "L",
    #       "blue":   "B",
    #   }
      self._color_to_notation: Dict[str, str] = {
          "white":  "U",
          "red":    "F",
          "green":  "L",
          "yellow": "D",
          "orange": "B",
          "blue":   "R",
      }
      # Temporal smoothing
      self._avg_color_buffers: Dict[int, List[Tuple[int, int, int]]] = {}
      self._stable_indices: set[int] = set()

      # Calibration state
      self._calibrating: bool = False
      self._calibrated_colors: Dict[str, Tuple[int, int, int]] = {}
      self._colors_to_calibrate: List[str] = ["green", "red", "blue", "orange", "white", "yellow"]
      self._calibrate_index: int = 0

    # --------------------------- Framework API --------------------------------

    def reset(self) -> None:
        """Reset temporal and calibration states."""
        self._avg_color_buffers.clear()
        self._stable_indices.clear()
        self._calibrating = False
        self._calibrated_colors.clear()
        self._calibrate_index = 0

    def start_calibration(self) -> None:
        """Enter calibration mode."""
        self._calibrating = True
        self._calibrated_colors.clear()
        self._calibrate_index = 0

    def capture_calibration_frame(self, frame_bgr: np.ndarray) -> bool:
        """
        Use current frame to capture center sticker color for calibration.
        Returns True when all six colors are collected.
        """
        if not self._calibrating or self._calibrate_index >= len(self._colors_to_calibrate):
            return True

        # Preprocess like detection
        bgr_f = frame_bgr.astype(np.float32)
        mean_b, mean_g, mean_r = bgr_f[...,0].mean(), bgr_f[...,1].mean(), bgr_f[...,2].mean()
        gray_mean = (mean_b + mean_g + mean_r) / 3.0 + 1e-6
        gain_b, gain_g, gain_r = gray_mean/mean_b, gray_mean/mean_g, gray_mean/mean_r
        # Clamp gains to avoid over-correction
        gain_b = np.clip(gain_b, 0.85, 1.15)
        gain_g = np.clip(gain_g, 0.85, 1.15)
        gain_r = np.clip(gain_r, 0.85, 1.15)
        bgr_f[...,0] *= gain_b; bgr_f[...,1] *= gain_g; bgr_f[...,2] *= gain_r
        bgr_f = np.clip(bgr_f, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(bgr_f, cv2.COLOR_BGR2HSV)
        S = hsv[...,1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        S_eq = clahe.apply(S)

        v = float(np.median(S_eq))
        low = int(max(0, 0.66 * v))
        high = int(min(255, 1.33 * v))
        edges = cv2.Canny(S_eq, low, high, apertureSize=3, L2gradient=True)

        k = max(3, int(min(frame_bgr.shape[:2]) * 0.02))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        dilated = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        dilated = cv2.dilate(dilated, kernel, iterations=1)

        contours = self._find_face_contours(dilated)
        if len(contours) == 9:
            # Center sticker (index 4)
            x, y, w, h = contours[4]
            roi = frame_bgr[
                y + max(2, h // 10): y + h - max(2, h // 10),
                x + max(2, w // 6):  x + w - max(2, w // 6)
            ]
            avg_bgr = self._get_dominant_color(roi)
            color_name = self._colors_to_calibrate[self._calibrate_index]
            self._calibrated_colors[color_name] = avg_bgr
            self._calibrate_index += 1

            if self._calibrate_index == len(self._colors_to_calibrate):
                for name, bgr in self._calibrated_colors.items():
                    self.cube_color_palette[name] = tuple(int(c) for c in bgr)
                self._calibrating = False
                return True
        return False

    def process(
        self, frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Tuple[float, float, float]]]:
        """
        Detect a 3x3 face and classify colors.
        Returns:
          processed_frame (BGR), cube_colors (6×3×3 letters or None), rotation (x,y,z).
        """
        out = frame_bgr.copy()
        cube_face_labels: Optional[np.ndarray] = None
        rotation = (15.0, 15.0, 0.0)

        # --- Robust preprocessing ---
        bgr_f = frame_bgr.astype(np.float32)
        mean_b, mean_g, mean_r = bgr_f[...,0].mean(), bgr_f[...,1].mean(), bgr_f[...,2].mean()
        gray_mean = (mean_b + mean_g + mean_r) / 3.0 + 1e-6
        gain_b, gain_g, gain_r = gray_mean/mean_b, gray_mean/mean_g, gray_mean/mean_r
        gain_b = np.clip(gain_b, 0.85, 1.15)
        gain_g = np.clip(gain_g, 0.85, 1.15)
        gain_r = np.clip(gain_r, 0.85, 1.15)
        bgr_f[...,0] *= gain_b; bgr_f[...,1] *= gain_g; bgr_f[...,2] *= gain_r
        bgr_f = np.clip(bgr_f, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(bgr_f, cv2.COLOR_BGR2HSV)
        S = hsv[...,1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        S_eq = clahe.apply(S)

        v = float(np.median(S_eq))
        low = int(max(0, 0.66 * v))
        high = int(min(255, 1.33 * v))
        edges = cv2.Canny(S_eq, low, high, apertureSize=3, L2gradient=True)

        k = max(3, int(min(frame_bgr.shape[:2]) * 0.02))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        dilated = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        dilated = cv2.dilate(dilated, kernel, iterations=1)

        # --- Find 3x3 grid ---
        contours = self._find_face_contours(dilated)
        if len(contours) == 9:
            # Draw green rectangles
            for (x, y, w, h) in contours:
                cv2.rectangle(out, (x, y), (x + w, y + h), (36, 255, 12), 2)

            if not self._calibrating:
                detected_names: List[str] = ["" for _ in range(9)]
                for idx, (x, y, w, h) in enumerate(contours):
                    # Crop inside the sticker to avoid borders/glare
                    roi = frame_bgr[
                        y + max(2, h // 10): y + h - max(2, h // 10),
                        x + max(2, w // 6):  x + w - max(2, w // 6)
                    ]
                    avg_bgr = self._get_dominant_color(roi)
                    nearest = self._get_closest_color(avg_bgr)
                    color_name = nearest["color_name"] or "?"

                    # Temporal smoothing (majority over last N)
                    buf = self._avg_color_buffers.get(idx, [])
                    buf.append(self.cube_color_palette.get(color_name, (0, 0, 0)))
                    if len(buf) > 8:
                        buf = buf[-8:]
                    self._avg_color_buffers[idx] = buf

                    if idx not in self._stable_indices and len(buf) == 8:
                        counts: Dict[Tuple[int, int, int], int] = {}
                        for bgr in buf:
                            key = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
                            counts[key] = counts.get(key, 0) + 1
                        most_common_bgr = max(counts, key=counts.get)
                        stable_name = color_name
                        for name, bgr in self.cube_color_palette.items():
                            if (int(bgr[0]), int(bgr[1]), int(bgr[2])) == most_common_bgr:
                                stable_name = name
                                break
                        color_name = stable_name
                        self._stable_indices.add(idx)
                        self._avg_color_buffers[idx] = []

                    detected_names[idx] = color_name

                    # ----------------- NEW: draw the predicted letter on each box -----------------
                    # Derive the cube notation letter (U/R/F/D/L/B) for this sticker.
                    letter = self._color_to_notation.get(color_name, "?")

                    # Compute text size and center it in the rectangle.
                    # Use thicker black outline + white text for better contrast.
                    text_scale = max(0.5, min(w, h) / 55.0)  # scale by sticker size
                    text_thick_body = max(2, int(round(text_scale * 2.0)))
                    text_thick_outline = text_thick_body + 2
                    (tw, th), baseline = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thick_body)

                    cx = x + w // 2
                    cy = y + h // 2
                    org = (int(cx - tw / 2), int(cy + th / 2))  # baseline near vertical center

                    # Outline first (black), then body (white)
                    cv2.putText(out, letter, org, cv2.FONT_HERSHEY_SIMPLEX,
                                text_scale, (0, 0, 0), text_thick_outline, cv2.LINE_AA)
                    cv2.putText(out, letter, org, cv2.FONT_HERSHEY_SIMPLEX,
                                text_scale, (255, 255, 255), text_thick_body, cv2.LINE_AA)
                    # ------------------------------------------------------------------------------

                # Map to 6×3×3 (Front face index = 2)
                face_letters = [
                    [self._color_to_notation.get(n, "?") for n in detected_names[i:i + 3]]
                    for i in range(0, 9, 3)
                ]
                cube_face_labels = np.full((6, 3, 3), "?", dtype="<U2")
                cube_face_labels[2] = np.array(face_letters, dtype="<U2")
        else:
            cv2.putText(out, "Face not found (adjust distance/lighting)",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 200, 255), 2, cv2.LINE_AA)

        return out, cube_face_labels, rotation

    # --------------------------------- Helpers --------------------------------

    def _find_face_contours(self, dilated_frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Find a 3×3 grid of sticker-like contours via two-pass candidate selection.
        Returns a list of 9 (x, y, w, h) or [] if not found.
        """
        H, W = dilated_frame.shape[:2]

        def _collect_candidates(relax: bool = False):
            min_side = int((0.035 if relax else 0.045) * min(H, W))
            max_side = int((0.30  if relax else 0.24 ) * min(H, W))
            contours, hierarchy = cv2.findContours(dilated_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is None:
                return []
            cands = []
            for cnt in contours:
                if len(cnt) < 4:
                    continue
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, (0.06 if relax else 0.04) * peri, True)
                if len(approx) != 4:
                    continue
                x, y, w, h = cv2.boundingRect(approx)
                if w < min_side or h < min_side or w > max_side or h > max_side:
                    continue
                ar = w / float(h)
                if not (0.70 if relax else 0.75) <= ar <= (1.30 if relax else 1.25):
                    continue
                area = cv2.contourArea(cnt)
                fill = area / float(max(w * h, 1))
                if fill < (0.18 if relax else 0.28):
                    continue
                cx, cy = x + w * 0.5, y + h * 0.5
                cands.append((x, y, w, h, cx, cy, float(w * h)))
            return cands

        def _pick_grid(candidates):
            if len(candidates) < 9:
                return None
            areas = np.array([c[-1] for c in candidates], dtype=np.float32)
            med = float(np.median(areas))
            candidates.sort(key=lambda c: abs(c[-1] - med))
            picked = candidates[:max(9, min(16, len(candidates)))]
            picked.sort(key=lambda c: c[5])  # by cy

            best, best_score = None, 1e9
            for start in range(0, max(1, len(picked) - 9 + 1)):
                grp = picked[start:start + 9]
                if len(grp) < 9:
                    break
                g = sorted(grp, key=lambda c: c[5])
                rows = [g[0:3], g[3:6], g[6:9]]
                rows = [sorted(r, key=lambda c: c[4]) for r in rows]

                ys = [np.mean([c[5] for c in r]) for r in rows]
                row_gap_var = float(np.var(np.diff(ys)))

                wh = []
                for r in rows:
                    for c in r:
                        wh.extend([c[2], c[3]])
                wh = np.array(wh, dtype=np.float32)
                size_var = float(np.var(wh))

                hx = []
                for r in rows:
                    r_sorted = sorted(r, key=lambda c: c[4])
                    hx.append((r_sorted[1][4] - r_sorted[0][4]) + (r_sorted[2][4] - r_sorted[1][4]))
                hx = np.array(hx, dtype=np.float32)
                hgap_var = float(np.var(hx))

                score = row_gap_var + 0.002 * size_var + 0.002 * hgap_var
                if score < best_score:
                    best_score, best = score, rows

            if best is None:
                return None
            out: List[Tuple[int, int, int, int]] = []
            for r in best:
                for c in r:
                    out.append((int(c[0]), int(c[1]), int(c[2]), int(c[3])))
            return out

        grid = _pick_grid(_collect_candidates(relax=False))
        if grid is not None:
            return grid
        grid = _pick_grid(_collect_candidates(relax=True))
        return grid if grid is not None else []

    def _get_dominant_color(self, roi: np.ndarray) -> Tuple[int, int, int]:
        """Dominant BGR via k-means (k=1) on the ROI pixels."""
        if roi is None or roi.size == 0:
            return (0, 0, 0)
        pixels = np.float32(roi.reshape(-1, 3))
        _, _, palette = cv2.kmeans(
            pixels, 1, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1),
            10, cv2.KMEANS_RANDOM_CENTERS
        )
        c = palette[0]
        return (int(c[0]), int(c[1]), int(c[2]))

    def _get_closest_color(self, bgr: Tuple[int, int, int]) -> Dict[str, Any]:
        """Match a BGR to nearest palette color by CIEDE2000 in Lab space."""
        lab = _bgr_to_lab_list((int(bgr[0]), int(bgr[1]), int(bgr[2])))
        best = {"color_name": None, "distance": float("inf")}
        for name, ref_bgr in self.cube_color_palette.items():
            ref_lab = _bgr_to_lab_list((int(ref_bgr[0]), int(ref_bgr[1]), int(ref_bgr[2])))
            dist = _ciede2000(lab, ref_lab)
            if dist < best["distance"]:
                best["color_name"] = name
                best["distance"] = dist
        return best


    def set_color_palette(self, new_palette: Dict[str, Tuple[int, int, int]]) -> None:
        """Override the current palette manually (BGR per color)."""
        for name, bgr in new_palette.items():
            self.cube_color_palette[name] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
