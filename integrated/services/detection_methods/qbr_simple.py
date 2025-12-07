from __future__ import annotations
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import cv2

from .base import DetectionMethod


def _bgr_to_lab_list(bgr: Tuple[int, int, int]) -> List[float]:
    """Convert a BGR color to CIE L*a*b* as [L, a, b] without cv2.cvtColor."""
    b, g, r = bgr
    # Convert to normalized linear RGB
    R, G, B = (r / 255.0), (g / 255.0), (b / 255.0)

    def _to_linear(c: float) -> float:
        # sRGB gamma correction
        return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92

    R, G, B = _to_linear(R), _to_linear(G), _to_linear(B)

    # Linear RGB -> XYZ using D65 illuminant
    X = R * 0.4124 + G * 0.3576 + B * 0.1805
    Y = R * 0.2126 + G * 0.7152 + B * 0.0722
    Z = R * 0.0193 + G * 0.1192 + B * 0.9505

    # Normalize by D65 reference white
    X /= 0.95047
    Y /= 1.00000
    Z /= 1.08883

    def _f(t: float) -> float:
        # Piecewise conversion for LAB
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

    C1 = math.sqrt(a1 * a1 + b1 * b1)
    C2 = math.sqrt(a2 * a2 + b2 * b2)
    C_ave = (C1 + C2) / 2.0

    # Compensation factor for chroma
    G = 0.5 * (1.0 - math.sqrt((C_ave**7) / (C_ave**7 + 25.0**7)))
    a1p, a2p = (1.0 + G) * a1, (1.0 + G) * a2
    C1p = math.sqrt(a1p * a1p + b1 * b1)
    C2p = math.sqrt(a2p * a2p + b2 * b2)

    def _hp(ap: float, bp: float) -> float:
        # Hue angle in radians
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
        if dh > math.pi:
            dh -= 2.0 * math.pi
        elif dh < -math.pi:
            dh += 2.0 * math.pi
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
            # "red":    (50, 50, 200),
            # "orange": (0, 180, 255),
            # "blue":   (180, 80, 30),
            # "green":  (40, 180, 40),
            # "white":  (225, 225, 225),
            # "yellow": (40, 210, 210),
        }
        self.cube_color_palette: Dict[str, Tuple[int, int, int]] = dict(self._prominent_palette)

        # Map color names to cube notation letters. Adjust for your face orientation.
        self._color_to_notation: Dict[str, str] = {
            "white":  "U",
            "red":    "F",
            "green":  "L",
            "yellow": "D",
            "orange": "B",
            "blue":   "R",
        }

        # Temporal smoothing buffers for each sticker index
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
        """Enter calibration mode (call capture_calibration_frame per face)."""
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

        # Preprocess using fixed Canny thresholds and morphological closing
        processed = self._preprocess_frame(frame_bgr)
        contours = self._find_face_contours(processed)
        if len(contours) == 9:
            # Use the center sticker (index 4) for calibration sample
            x, y, w, h = contours[4]
            # Crop inside the sticker to avoid borders/glare
            roi = frame_bgr[
                y + max(2, h // 10): y + h - max(2, h // 10),
                x + max(2, w // 6):  x + w - max(2, w // 6)
            ]
            avg_bgr = self._get_dominant_color(roi)
            color_name = self._colors_to_calibrate[self._calibrate_index]
            self._calibrated_colors[color_name] = avg_bgr
            self._calibrate_index += 1

            if self._calibrate_index == len(self._colors_to_calibrate):
                # Apply the calibrated palette
                for name, bgr in self._calibrated_colors.items():
                    self.cube_color_palette[name] = tuple(int(c) for c in bgr)
                self._calibrating = False
                return True
        return False

    def process(
        self, frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Tuple[float, float, float]]]:
        """
        Overlay a 3×3 grid on the camera feed and classify the color in each grid cell.

        This implementation draws a 3×3 guideline in the center of the frame. Users are
        expected to position a Rubik's cube face within the grid. For each of the 9
        grid cells, the average color is computed (using k-means) from the central
        portion of the cell. The nearest cube color is determined via Lab space
        distance. Results are smoothed across recent frames to reduce flicker. The
        detected letters (U,R,F,D,L,B) are drawn on the output frame at the center
        of each cell. The returned cube face labels array will contain the detected
        face in index 2 (Front) and '?' elsewhere. Rotation is unused here.

        Returns:
            out (np.ndarray): Frame with grid and letter annotations.
            cube_face_labels (np.ndarray|None): 6×3×3 array of cube notations, or None if not detected.
            rotation (tuple|None): Placeholder rotation (unused).
        """
        # Copy original frame for drawing
        out = frame_bgr.copy()
        height, width = frame_bgr.shape[:2]
        # Determine size of grid: use 60% of the smaller dimension
        side_len = int(min(height, width) * 0.6)
        # Top-left of grid so that it's centered
        gx = (width - side_len) // 2
        gy = (height - side_len) // 2
        cell_w = side_len // 3
        cell_h = side_len // 3

        # Draw the grid lines
        # Outer rectangle
        cv2.rectangle(out, (gx, gy), (gx + side_len, gy + side_len), (0, 255, 0), 2)
        # Vertical lines
        for i in range(1, 3):
            x_pos = gx + i * cell_w
            cv2.line(out, (x_pos, gy), (x_pos, gy + side_len), (0, 255, 0), 2)
        # Horizontal lines
        for i in range(1, 3):
            y_pos = gy + i * cell_h
            cv2.line(out, (gx, y_pos), (gx + side_len, y_pos), (0, 255, 0), 2)

        cube_face_labels: Optional[np.ndarray] = None
        rotation: Optional[Tuple[float, float, float]] = (0.0, 0.0, 0.0)

        # If not calibrating, perform color sampling and classification
        if not self._calibrating:
            detected_names: List[str] = ["" for _ in range(9)]
            for idx in range(9):
                row = idx // 3
                col = idx % 3
                # Cell top-left
                cx = gx + col * cell_w
                cy = gy + row * cell_h
                # Sample central region (50% of cell size) to avoid grid lines
                sx = int(cx + cell_w * 0.25)
                sy = int(cy + cell_h * 0.25)
                ex = int(cx + cell_w * 0.75)
                ey = int(cy + cell_h * 0.75)
                # Ensure ROI within frame
                sx = max(0, min(width - 1, sx))
                ex = max(0, min(width, ex))
                sy = max(0, min(height - 1, sy))
                ey = max(0, min(height, ey))
                roi = frame_bgr[sy:ey, sx:ex]
                avg_bgr = self._get_dominant_color(roi)
                match = self._get_closest_color(avg_bgr)
                color_name = match.get("color_name") or "?"
                # Temporal smoothing
                buf = self._avg_color_buffers.get(idx, [])
                buf.append(self.cube_color_palette.get(color_name, (0, 0, 0)))
                if len(buf) > 8:
                    buf = buf[-8:]
                self._avg_color_buffers[idx] = buf
                # Determine stable color once buffer is full and not yet stable
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
                # Draw the predicted letter on the output frame
                letter = self._color_to_notation.get(color_name, "?")
                # Compute text parameters based on cell size
                text_scale = max(0.5, min(cell_w, cell_h) / 55.0)
                thick_body = max(2, int(round(text_scale * 2.0)))
                thick_outline = thick_body + 2
                (tw, th), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thick_body)
                ccx = cx + cell_w // 2
                ccy = cy + cell_h // 2
                org = (int(ccx - tw / 2), int(ccy + th / 2))
                cv2.putText(out, letter, org, cv2.FONT_HERSHEY_SIMPLEX,
                            text_scale, (0, 0, 0), thick_outline, cv2.LINE_AA)
                cv2.putText(out, letter, org, cv2.FONT_HERSHEY_SIMPLEX,
                            text_scale, (255, 255, 255), thick_body, cv2.LINE_AA)
            # Build cube face labels: place letters into front face (index 2)
            face_letters = [
                [self._color_to_notation.get(n, "?") for n in detected_names[i:i + 3]]
                for i in range(0, 9, 3)
            ]
            cube_face_labels = np.full((6, 3, 3), "?", dtype="<U2")
            cube_face_labels[2] = np.array(face_letters, dtype="<U2")
        return out, cube_face_labels, rotation

    # ------------------------------- Helpers ----------------------------------

    def _preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.blur(gray, (3, 3))
        edges = cv2.Canny(blurred, 20, 50, apertureSize=3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        processed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        return processed

    def _find_face_contours(self, binary_img: np.ndarray) -> List[Tuple[int, int, int, int]]:

        contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        final_contours: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if w < 20 or w > 200:
                    continue
                area = cv2.contourArea(cnt)
                if h == 0:
                    continue
                aspect = w / float(h)
                if aspect < 0.7 or aspect > 1.3:
                    continue
                if area / (w * h + 1e-6) < 0.3:
                    continue
                final_contours.append((x, y, w, h))

        # Need at least 9 candidates
        if len(final_contours) < 9:
            return []

        # Determine neighbors by center proximity. For each candidate,
        # compute centers and predefine neighbor offset positions relative to
        # candidate size. Pick the first candidate that has 9 neighbors.
        neighbors_map: Dict[int, List[int]] = {}
        for i, (x, y, w, h) in enumerate(final_contours):
            neighbors_map[i] = []
            cx, cy = x + w / 2.0, y + h / 2.0
            r = 1.5
            neighbor_centers = [
                (cx - w * r, cy - h * r), (cx, cy - h * r), (cx + w * r, cy - h * r),
                (cx - w * r, cy),        (cx, cy),        (cx + w * r, cy),
                (cx - w * r, cy + h * r), (cx, cy + h * r), (cx + w * r, cy + h * r),
            ]
            for j, (x2, y2, w2, h2) in enumerate(final_contours):
                c2x, c2y = x2 + w2 / 2.0, y2 + h2 / 2.0
                for (nx, ny) in neighbor_centers:
                    # If center of j is close to neighbor position of i
                    if abs(c2x - nx) < w2 / 2.0 and abs(c2y - ny) < h2 / 2.0:
                        neighbors_map[i].append(j)
                        break

        center_index: Optional[int] = None
        for i, neighbor_list in neighbors_map.items():
            if len(neighbor_list) >= 9:
                center_index = i
                break

        if center_index is None:
            return []

        # Extract 9 contours and sort into 3 rows and 3 columns
        indices = neighbors_map[center_index][:9]
        face_contours = [final_contours[j] for j in indices]
        if len(face_contours) != 9:
            return []
        # Sort by y, then by x
        face_contours.sort(key=lambda rect: rect[1])
        top = sorted(face_contours[0:3], key=lambda rect: rect[0])
        middle = sorted(face_contours[3:6], key=lambda rect: rect[0])
        bottom = sorted(face_contours[6:9], key=lambda rect: rect[0])
        return top + middle + bottom

    def _get_dominant_color(self, roi: np.ndarray) -> Tuple[int, int, int]:
        """Compute the dominant BGR color of the given region using k-means."""
        if roi is None or roi.size == 0:
            return (0, 0, 0)
        pixels = np.float32(roi.reshape(-1, 3))
        if pixels.shape[0] < 8:
            # Fallback to median if too few pixels
            m = np.median(pixels, axis=0)
            return (int(m[0]), int(m[1]), int(m[2]))
        _, _, palette = cv2.kmeans(
            pixels, 1, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
            5, cv2.KMEANS_RANDOM_CENTERS
        )
        dominant = palette[0]
        return (int(dominant[0]), int(dominant[1]), int(dominant[2]))
    

    def _get_closest_color(self, bgr: Tuple[int, int, int]) -> Dict[str, Any]:
        lab = _bgr_to_lab_list((int(bgr[0]), int(bgr[1]), int(bgr[2])))
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        hue = hsv[0]

        closest_match = {"color_name": None, "distance": float("inf")}
        for name, ref_bgr in self.cube_color_palette.items():
            ref_lab = _bgr_to_lab_list((int(ref_bgr[0]), int(ref_bgr[1]), int(ref_bgr[2])))
            dist = _ciede2000(lab, ref_lab)

        # Hue disambiguation between orange/red
            if name in ["red", "orange"]:
                ref_hue = cv2.cvtColor(np.uint8([[ref_bgr]]), cv2.COLOR_BGR2HSV)[0][0][0]
                hue_diff = abs(int(hue) - int(ref_hue))
                if hue_diff > 90:  # e.g., wrap-around adjustment
                    hue_diff = 180 - hue_diff
                dist += 0.5 * (hue_diff / 180.0)

            if dist < closest_match["distance"]:
                closest_match["color_name"] = name
                closest_match["distance"] = dist
        return closest_match

    def _get_closest_color(self, bgr: Tuple[int, int, int]) -> Dict[str, Any]:
        """Find the nearest cube color (by LAB distance) to the given BGR color."""
        lab = _bgr_to_lab_list((int(bgr[0]), int(bgr[1]), int(bgr[2])))
        closest_match = {"color_name": None, "distance": float("inf")}
        for name, ref_bgr in self.cube_color_palette.items():
            ref_lab = _bgr_to_lab_list((int(ref_bgr[0]), int(ref_bgr[1]), int(ref_bgr[2])))
            dist = _ciede2000(lab, ref_lab)
            if dist < closest_match["distance"]:
                closest_match["color_name"] = name
                closest_match["distance"] = dist
        return closest_match
