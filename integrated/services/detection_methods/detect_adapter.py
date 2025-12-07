# integrated/services/detection_methods/detect_adapter.py
from __future__ import annotations
from typing import Tuple, Optional, Dict
import numpy as np

from .base import DetectionMethod
from ..vision.color_grid_detector import detect_color_grid


COLOR_NAME_TO_CHAR = {
    "white": "U",
    "yellow": "D",
    "red":    "F",
    "orange": "B",
    "blue":   "R",
    "green":  "L",
}


class DetectAdapterDetectionMethod(DetectionMethod):
    """Adapter that uses detect_color_grid() for Rubik face color recognition."""

    name = "HSV (Grid Rectify)"

    def __init__(self) -> None:
        self._last_processed = None
        self._last_cube_colors = None
        self._last_rotation = None

    def reset(self) -> None:
        """Clear cached results."""
        self._last_processed = None
        self._last_cube_colors = None
        self._last_rotation = None

    def process(
        self, frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Tuple[float, float, float]]]:
        """Process a single frame using detect_color_grid()."""
        res: Dict = detect_color_grid(frame_bgr, debug=True)

        # Safe selection: explicitly check for None instead of using `or`
        warp = res.get("warp_bgr")
        overlay = res.get("overlay_bgr")

        if warp is not None:
            processed = warp
        elif overlay is not None:
            processed = overlay
        else:
            processed = frame_bgr

        # Build cube color grid
        grid = res.get("grid")
        cube_colors = None
        if grid is not None:
            mapped = np.full_like(grid, "?", dtype="<U2")
            for i in range(3):
                for j in range(3):
                    mapped[i, j] = COLOR_NAME_TO_CHAR.get(str(grid[i, j]).lower(), "?")
            cube_colors = np.full((6, 3, 3), "?", dtype="<U2")
            cube_colors[2] = mapped

        rotation = (15.0, 15.0, 0.0)

        self._last_processed = processed
        self._last_cube_colors = cube_colors
        self._last_rotation = rotation

        return processed, cube_colors, rotation
