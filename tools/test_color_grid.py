# tools/test_color_grid.py
# -*- coding: utf-8 -*-
"""Test script for color grid detection.
Usage:
    conda run -n cube python tools/test_color_grid.py path/to/image.jpg
"""
import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root to sys.path
from services.vision.color_grid_detector import detect_color_grid

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/test_color_grid.py <image_path>")
        sys.exit(1)
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        sys.exit(2)

    out = detect_color_grid(
        frame_bgr=img,
        debug=True,                 # keep overlays
        expected_size=320,          # rectified size
        sample_ratio=0.36,
        min_area_ratio=0.05,        # slightly lower for photos
        approx_epsilon_ratio=0.02,

    )

    print("ok:", out["ok"], "msg:", out["msg"])
    if out["grid"] is not None:
        print("3x3 labels:\n", out["grid"])
        print("LAB distances:\n", out["grid_dists"])

    # Save visual outputs for inspection
    if out["overlay_bgr"] is not None:
        cv2.imwrite("out_overlay.png", out["overlay_bgr"])
    if out["warp_bgr"] is not None:
        cv2.imwrite("out_warp.png", out["warp_bgr"])

    print("Saved out_overlay.png and out_warp.png (if available).")

if __name__ == "__main__":
    main()
