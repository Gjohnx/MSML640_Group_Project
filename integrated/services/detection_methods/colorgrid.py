from typing import Tuple, Optional, Dict
import numpy as np
import cv2
from .base import DetectionMethod

class ColorGridDetectionMethod(DetectionMethod):
    """
    Static Grid Detection Method.
    
    CORRECTED COLOR MAPPING:
    - Physical Red    -> App Key 'F' (View renders F as Red)
    - Physical Orange -> App Key 'B' (View renders B as Orange)
    - Physical Green  -> App Key 'L' (View renders L as Green)
    - Physical Blue   -> App Key 'R' (View renders R as Blue)
    - Physical White  -> App Key 'U'
    - Physical Yellow -> App Key 'D'
    """

    def __init__(self):
        self.color_ranges = {}
        
        # --- PHYSICAL WHITE -> App 'U' ---
        self.color_ranges['U'] = [(np.array([0, 0, 200]), np.array([180, 80, 255]))]
        
        # --- PHYSICAL RED -> App 'F' ---
        # Ranges: 0-10, 170-180
        self.color_ranges['F'] = [
            (np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([170, 100, 100]), np.array([180, 255, 255]))
        ]

        # --- PHYSICAL ORANGE -> App 'B' ---
        # Range: 11-25
        self.color_ranges['B'] = [(np.array([11, 100, 100]), np.array([25, 255, 255]))]
        
        # --- PHYSICAL YELLOW -> App 'D' ---
        # Range: 26-35
        self.color_ranges['D'] = [(np.array([26, 100, 100]), np.array([35, 255, 255]))]
        
        # --- PHYSICAL GREEN -> App 'L' ---
        # Range: 36-85
        self.color_ranges['L'] = [(np.array([36, 100, 100]), np.array([85, 255, 255]))]
        
        # --- PHYSICAL BLUE -> App 'R' ---
        # Range: 86-128
        self.color_ranges['R'] = [(np.array([86, 100, 100]), np.array([128, 255, 255]))]

        
        # UI Display Colors (BGR) for the dots on screen
        # These match what the user sees on the cube_view
        self.ui_colors = {
            'U': (255, 255, 255), # White
            'F': (0, 0, 255),     # Red (BGR)
            'B': (0, 165, 255),   # Orange (BGR)
            'D': (0, 255, 255),   # Yellow
            'L': (0, 255, 0),     # Green
            'R': (255, 0, 0),     # Blue
            '?': (128, 128, 128)
        }

        # Map Center Color to Face Index (0=U, 1=R, 2=F, 3=D, 4=L, 5=B)
        # We use the App Keys here.
        # If center is Red ('F'), it maps to Face 2.
        self.face_map = {
            'U': 0, # Up
            'R': 1, # Right (Blue)
            'F': 2, # Front (Red)
            'D': 3, # Down
            'L': 4, # Left (Green)
            'B': 5  # Back (Orange)
        }

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[float, float, float]]]:
        if frame is None or frame.size == 0:
            return frame, np.full((6, 3, 3), '?', dtype='<U1'), None

        # 1. Flip frame
        display_frame = cv2.flip(frame, 1)

        # 2. Convert to HSV
        # [FIX] Reverted to BGR2HSV because frame is BGR from OpenCV
        hsv = cv2.cvtColor(display_frame, cv2.COLOR_BGR2HSV)
        
        h, w, _ = display_frame.shape

        # 3. Grid Configuration
        box_size = 100
        grid_w = box_size * 3
        start_x = (w - grid_w) // 2
        start_y = (h - grid_w) // 2

        detected_face = np.full((3, 3), '?', dtype='<U1')

        # 4. Loop 3x3 Grid
        for i in range(3):
            for j in range(3):
                x = start_x + (j * box_size)
                y = start_y + (i * box_size)
                
                # Bounds check
                if x+60 < w and y+60 < h:
                    # Analyze center of sticker (20x20px sample)
                    roi = hsv[y+40:y+60, x+40:x+60]
                    color_code = self._get_color(roi)
                    detected_face[i, j] = color_code
                    
                    # Visual Feedback
                    cv2.rectangle(display_frame, (x, y), (x+box_size, y+box_size), (0, 255, 0), 2)
                    
                    dot_color = self.ui_colors.get(color_code, (128,128,128))
                    cv2.circle(display_frame, (x+50, y+50), 8, dot_color, -1)
                    
                    # Draw text (optional)
                    # cv2.putText(display_frame, color_code, (x+5, y+25), 
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        # 5. Update Cube State
        cube_state = np.full((6, 3, 3), '?', dtype='<U1')
        
        # Check Center Sticker to identify face
        center_color = detected_face[1, 1]
        
        if center_color in self.face_map:
            face_idx = self.face_map[center_color]
            cube_state[face_idx] = detected_face
            
            # Label
            label_map = {'U':'Up', 'D':'Down', 'F':'Front (Red)', 'B':'Back (Orange)', 'L':'Left (Green)', 'R':'Right (Blue)'}
            face_name = label_map.get(center_color, center_color)
            
            cv2.putText(display_frame, f"Updating: {face_name}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Align Center Sticker", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return display_frame, cube_state, None

    def _get_color(self, hsv_roi):
        """ Determines the dominant color in a region of interest """
        if hsv_roi.size == 0: return '?'
        
        mean_hsv = np.mean(hsv_roi, axis=(0,1))
        pixel = np.uint8([[mean_hsv]])

        for color_code, ranges in self.color_ranges.items():
            for (low, high) in ranges:
                if cv2.inRange(pixel, low, high).any():
                    return color_code
        
        return '?'
    def reset(self):
        """
        Resets the internal state of the detection method.
        """
        # Add logic here if you need to clear buffers, history, or counters.
        # If no logic is needed yet, use 'pass'.
        pass