import cv2
import numpy as np
from .video_filter import VideoFilter

class RemoveGlareFilter(VideoFilter):
    
    def __init__(self, glare_threshold=220, inpaint_radius=10):
        self.glare_threshold = glare_threshold
        self.inpaint_radius = inpaint_radius
    
    def detect_glare_mask(self, img):
        # Convert to grayscale and apply the threshold to find bright regions
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        _, glare_mask = cv2.threshold(gray, self.glare_threshold, 255, cv2.THRESH_BINARY)
        
        # This operation smooths the mask which could be rough or have small spots
        kernel = np.ones((5, 5), np.uint8)
        glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_CLOSE, kernel)
        glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_OPEN, kernel)
        
        return glare_mask
    
    def apply(self, img: np.ndarray) -> np.ndarray:
        # Detect glare regions
        glare_mask = self.detect_glare_mask(img)
        
        # Apply inpainting to fill the glare pixels with the surrounding pixels
        inpainted = cv2.inpaint(img, glare_mask, self.inpaint_radius, cv2.INPAINT_TELEA)
        
        return inpainted

