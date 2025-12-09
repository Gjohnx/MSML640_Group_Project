import cv2
import numpy as np
from .video_filter import VideoFilter

class CannyEdgeFilter(VideoFilter):
    
    def __init__(self, low_threshold=50, high_threshold=150, aperture_size=3, return_colored=False):
        """
        Initialize the Canny Edge Detection filter.
        
        Args:
            low_threshold: Lower threshold for the hysteresis procedure (default: 50)
            high_threshold: Upper threshold for the hysteresis procedure (default: 150)
            aperture_size: Aperture size for the Sobel operator (must be 3, 5, or 7; default: 3)
            return_colored: If True, converts the edge-detected image back to BGR for visualization (default: False)
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.aperture_size = aperture_size
        self.return_colored = return_colored
    
    def apply(self, img: np.ndarray) -> np.ndarray:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # OpenCV Canny edge detection
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold, 
                         apertureSize=self.aperture_size)
        
        # In some cases is better to transform the grayscale image to a BGR format
        if self.return_colored:
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        return edges

