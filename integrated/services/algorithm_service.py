"""Service layer for image processing algorithms."""
import cv2
import numpy as np
from typing import Dict, Callable


class AlgorithmService:
    """Service that provides image processing algorithm implementations."""
    
    @staticmethod
    def get_all_algorithms() -> Dict[str, Callable]:
        """Get all available algorithm implementations."""
        return {
            "Grayscale": AlgorithmService.grayscale,
            "Blur": AlgorithmService.blur,
            "Edge Detection": AlgorithmService.edge_detection,
            "Invert": AlgorithmService.invert,
            "Sepia": AlgorithmService.sepia,
            "Cartoon": AlgorithmService.cartoon,
        }
    
    @staticmethod
    def grayscale(frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def blur(frame: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to frame."""
        return cv2.GaussianBlur(frame, (15, 15), 0)
    
    @staticmethod
    def edge_detection(frame: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection to frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 100, 200)
    
    @staticmethod
    def invert(frame: np.ndarray) -> np.ndarray:
        """Invert the colors of the frame."""
        return cv2.bitwise_not(frame)
    
    @staticmethod
    def sepia(frame: np.ndarray) -> np.ndarray:
        """Apply sepia tone filter to frame."""
        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        return cv2.transform(frame, kernel)
    
    @staticmethod
    def cartoon(frame: np.ndarray) -> np.ndarray:
        """Apply cartoon effect to frame."""
        # Bilateral filter for edge-preserving smoothing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 9, 9
        )
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon

