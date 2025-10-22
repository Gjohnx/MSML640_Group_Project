import streamlit as st
from streamlit_webrtc import VideoProcessorBase
import numpy as np
import av
import cv2
from .filters import RemoveGlareFilter
from .filters import CannyEdgeFilter

GLARE_THRESHOLD = 220
INPAINT_RADIUS = 5
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
CANNY_APERTURE_SIZE = 3

class VideoProcessor(VideoProcessorBase):
    
    def __init__(self):
        self.original_frame = None
        self.processed_frame = None
        
        
        self.glare_filter = RemoveGlareFilter(
            glare_threshold=GLARE_THRESHOLD,
            inpaint_radius=INPAINT_RADIUS
        )

        self.canny_edge_filter = CannyEdgeFilter(
            low_threshold=CANNY_LOW_THRESHOLD,
            high_threshold=CANNY_HIGH_THRESHOLD,
            aperture_size=CANNY_APERTURE_SIZE,
            return_colored=False
        )
        
    def recv(self, frame):
        
        image = frame.to_ndarray(format="bgr24")
        
        self.original_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = image.copy()
        
        # I removed the glare filter because it was not working as expected
        # processed_image = self.glare_filter.apply(processed_image)
        processed_image = self.canny_edge_filter.apply(processed_image)
        
        self.processed_frame = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        
        return av.VideoFrame.from_ndarray(image, format="bgr24")
