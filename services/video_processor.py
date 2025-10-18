#camera reading
'''
Functionality to capture image input (e.g., from a webcam). 
Image processing/CV logic to detect sticker colors and map them to a CubeState.
'''
import streamlit as st
from streamlit_webrtc import VideoTransformerBase
import numpy as np

class VideoProcessor(VideoTransformerBase):
    
    def __init__(self):
        self.frame_count = 0
        
    def transform(self, frame):
        
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # DO SOMETHING HERE
        
        return img
