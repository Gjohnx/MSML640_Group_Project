#camera reading
'''
Functionality to capture image input (e.g., from a webcam). 
Image processing/CV logic to detect sticker colors and map them to a CubeState.
'''
import streamlit as st
from streamlit_webrtc import VideoProcessorBase
import numpy as np
import av

class VideoProcessor(VideoProcessorBase):
    
    def __init__(self):
        self.frame_count = 0
        
    def recv(self, frame):
        
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Image processing will be here
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
