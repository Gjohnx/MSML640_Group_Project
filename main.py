#Streamlit UI
'''
Application entry point for Rubik's Cube Solver with webcam input
'''
import streamlit as st
from core import cube, beginner_solver
from services.video_processor import VideoProcessor
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

def start_webcam_capture():
    st.subheader("Webcam Cube Scanner")
    
    # WebRTC configuration required
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Create webcam streamer
    ctx = webrtc_streamer(
        key="rubiks-cube-detector",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    st.markdown("""
    ### IN DEVELOPMENT""")
    
    if ctx.video_processor:
        st.write(f"Frames processed: {ctx.video_processor.frame_count}")

def main():
    st.title("Rubik's Cube Solver")
    
    start_webcam_capture()

if __name__ == "__main__":
    main()