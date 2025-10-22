import streamlit as st
from core import cube, beginner_solver
from services.vision.video_processor import VideoProcessor
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from visualization.plotly_cube import create_cube_visualization
import time

def view_web_app():
    st.title("Rubik's Cube Solver")

    # WebRTC configuration required
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    col1, col2= st.columns(2)
    
    # Webcam in first column
    with col1:
        st.markdown("## Webcam Feed")
        ctx = webrtc_streamer(
            key="rubiks-cube-detector",
            video_processor_factory=VideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    
    # Processed image in second column
    with col2:
        st.markdown("## Processed Image")
        processed_placeholder = st.empty()
    
    st.markdown("## Rubik's Cube Visualization")
    fig = create_cube_visualization()
    st.plotly_chart(fig, config={'displayModeBar': False})
    
    if ctx.video_processor:
        if ctx.state.playing:
            # Display processed frame if available
            if ctx.video_processor.processed_frame is not None:
                processed_placeholder.image(ctx.video_processor.processed_frame, width='stretch')
            
            # Auto refresh every 0.1 seconds
            time.sleep(0.1)
            st.rerun()