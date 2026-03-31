import streamlit as st
import os
import tempfile
import time
import sys

# Ensure local src imports work seamlessly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.detector import PlayerDetector
from src.tracker import PlayerTracker
from src.video_processor import VideoProcessor

# Dummy Args class to mimic the standard python argparse from main.py
class ArgsConfig:
    pass

# --- UI Configuration ---
st.set_page_config(page_title="Sports AI Tracker", page_icon="⚽", layout="wide")

st.title("⚽ Multi-Object Detection & Tracking")
st.markdown("Upload a sports clip to extract player trajectories, heatmaps, and tactical top-views natively using **YOLOv11** & **ByteTrack**.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05, help="Minimum confidence for bounding box detection.")
    max_frames = st.number_input("Max Frames to Process", min_value=10, max_value=2000, value=150, step=50, help="For the Live Demo, limiting frames keeps processing fast!")
    
    st.subheader("Visual Enhancements")
    trajectories = st.checkbox("Show Trajectories", value=True)
    heatmap = st.checkbox("Generate Heatmap", value=True)
    topview = st.checkbox("Generate Top-View", value=True)
    metrics = st.checkbox("Calculate Metrics", value=True)
    team_cluster = st.checkbox("Cluster Teams (Colors)", value=True)
    
    st.markdown("---")
    st.markdown("### Project Info")
    st.markdown("[View Full Code on GitHub](https://github.com/08arpit/multi-object-tracking-sports)")
    st.caption("Engineered for Deep Learning & Computer Vision Analysis.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload a raw sports video (MP4/AVI/MOV)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.markdown("### Original Input Video")
    st.video(uploaded_file)
    
    if st.button("🚀 Run Tracking Pipeline", type="primary", use_container_width=True):
        with st.spinner(f"Engine Running! Processing up to {max_frames} frames... This may take a moment."):
            
            # Safely create output directories if missing
            os.makedirs("output/sample_screenshots", exist_ok=True)
            
            # 1. Save uploaded file to disk temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            input_path = tfile.name
            output_path = "output/annotated_output.mp4"
            
            # 2. Setup arguments reflecting the Sidebar UI
            args = ArgsConfig()
            args.input_video = input_path
            args.output_video = output_path
            args.model = "yolo11n.pt"
            args.tracker = "bytetrack.yaml"
            args.conf = conf_threshold
            args.screenshot_interval = 300
            args.max_frames = int(max_frames)
            args.trajectories = trajectories
            args.heatmap = heatmap
            args.topview = topview
            args.metrics = metrics
            args.team_cluster = team_cluster
            
            start_time = time.time()
            try:
                # 3. Instantiate core components cleanly
                detector = PlayerDetector(model_weight=args.model)
                tracker = PlayerTracker(detector=detector, tracker_type=args.tracker)
                
                processor = VideoProcessor(
                    tracker=tracker, 
                    input_path=input_path, 
                    output_path=output_path, 
                    args=args
                )
                
                # 4. Trigger processing loop
                processor.process_video(conf=args.conf, screenshot_interval=args.screenshot_interval)
                
                duration = time.time() - start_time
                st.success(f"✅ Processing completed successfully in {duration:.2f} seconds!")
                
                # --- DISPLAY RESULTS ---
                st.markdown("---")
                st.header("📊 Interactive Results Overview")
                
                st.markdown("### Annotated Output Video")
                if os.path.exists(output_path):
                    # Note: OpenCV generates mp4v containers which Web Browsers often fail to render. 
                    # If it plays as an audio-only or black box, download the actual file locally!
                    try:
                        st.video(output_path)
                    except Exception:
                        pass
                    
                    with open(output_path, 'rb') as v_file:
                        st.download_button(
                            label="📥 Download Annotated MP4", 
                            data=v_file, 
                            file_name="annotated_output.mp4", 
                            mime="video/mp4"
                        )
                
                st.markdown("---")
                # Grid view for Images
                col1, col2 = st.columns(2)
                
                if heatmap and os.path.exists("output/movement_heatmap.jpg"):
                    with col1:
                        st.markdown("#### Spatial Movement Heatmap")
                        st.image("output/movement_heatmap.jpg", use_container_width=True)
                        
                if topview and os.path.exists("output/topview.jpg"):
                    with col2:
                        st.markdown("#### Tactical Top-View Projection")
                        st.image("output/topview.jpg", use_container_width=True)
                        
                if metrics and os.path.exists("output/player_count_plot.png"):
                    st.markdown("#### Player Presence Analytics")
                    st.image("output/player_count_plot.png", use_container_width=True)
                    
            except Exception as e:
                st.error(f"Critial Error during pipeline execution: {str(e)}")
