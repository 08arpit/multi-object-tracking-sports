import streamlit as st
import os
import tempfile
import time
import urllib.request
from PIL import Image
import sys

# Ensure local src imports work seamlessly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.detector import PlayerDetector
from src.tracker import PlayerTracker
from src.video_processor import VideoProcessor

# Define dummy config mimicking argparse
class ArgsConfig:
    pass

st.set_page_config(page_title="Sports AI Tracker", page_icon="⚽", layout="wide")

# Cached resource loading so YOLO model isn't reloaded every run
@st.cache_resource
def load_models(model_name="yolo11n.pt", tracker_name="bytetrack.yaml"):
    detector = PlayerDetector(model_weight=model_name)
    tracker = PlayerTracker(detector=detector, tracker_type=tracker_name)
    return detector, tracker

# Setup Header
st.title("⚽ Multi-Object Detection & Persistent Tracking")
st.markdown("Upload a sports clip to extract player trajectories, heatmaps, and tactical top-views natively using **YOLOv11** & **ByteTrack**.")

with st.expander("📖 How it Works (YOLO + ByteTrack)"):
    st.markdown("""
    When analyzing sports footage, standard detection models drop bounding boxes during **high-motion blurs** or when athletes **cross paths** (occlusion). 
    - **YOLOv11** handles the primary frame-by-frame deep-learning inference to identify all people and the ball.
    - **ByteTrack** functions as the persistence layer. Unlike older trackers that discard low-confidence detections (like a blurred runner), ByteTrack safely matches these artifacts to known trajectories utilizing Kalman filters. This guarantees an athlete crossing behind another won't lose their tracking ID!
    """)

# Sidebar config
with st.sidebar:
    st.header("⚙️ Pipeline Configuration")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05, help="Minimum confidence for bounding box detection.")
    max_frames = st.slider("Max Frames to Process", 50, 500, 200, 50, help="For the Live Demo, limiting frames keeps processing under 60 seconds!")
    
    st.subheader("✨ Visual Enhancements")
    trajectories = st.checkbox("Show Trajectories", value=True)
    heatmap = st.checkbox("Generate Heatmap", value=True)
    topview = st.checkbox("Generate Top-View", value=True)
    metrics = st.checkbox("Calculate Metrics", value=True)
    team_cluster = st.checkbox("Cluster Teams (Colors)", value=True)
    
    st.markdown("---")
    st.markdown("[View Full Code on GitHub](https://github.com/08arpit/multi-object-tracking-sports)")

# --- Input Handling ---
st.header("1. Provide Source Video 📹")
col1, col2 = st.columns([1, 1])

uploaded_file = None
use_sample = False

with col1:
    uploaded_file = st.file_uploader("Upload a raw sports video (MP4/MOV)", type=["mp4", "mov", "avi"])

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔽 Or Use Sample Video", use_container_width=True):
        use_sample = True

def get_sample_video():
    # If the user's soccer.mp4 is available locally or pushed to hugging face, use it
    if os.path.exists("input_video/soccer.mp4"):
        return "input_video/soccer.mp4"
    
    # Otherwise, download a brief public domain MP4 directly to bypass any missing file errors
    sample_path = "input_video/sample_clip.mp4"
    os.makedirs("input_video", exist_ok=True)
    if not os.path.exists(sample_path):
        with st.spinner("Downloading sample video..."):
            urllib.request.urlretrieve("https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4", sample_path)
    return sample_path

input_video_path = None

if use_sample:
    input_video_path = get_sample_video()
    st.info(f"Using Sample Video: {input_video_path}")
elif uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    input_video_path = tfile.name
    st.success("Custom video uploaded successfully!")

if input_video_path:
    st.video(input_video_path)
    
    st.markdown("---")
    st.header("2. Processing Execution 🚀")
    
    if st.button("🔥 Run Tracking Pipeline", type="primary", use_container_width=True):
        os.makedirs("output/sample_screenshots", exist_ok=True)
        output_path = "output/annotated_output.mp4"
        
        args = ArgsConfig()
        args.input_video = input_video_path
        args.output_video = output_path
        args.model = "yolo11n.pt"
        args.tracker = "bytetrack.yaml"
        args.conf = conf_threshold
        args.screenshot_interval = max(30, max_frames // 4)
        args.max_frames = int(max_frames)
        args.trajectories = trajectories
        args.heatmap = heatmap
        args.topview = topview
        args.metrics = metrics
        args.team_cluster = team_cluster
        
        # Load cached models to ensure repeat runs are exceptionally fast
        with st.spinner("Loading AI Models into memory..."):
            detector, tracker = load_models()
            
        st.info(f"Initializing Video Processor for {max_frames} frames...")
        
        # Streamlit Progress Wrappers
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total):
            progress_pct = int((current / total) * 100)
            progress_bar.progress(progress_pct)
            status_text.text(f"Processing Frame {current} of {total} ({progress_pct}%)")
            
        start_time = time.time()
        
        try:
            # Reuses standard VideoProcessor natively, utilizing newly-added callback logic seamlessly
            processor = VideoProcessor(tracker=tracker, input_path=input_video_path, output_path=output_path, args=args)
            processor.process_video(conf=args.conf, screenshot_interval=args.screenshot_interval, progress_callback=update_progress)
            
            duration = time.time() - start_time
            progress_bar.progress(100)
            status_text.text("Processing Complete!")
            st.success(f"✅ Pipeline executed safely in {duration:.2f} seconds.")
            
            # --- Results ---
            st.markdown("---")
            st.header("3. Analytical Results 📊")
            
            st.subheader("Annotated Video")
            try:
                st.video(output_path)
            except Exception:
                st.warning("Browser couldn't play raw OpenCV mp4v stream natively.")
            
            with open(output_path, 'rb') as v_file:
                st.download_button(label="📥 Download Annotated Video", data=v_file, file_name="annotated_output.mp4", mime="video/mp4")
                
            col1, col2 = st.columns(2)
            
            if heatmap and os.path.exists("output/movement_heatmap.jpg"):
                with col1:
                    st.markdown("#### Spacial Movement Heatmap")
                    img = Image.open("output/movement_heatmap.jpg")
                    st.image(img, use_container_width=True)
                    with open("output/movement_heatmap.jpg", "rb") as f:
                        st.download_button("📥 Download Heatmap", f, file_name="heatmap.jpg")
                        
            if topview and os.path.exists("output/topview.jpg"):
                with col2:
                    st.markdown("#### Tactical Top-View")
                    img = Image.open("output/topview.jpg")
                    st.image(img, use_container_width=True)
                    with open("output/topview.jpg", "rb") as f:
                        st.download_button("📥 Download Topview", f, file_name="topview.jpg")
                        
            if metrics and os.path.exists("output/player_count_plot.png"):
                st.markdown("#### Active Target Logistics")
                img = Image.open("output/player_count_plot.png")
                st.image(img, use_container_width=True)
                with open("output/player_count_plot.png", "rb") as f:
                    st.download_button("📥 Download Plot", f, file_name="plot.png")
                    
        except Exception as e:
            st.error(f"Critical execution error: {str(e)}")
