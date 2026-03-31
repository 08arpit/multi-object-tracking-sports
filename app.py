import streamlit as st
import cv2
import tempfile
import time
import os
import sys

# Ensure src modules can be imported smoothly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.detector import PlayerDetector
    from src.tracker import PlayerTracker
    from src.video_processor import VideoProcessor
except ImportError as e:
    st.error(f"Missing core modules or unresolved imports! Error: {e}")
    st.stop()

# Dynamic Args to prevent modifying CLI structures in core logic
class PipelineArgs:
    def __init__(self):
        self.input_video = ""
        self.output_video = ""
        self.model = "yolov8n.pt"  # Safely fallback to highly reliable v8n
        self.tracker = "bytetrack.yaml"
        self.conf = 0.3
        self.screenshot_interval = 150
        self.max_frames = None
        self.trajectories = False
        self.heatmap = False
        self.topview = False
        self.metrics = False
        self.team_cluster = False

st.set_page_config(page_title="Multi-Object Tracking - Sports", layout="wide")
st.title("Multi-Object Tracking - Sports")
st.markdown("Upload a sports clip to run YOLOv8 and ByteTrack multiple object tracking.")

# --- SIDEBAR ---
st.sidebar.header("Tracking Configuration")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
frame_skip = st.sidebar.slider("Frame Skip (Process 1 of N frames)", 1, 5, 2)
max_frames = st.sidebar.number_input("Max Output Frames to Process", min_value=10, max_value=2000, value=200, step=50)

# --- INPUT SECTION ---
st.header("1. Input Section")
uploaded_video = st.file_uploader("Upload a video file to analyze", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # --- SIZE CHECK (Rule 8) ---
    MAX_FILE_SIZE_MB = 200
    if uploaded_video.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"Upload failed: File size exceeds the {MAX_FILE_SIZE_MB}MB safety limit.")
        st.stop()

    st.video(uploaded_video)
    
    if st.button("Run Tracking", type="primary"):
        st.header("2. Output Section")
        
        # --- DYNAMIC NAMING (Rule 5) ---
        timestamp_id = int(time.time())
        output_video_path = f"output/output_{timestamp_id}.mp4"
        os.makedirs("output", exist_ok=True)
        
        # --- INIT PROGRESS BAR (Rule 6) ---
        progress_bar = st.progress(0.0, text="Initializing Model Weights & Pipelines...")
        
        try:
            # Safely trap to temp memory
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_input.write(uploaded_video.read())
            temp_input.close()
            input_path = temp_input.name
            
            # Init Pipeline Context
            args = PipelineArgs()
            args.input_video = input_path
            args.output_video = output_video_path
            args.conf = conf_threshold
            args.max_frames = int(max_frames)
            
            start_time = time.time()
            
            # Progress Callback Hook
            def st_progress_callback(current, total):
                progress = min(current / total, 1.0)
                progress_bar.progress(progress, text=f"Processing frame {current} / {total}")
            
            detector = PlayerDetector(model_weight=args.model)
            tracker = PlayerTracker(detector=detector, tracker_type=args.tracker)
            
            processor = VideoProcessor(
                tracker=tracker,
                input_path=args.input_video,
                output_path=args.output_video,
                args=args
            )
            
            # Execute logic injection safely
            processor.process_video(
                conf=args.conf, 
                screenshot_interval=args.screenshot_interval, 
                frame_skip=frame_skip,
                progress_callback=st_progress_callback
            )
            
            # Clean up instantly for memory limits
            if os.path.exists(input_path):
                os.unlink(input_path)
            
            progress_bar.progress(1.0, text="Finalizing Output Stream!")
            duration = time.time() - start_time
            st.success(f"Tracking and encoding completed successfully in {duration:.2f} seconds.")
            
            # Verify and Inject Codec Media Player Stream
            if os.path.exists(output_video_path):
                # We attempt to play whatever the browser can stomach natively!
                try:
                    st.video(output_video_path)
                except Exception as playback_err:
                    st.warning(f"Browser playback limited by native codec. You can download the pristine mp4 below. (Sys details: {playback_err})")
                
                with open(output_video_path, "rb") as file:
                    st.download_button("Download Annotated Results MP4", data=file, file_name=output_video_path, mime="video/mp4")
            else:
                st.error("Encoding failed: the system did not successfully generate an output.mp4 container.")
                
        except Exception as e:
            st.error(f"Failed to process video gracefully: {str(e)}")
