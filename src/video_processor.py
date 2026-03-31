import cv2
import os
from tqdm import tqdm
from .utils import draw_annotations, save_screenshot
from .enhancements import PipelineEnhancements

class VideoProcessor:
    """
    Coordinates video I/O, runs tracking, visualizes annotations, and saves output.
    """
    def __init__(self, tracker, input_path, output_path, screenshot_dir="output/sample_screenshots", args=None):
        self.tracker = tracker
        self.input_path = input_path
        self.output_path = output_path
        self.screenshot_dir = screenshot_dir
        self.args = args
        
        # Ensure output directories exist
        os.makedirs(self.screenshot_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def process_video(self, conf=0.3, screenshot_interval=100, frame_skip=1, progress_callback=None):
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video at path: {self.input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Apply testing limits if overridden by user
        if getattr(self.args, 'max_frames', None):
            total_frames = min(total_frames, self.args.max_frames)

        # Initialize Enhancements module if any optional features are active
        enhancer = None
        if self.args and any([self.args.trajectories, self.args.heatmap, self.args.metrics, self.args.topview, self.args.team_cluster]):
            enhancer = PipelineEnhancements(width, height, self.args)

        # Attempt robust 'avc1' h264 browser codec! Fallback smoothly to standard mp4v.
        adjusted_fps = fps / frame_skip if frame_skip > 0 else fps
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(self.output_path, fourcc, adjusted_fps, (width, height))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, adjusted_fps, (width, height))

        frame_count = 0
        screenshot_count = 0

        print(f"Processing source video: {self.input_path}")
        print(f"Total frame count: {total_frames}")

        try:
            for i in tqdm(range(total_frames), desc="Tracking players"):
                if progress_callback:
                    progress_callback(i + 1, total_frames)
                    
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Frame skipping for performance improvement
                if frame_skip > 1 and frame_count % frame_skip != 0:
                    frame_count += 1
                    continue

                # 1. Detect and Track objects ensuring persistent IDs
                results = self.tracker.track(frame, conf=conf)

                # Safely trap extremely rare completely empty tracker outcome frames
                if not results or not hasattr(results[0], 'boxes'):
                    out.write(frame)
                    frame_count += 1
                    continue

                # 2. Add bounding boxes natively, fallback to plot if drawing missing!
                try:
                    annotated_frame = draw_annotations(frame, results)
                except Exception:
                    annotated_frame = results[0].plot()

                # Optional: Apply active enhancements post base-annotation
                if enhancer:
                    annotated_frame = enhancer.process_frame(frame, annotated_frame, results)

                # 3. Write annotated frame to final video
                out.write(annotated_frame)

                # 4. Generate high-quality screenshots periodically
                if frame_count % screenshot_interval == 0 and screenshot_count < 6:
                    screenshot_file = os.path.join(self.screenshot_dir, f"screenshot_{frame_count}.jpg")
                    save_screenshot(annotated_frame, screenshot_file)
                    screenshot_count += 1
                    
                frame_count += 1
        finally:
            cap.release()
            out.release()
        
        # Export all aggregated stats or additional optional media structures
        if enhancer:
            print("Aggregating outputs for optional features...")
            enhancer.generate_final_outputs(os.path.dirname(self.output_path))
            
        print(f"Pipeline complete. Fully annotated output video saved to: {self.output_path}")
