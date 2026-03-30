import argparse
import os
import sys

# Ensure project structure is loaded correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.detector import PlayerDetector
from src.tracker import PlayerTracker
from src.video_processor import VideoProcessor

def main():
    parser = argparse.ArgumentParser(description="Multi-Object Detection and Tracking in Sports Video")
    parser.add_argument("--input_video", type=str, default="input_video/soccer.mp4", 
                        help="Path to the input original raw video")
    parser.add_argument("--output_video", type=str, default="output/annotated_output.mp4", 
                        help="Relative path to save the final annotated output video")
    parser.add_argument("--model", type=str, default="yolo11n.pt", 
                        help="YOLO model path/name (Defaults to YOLO11 Nano)")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", 
                        choices=["bytetrack.yaml", "botsort.yaml"], 
                        help="Config file for tracker (ByteTrack is the recommended default)")
    parser.add_argument("--conf", type=float, default=0.3, 
                        help="Minimum confidence threshold for players/athletes detection")
    parser.add_argument("--screenshot-interval", type=int, default=150, 
                        help="Frame interval for generating high quality samples")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit processing to a specific number of frames for faster testing/debugging")
    
    # Optional Enhancements
    parser.add_argument("--trajectories", action="store_true", help="Draw smooth trailing lines for recent movement")
    parser.add_argument("--heatmap", action="store_true", help="Generate 2D movement heatmap image")
    parser.add_argument("--topview", action="store_true", help="Generate top-view projection image representation")
    parser.add_argument("--metrics", action="store_true", help="Compute metrics and plot player count over time")
    parser.add_argument("--team-cluster", action="store_true", help="Cluster players by jersey color")
    
    args = parser.parse_args()

    print(f"Loading Model: {args.model}")
    print(f"Applying Tracker: {args.tracker}")
    
    if not os.path.exists(args.input_video):
        print(f"Error: Target Input video not found at: '{args.input_video}'")
        print("Please ensure you placed your royalty-free video file correctly based on the README instructions.")
        sys.exit(1)

    # 1. Initialize YOLO Athlete Detector
    detector = PlayerDetector(model_weight=args.model)

    # 2. Attach Tracker Core
    tracker = PlayerTracker(detector=detector, tracker_type=args.tracker)

    # 3. Configure Video Processor
    processor = VideoProcessor(
        tracker=tracker,
        input_path=args.input_video,
        output_path=args.output_video,
        args=args
    )

    # 4. Run full tracking pipeline end-to-end
    processor.process_video(conf=args.conf, screenshot_interval=args.screenshot_interval)

if __name__ == "__main__":
    main()
