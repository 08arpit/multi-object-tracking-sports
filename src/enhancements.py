import cv2
import numpy as np
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import os

class PipelineEnhancements:
    """
    Handles all optional features: Trajectories, Heatmaps, Object Counting, 
    Top-down Projection, Metrics, and Team Clustering.
    """
    def __init__(self, frame_width, frame_height, args):
        self.args = args
        self.width = frame_width
        self.height = frame_height
        
        # Trajectories (last 45 frames)
        self.trajectories = defaultdict(lambda: deque(maxlen=45))
        
        # Heatmap
        self.heatmap_accum = np.zeros((frame_height, frame_width), dtype=np.float32)
        
        # Metrics & Count
        self.player_counts_per_frame = []
        self.unique_ids = set()
        self.track_lengths = defaultdict(int)
        self.max_simultaneous = 0
        
        # Top-down projection (Approximate homography for a general pitch)
        src_pts = np.float32([
            [0, frame_height * 0.4], [frame_width, frame_height * 0.4], 
            [0, frame_height], [frame_width, frame_height]
        ])
        self.top_w, self.top_h = 600, 800
        dst_pts = np.float32([
            [0, 0], [self.top_w, 0], 
            [0, self.top_h], [self.top_w, self.top_h]
        ])
        self.homography_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.topdown_canvas = np.zeros((self.top_h, self.top_w, 3), dtype=np.uint8)
        # Draw some fake pitch lines on the topdown canvas
        cv2.rectangle(self.topdown_canvas, (10, 10), (self.top_w-10, self.top_h-10), (255, 255, 255), 2)
        cv2.line(self.topdown_canvas, (10, self.top_h//2), (self.top_w-10, self.top_h//2), (255, 255, 255), 2)

    def process_frame(self, frame, annotated_frame, results):
        current_ids = []
        
        # Safely handle empty frames or tracks
        if results is None or not hasattr(results, 'boxes') or results.boxes is None:
            return annotated_frame
            
        if len(results.boxes) == 0 or results.boxes.id is None:
            return annotated_frame
            
        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.int().cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        
        simultaneous = 0
        
        for box, track_id, cls in zip(boxes, track_ids, classes):
            if int(cls) != 0: continue
            simultaneous += 1
            current_ids.append(track_id)
            self.unique_ids.add(track_id)
            self.track_lengths[track_id] += 1
            
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, y2  # Compute bottom center (feet)
            
            # 1. Trajectory Visualization with Age Fading
            if getattr(self.args, 'trajectories', False):
                self.trajectories[track_id].append((cx, cy))
                pts = list(self.trajectories[track_id])
                max_len = len(pts)
                for i in range(1, max_len):
                    # Fading trail: newest points are brightest yellow, older fade to black
                    alpha = i / max_len
                    color = (0, int(255 * alpha), int(255 * alpha))
                    cv2.line(annotated_frame, pts[i-1], pts[i], color, 2)
                    
            # 2. Movement Heatmap Accumulation
            if getattr(self.args, 'heatmap', False):
                # Add heat via a solid circle (will be blurred later)
                cv2.circle(self.heatmap_accum, (cx, cy), 20, 1.0, -1)
                
            # 3. Enhanced Team/Role Clustering via 2-cluster KMeans
            if getattr(self.args, 'team_cluster', False):
                # Crop the upper 60% body (ignore legs/boots)
                jersey = frame[y1:y1+int((y2-y1)*0.6), x1:x2]
                team_label = "Unknown"
                team_color = (200, 200, 200) # Default grey
                
                if jersey.size > 10:
                    pixels = np.float32(jersey.reshape(-1, 3))
                    # 2-Cluster K-means logic
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
                    
                    # Discover dominant cluster color purely within the jersey bounds
                    counts = np.bincount(labels.flatten())
                    dominant_bgr = centers[np.argmax(counts)]
                    
                    # Translate to HSV to read hue
                    dom_hsv = cv2.cvtColor(np.uint8([[dominant_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
                    dom_hue = dom_hsv[0]
                    
                    # Distinct thresholds for Team A & Team B representation
                    if dom_hue < 35 or dom_hue > 150: # Warm colors (Red/Yellow boundary)
                        team_label = "Team A"
                        team_color = (0, 255, 255) # Yellow font visual
                    else: # Cool colors (Blue/Green boundary)
                        team_label = "Team B"
                        team_color = (255, 150, 0) # Distinct Light Blue font visual
                        
                # Ensure Team Text pops with clear positioning directly over the player
                cv2.putText(annotated_frame, team_label, (x1, y1 - 38), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, team_color, 2)

            # 4. Top-view / Bird's-eye projection
            if getattr(self.args, 'topview', False):
                pt = np.array([[[cx, cy]]], dtype=np.float32)
                warped_pt = cv2.perspectiveTransform(pt, self.homography_matrix)
                wx, wy = int(warped_pt[0][0][0]), int(warped_pt[0][0][1])
                if 0 <= wx < self.top_w and 0 <= wy < self.top_h:
                    # Draw persistent dot on top-down canvas
                    cv2.circle(self.topdown_canvas, (wx, wy), 4, (0, 255, 0), -1)

        # Update Max Simultaneous Players map
        if simultaneous > self.max_simultaneous:
            self.max_simultaneous = simultaneous
            
        # 5. Object count over time
        self.player_counts_per_frame.append(len(current_ids))
        if getattr(self.args, 'metrics', False):
            # Text Overlay of current frame count
            overlay_text = f"Tracked Players: {len(current_ids)}"
            cv2.putText(annotated_frame, overlay_text, (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        return annotated_frame

    def generate_final_outputs(self, output_dir):
        """Export all requested optional files after pipeline completion."""
        # Ensure output dir exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Heatmap export
        if getattr(self.args, 'heatmap', False):
            norm_heat = cv2.normalize(self.heatmap_accum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            norm_heat = cv2.GaussianBlur(norm_heat, (81, 81), 0)
            color_heat = cv2.applyColorMap(norm_heat, cv2.COLORMAP_JET)
            color_heat[norm_heat < 5] = 0 
            cv2.imwrite(os.path.join(output_dir, "movement_heatmap.jpg"), color_heat)
            
        # Top-view export
        if getattr(self.args, 'topview', False):
            cv2.imwrite(os.path.join(output_dir, "topview.jpg"), self.topdown_canvas)
            
        # Metrics export
        if getattr(self.args, 'metrics', False):
            # 1. Save plot
            plt.figure(figsize=(10, 5))
            plt.plot(self.player_counts_per_frame, label="Tracked Players", color="green")
            plt.xlabel("Frame Index")
            plt.ylabel("Player Count")
            plt.title("Player Count Over Time")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(output_dir, "player_count_plot.png"))
            plt.close()
            
            # 2. Compute metrics
            avg_length = np.mean(list(self.track_lengths.values())) if self.track_lengths else 0
            approx_switches = max(0, len(self.unique_ids) - self.max_simultaneous)
            
            with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
                f.write("--- Simple Evaluation Metrics ---\n")
                f.write(f"Total unique IDs observed: {len(self.unique_ids)}\n")
                f.write(f"Average track length (frames): {avg_length:.2f}\n")
                f.write(f"Approximate number of ID switches: {approx_switches}\n")
                f.write(f"Max simultaneous players tracked: {self.max_simultaneous}\n")
