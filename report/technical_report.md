# Technical Report: Multi-Object Detection and Persistent ID Tracking

**Video Source**: [Publicly Accessible Pexels Soccer Field Footage](https://www.pexels.com/video/aerial-view-of-youth-soccer-match-on-green-field-31370176/)

## 1. Model & Tracker Used

### Detector
**Ultralytics YOLOv11** is utilized as the baseline detection model (using the standard `.pt` weights layout). It is configured for high processing throughput and robust generalization parameters, explicitly targeting the 'Person' dataset class for athlete detection.

### Tracking Algorithm
**ByteTrack** is employed as the primary multi-object tracking algorithm (`persist=True`). By leveraging a "tracking-by-detection" paradigm combined with robust data association, it builds unique persistent representations for each detected athlete.

## 2. Why This Combination Was Selected

**YOLOv11** represents the state-of-the-art in real-time object detection paradigms. Analyzing high-framerate sporting events requires rapid turnaround per frame without sacrificing generalized bounding box integrity. YOLOv11’s low-level representation capability successfully catches athletes reliably even as they rotate backwards, drop near the ground relative to the bounds of the frame, or blur severely across rapid cross-camera sprints.

**ByteTrack** was selected specifically to overcome the hard dropout flaws of classical trackers (e.g., standard SORT methods). In heavily congested athletic environments, if a player is momentarily obscured by another mid-sprint, baseline detectors output a lowered detection confidence on the partially obscured player. Standard trackers treat this lowered confidence as "noise", abandon the tracked ID entirely, and cast a completely new ID once they regain clarity. ByteTrack explicitly solves this by aggressively retaining low-confidence bounding box artifacts and linking their active identities sequentially. This reduces generalized identity assignment swaps precipitously while requiring virtually zero external deep-learning dependencies compared to complex ReID networks.

## 3. How ID Consistency is Maintained

ID integrity relies dynamically on two core mathematical concepts:

1. **Kalman Filtering:** The tracker generates mathematically predicted trajectories scaling across an active time axis. This effectively predicts where a localized bounding box *should* physically be in the subsequent frame prior to receiving the new true visual boundary prediction from YOLO.
2. **Hungarian Bipartite Matching:** This step pairs the physical filter predictions mathematically to the incoming frame detections. By associating high-confidence boxes first, and then resolving remaining low-confidence boxes using the Kalman predictions, tracked identities persist continuously even when athletes visually cross paths or wear similar team uniforms.

## 4. Challenges Faced & Observed Failure Cases

### Challenges Experienced
- **Threshold Optimization:** Finding and optimizing baseline confidence thresholds. High thresholds limit tracking strictly to perfectly clear camera orientations, whereas low thresholds naturally introduce background artifacts (such as stadium chairs or complex advertisement geometry) as participant vectors.
- **Velocity Skewing:** Very severe focal shifts—such as manual camera zooming or rapid physical panning—artificially skew the relative movement velocities sent to the Kalman Filter, forcing the tracker to briefly reassess motion vectors.

### Observed Failure Cases
- **Deep Occlusion Swaps:** In scenarios where athletes cross exactly in parallel over the localized camera viewing plane (forcing total continuous profile occlusion for extended segments of >6-8 frames), the matrix matching tracker occasionally mistakenly cross-swaps tracking allocations upon their final physical lateral separation.
- **ReID Failure on Extended Departures:** Due to computational boundaries naturally lacking deep cross-frame global integration, players completely departing the physical bounds of the camera frame line will have tracking severed. They are logically recast under computationally new IDs upon absolute field re-entry.

## 5. Possible Improvements

While the baseline infrastructure completes ID assignments smoothly, future architectural expansion could include:
- **Global Re-Identification (ReID):** Integrating deep appearance-based ReID networks (like OSNet or DeepSORT pipelines) to generate robust vector embeddings for each player, guaranteeing re-assignment even if they completely exit the frame for long intervals.
- **Camera Motion Compensation (CMC):** Employing background registration tools (like ECC - Enhanced Correlation Coefficient maximization) to mathematically negate background panning shifts, resulting in much cleaner absolute velocity readings for the tracking logic.
- **Enhanced Spatial Mapping:** Integrating true 4-point homography calibration natively via OpenCV `findHomography` to map athletes onto a strict geometrically accurate tactical soccer pitch grid, moving beyond rough estimations.

---

### Appendix: Optional Enhancements Implemented
To robustly demonstrate applied computer vision competency, five key baseline enhancements were successfully injected modularly on top of the original working timeline:
1. **Trajectory Visualization**: Historical footprints stored in memory deques trace age-faded trailing lines over players.
2. **Movement Heatmap**: Translates mass accumulation into scaled spatial bias temperature mapping.
3. **Bird's Eye Top-View Projection**: Applies localized Perspective Warp transforms to map field densities.
4. **Evaluation Metrics Engine**: Aggregation logs generating real-time population plots and bounds tracking.
5. **Team Heuristic Clustering**: Isolates jersey RGB vectors using a dynamic 2-cluster geometric split to properly separate "Team A" vs "Team B" identifiers automatically.
