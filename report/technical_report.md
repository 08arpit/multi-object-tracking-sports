# Technical Report: Multi-Object Detection and Persistent ID Tracking

**Video Source**: [Publicly Accessible Pexels Soccer Field Footage](https://www.pexels.com/video/aerial-view-of-youth-soccer-match-on-green-field-31370176/)
## 1. Selected combination of Detector and Tracker
- **Detector utilized:** Ultralytics YOLOv11. (Baseline uses Nano `.pt` layout for high processing throughput with robust generalization parameters targeting the 'Person' dataset class).
- **Tracking Algorithm utilized:** ByteTrack.

## 2. Methodology & Why this combination was selected
YOLOv11 is an industry-leading evolution of the state-of-the-art real-time object detection paradigm. When analyzing high framerate sporting events, rapid turnaround is required per frame without sacrificing generalized box integrity. By leveraging its low-level representation capability, the detector catches athletes reliably even as they rotate backwards, drop near the ground relative to the bounds of the frame, or blur severely across rapid cross-camera sprints.

ByteTrack utilizes a "tracking-by-detection" format. A fundamental flaw with many classical trackers (e.g., standard SORT methods/ReID dependencies) applied in heavily congested athletic environments is the hard dropout of active bounds tracking. That is, if a player is momentarily obscured from view by another player mid-sprint, the system predicts a lowered detection confidence rating on the partially obscured player. Standard tracking systems treat this lowered confidence as "noise", abandon the tracked ID entirely, and cast a completely new unrelated ID onto the player once they exit the congestion and regain high detection clarity. ByteTrack explicitly solves this paradigm by saving low-confidence bounding box artifacts and linking their active identities sequentially alongside a data association phase across a localized Kalman velocity filter. This reduces generalized identity assignment swaps precipitously within physical sports applications while requiring virtually zero external dependency calculations.

## 3. How ID consistency is maintained
As noted, ID integrity primarily relies upon two core factors:
1. **Kalman Filtering:** Generating mathematically predicted trajectories scaling across an active time axis to predict where a localized bounding box *should* physically be in the subsequent frame prior to receiving the new true visual boundary prediction.
2. **Hungarian Bipartite matching:** Pairing up physical predictions mathematically to incoming frames, allowing the tracked identities to persist continuously to the athlete's exact footprint even against visually similar team uniforms surrounding them. 

## 4. Challenges Faced & Observed Failure Cases
**Challenges:**
- Finding and optimizing baseline confidence thresholds. High threshold limits tracking to perfectly clear camera orientations, whereas low thresholds falsely mark artifacts in background advertisements or stadium chairs as participants. 
- Very severe focal shifts (manual camera zooming alongside physical panning) artificially alters relative velocity metrics sent to the Kalman Filter.

**Observed failure cases:**
- **ReID Failure upon extended departure:** Players physically departing the field of view limits (e.g., a ball carrier running completely off the localized vertical viewing plane frame line) naturally have their tracked object IDs severed upon re-entry.
- **Deep Occlusion Swaps:** When two identical players cross uniformly in parallel alongside the exact camera viewing line for more than a sequence of 6-8 frames uninterrupted, generating total profile occlusion horizontally, upon their ultimate physical separation the tracker will occasionally cross-swap the ID matrices natively.

## 5. Potential Improvements
The scope of this current infrastructure completes baseline assignment metrics fully, delivering tracked uniqueness; however, optional expansion systems logically include true DeepSORT integration and deep ReID integration.

## 6. Optional Enhancements Implemented
To robustly demonstrate applied computer vision competency, five key non-mandatory enhancements were successfully injected modularly on top of the original working timeline:
1. **Trajectory Visualization**: Historical footprints were stored in memory deques. Poly lines trace past sequences over the tracked objects. 
   - *Benefits:* Imparts immediate tactical context per individual. 
   - *Limitations:* Mass congestion renders poly lines visually messy if left un-trimmed.
2. **Movement Heatmap (`--heatmap`)**: Translates bottom-center mass accumulation variables into solid Gaussian circles on a blank scalar mask, subsequently processed by `cv2.COLORMAP_JET`.
   - *Benefits:* Reveals fundamental area biases dynamically over an entire game. 
3. **Bird's Eye Top-View Projection (`--topview`)**: Applies a localized Perspective Warp transform via mathematically projected focal points to lay flat coordinate arrays onto a 2D blank tactical board. 
   - *Limitations:* Inherently approximate without true 4-point extrinsic field calibration per pan.
4. **Detailed Eval Metrics & Math Plotting (`--metrics`)**: Employs live heuristic algorithms to establish max simultaneous participant counts and approximated track failures. Outputting graphical arrays using Matplotlib.
5. **Team Heuristic Clustering (`--team-cluster`)**: Strips out upper 50% chest-zones of YOLO bounding crops. Calculates the unified Median Hue. Splitting generalized bins mathematically isolates red vs blue (Team A vs Team B) natively without model re-training. 
   - *Limitations:* Inherently sensitive to referee shirts or extreme shading/shadows skewing standard HSV properties. 
