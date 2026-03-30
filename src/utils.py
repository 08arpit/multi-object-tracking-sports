import cv2

def draw_annotations(frame, results):
    """
    Custom annotation drawing function to display thick bounding boxes, unique IDs, 
    and detection confidence natively.
    """
    annotated_frame = frame.copy()
    
    # Graceful exit if no valid tracked detection boxes exist
    if results is None or not hasattr(results, 'boxes') or results.boxes is None or len(results.boxes) == 0 or results.boxes.id is None:
        return annotated_frame
        
    boxes = results.boxes.xyxy.cpu().numpy()
    track_ids = results.boxes.id.int().cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    
    # Gracefully intercept confidence if tracker outputs drops it
    try:
        confs = results.boxes.conf.cpu().numpy()
    except AttributeError:
        confs = [0.0] * len(boxes)
    
    for box, track_id, conf, cls in zip(boxes, track_ids, confs, classes):
        # Process only class index 0 (person/player)
        if int(cls) != 0:
            continue
            
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, y2  # Feet coordinates
        
        # 1. Draw player bounding box with very thick lines for absolute visibility
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
        
        # 2. Bottom-center filled circle marking exact trajectory focal point 
        cv2.circle(annotated_frame, (cx, cy), 6, (0, 0, 255), -1)
        
        # 3. Draw persistent ID label coupled with the specific detection confidence
        label = f"ID: {track_id} ({conf:.2f})"
        
        # Text background for extreme contrast overlay
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated_frame, (x1, y1 - 30), (x1 + w, y1), (0, 255, 0), -1)
        
        # Overlay text label
        cv2.putText(annotated_frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
    return annotated_frame

def save_screenshot(frame, filepath):
    """Saves the current video frame as an image file."""
    cv2.imwrite(filepath, frame)
