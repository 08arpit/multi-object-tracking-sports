from ultralytics import YOLO

class PlayerDetector:
    """
    Wrapper for the YOLO model configured for player detection.
    """
    def __init__(self, model_weight='yolo11n.pt'):
        """
        Initializes the YOLO model for detection.
        Args:
            model_weight (str): Path to the YOLO model weights.
        """
        self.model = YOLO(model_weight)

    def detect(self, frame, conf=0.3):
        """
        Runs object detection on a single frame.
        """
        # We target class '0' (person) which works best for detecting athletes
        results = self.model.predict(frame, classes=[0], conf=conf, verbose=False)
        return results[0]
