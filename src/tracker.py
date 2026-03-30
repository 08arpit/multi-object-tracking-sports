class PlayerTracker:
    """
    Wrapper for Ultralytics built-in tracking (ByteTrack or BoT-SORT).
    """
    def __init__(self, detector, tracker_type="bytetrack.yaml"):
        """
        Initializes the tracker using an existing detector model.
        Args:
            detector (PlayerDetector): Instance of the PlayerDetector.
            tracker_type (str): Type of tracker, default 'bytetrack.yaml'.
        """
        self.detector = detector
        self.tracker_type = tracker_type

    def track(self, frame, conf=0.3):
        """
        Runs both detection and continuous tracking on a single frame.
        """
        # Tracking is performed purely on the 'person' class (index 0)
        # persist=True ensures consistent unique IDs
        results = self.detector.model.track(
            frame, 
            persist=True, 
            classes=[0], 
            conf=conf, 
            tracker=self.tracker_type, 
            verbose=False
        )
        return results[0]
