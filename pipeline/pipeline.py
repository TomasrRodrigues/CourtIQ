class CourtIQPipeline:
    """
    High-level inference pipeline that chains detection, optional filtering,
    and tracking for a single video frame.
    """

    def __init__(self, detector, tracker, filter_fn=None):
        """
        Args:
            detector: Object exposing detect(frame) -> list[dict].
            tracker: Object exposing update(detections) -> tracks structure.
            filter_fn (callable | None): Optional post-detector filter:
                filter_fn(raw_detections, frame) -> list[dict].
        """
        self.detector = detector
        self.tracker = tracker
        self.filter_fn = filter_fn

    def process(self, frame):
        """
        Runs one end-to-end pipeline step on a frame.

        Flow:
        1. Detect raw objects.
        2. Optionally filter detections.
        3. Update tracker with filtered detections.

        Args:
            frame: Input image/frame (typically numpy array in BGR format).

        Returns:
            dict: {
                "raw_detections": detector output before filtering,
                "filtered_detections": detections used for tracking,
                "tracks": tracker state/output after update
            }
        """
        raw_detections = self.detector.detect(frame)

        detections = raw_detections
        if self.filter_fn:
            detections = self.filter_fn(raw_detections, frame)

        tracks = self.tracker.update(detections)

        return {
            "raw_detections": raw_detections,
            "filtered_detections": detections,
            "tracks": tracks
        }