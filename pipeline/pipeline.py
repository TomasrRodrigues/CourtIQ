class CourtIQPipeline:
    """
    High-level inference pipeline that chains detection, optional filtering,
    tracking, and optional trajectory recording for a single video frame.
    """

    def __init__(self, detector, tracker, filter_fn=None, trajectory_builder=None):
        """
        Args:
            detector:            Object exposing detect(frame) -> list[dict].
            tracker:             Object exposing update(high, low) -> tracks.
            filter_fn:           Optional post-detector filter:
                                     filter_fn(raw_detections, frame) -> list[dict].
            trajectory_builder:  Optional TrajectoryBuilder instance.
                                 When provided, it is updated every frame
                                 automatically. Callers can then query it
                                 for motion features between frames.
        """
        self.detector = detector
        self.tracker = tracker
        self.filter_fn = filter_fn
        self.trajectory_builder = trajectory_builder

        # Monotonically increasing frame counter.
        # Used as the time axis inside TrajectoryBuilder's regression.
        # Not wall-clock time — frame index is stable regardless of
        # processing speed, which is what the velocity math requires.
        self.frame_idx = 0

    def process(self, frame):
        """
        Runs one end-to-end pipeline step on a frame.

        Flow:
            1. Detect raw objects.
            2. Optionally filter detections.
            3. Split into high / low confidence for two-stage tracking.
            4. Update tracker.
            5. Update trajectory builder (if attached).

        Args:
            frame: Input image in BGR format (numpy array).

        Returns:
            dict with keys:
                raw_detections     — detector output before filtering.
                filtered_detections — detections passed to the tracker.
                tracks             — tracker state after this frame.
                frame_idx          — index of this frame (0-based).
        """
        raw_detections = self.detector.detect(frame)

        detections = raw_detections
        if self.filter_fn:
            detections = self.filter_fn(raw_detections, frame)

        high_conf = [d for d in detections if d["conf"] >= 0.5]
        low_conf  = [d for d in detections if d["conf"] <  0.5]

        tracks = self.tracker.update(high_conf, low_conf)

        if self.trajectory_builder is not None:
            self.trajectory_builder.update(tracks, self.frame_idx)
            self.trajectory_builder.prune(set(tracks.keys()))

        current_frame = self.frame_idx
        self.frame_idx += 1

        return {
            "raw_detections":      raw_detections,
            "filtered_detections": detections,
            "tracks":              tracks,
            "frame_idx":           current_frame,
        }