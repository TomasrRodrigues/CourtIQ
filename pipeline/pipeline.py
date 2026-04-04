class CourtIQPipeline:
    def __init__(self, detector, tracker):
        self.detector = detector
        self.tracker = tracker

    def process(self, frame):
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections)
        return tracks