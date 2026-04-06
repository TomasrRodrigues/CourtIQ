
import cv2

from detection.yolo_detector import YOLODetector
from tracking.tracker import Tracker
from pipeline.pipeline import CourtIQPipeline

def court_filter(detections, frame):
    H, W = frame.shape[:2]
    filtered = []

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = det["center"]

        w = x2 - x1
        h = y2 - y1
        area = w * h

        if area < 100 or area > 50000:
            continue

        aspect_ratio = h / (w + 1e-6)

        if det["class"] == "person" and not (1.2 < aspect_ratio < 6):
            continue

        if cy < H * 0.2:
            continue

        filtered.append(det)

    return filtered

def main():
    cap = cv2.VideoCapture("data/clip_01.mp4")

    detector = YOLODetector()
    tracker = Tracker()
    pipeline = CourtIQPipeline(detector, tracker, filter_fn=court_filter)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = pipeline.process(frame)
        tracks = output["tracks"]

        for track in tracks.values():
            if not track.confirmed or track.bbox is None:
                continue

            x1, y1, x2, y2 = map(int, track.bbox)
            label = f"{track.class_name} ID:{track.id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow("CourtIQ", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
