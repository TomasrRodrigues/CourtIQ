
import cv2

from detection.yolo_detector import YOLODetector
from tracking.tracker import Tracker
from pipeline.pipeline import CourtIQPipeline


def main():
    cap = cv2.VideoCapture("data/clip_01.mp4")

    detector = YOLODetector()
    tracker = Tracker()
    pipeline = CourtIQPipeline(detector, tracker)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracks = pipeline.process(frame)

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
