import cv2
import numpy as np
from data.video_loader import VideoLoader
from ultralytics import YOLO
import math

CLASS_MAP = {
    0: "person",
    32: "ball",
}

DIST_THRESHOLD = 80
MAX_MISSED = 10


class KalmanTrack:
    def __init__(self, track_id, center, cls):
        self.id = track_id
        self.class_name = cls
        self.missed = 0

        self.kalman = cv2.KalmanFilter(4, 2)

        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        self.kalman.statePre = np.array([
            [center[0]],
            [center[1]],
            [0.0],
            [0.0]
        ], dtype=np.float32)

        self.kalman.statePost = self.kalman.statePre.copy().astype(np.float32)

        self.bbox = None

    def predict(self):
        pred = self.kalman.predict()
        return int(pred[0][0]), int(pred[1][0])

    def update(self, center):
        measurement = np.array([
            [center[0]],
            [center[1]]
        ], dtype=np.float32)
        self.kalman.correct(measurement)



def distance(c1, c2):
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)



def quick_test():
    video_path = "data/clip_01.mp4"
    video = VideoLoader(video_path)
    model = YOLO("yolov8m.pt")

    tracks = {}
    next_id = 0

    for i, frame in enumerate(video):

        if i > 300:
            break

        detections = []
        results = model(frame)[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls not in CLASS_MAP:
                continue
            if conf < 0.4:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "center": (cx, cy),
                "class": CLASS_MAP[cls]
            })


        predictions = {}
        for tid, track in tracks.items():
            pred_center = track.predict()
            predictions[tid] = pred_center
            track.missed += 1  

        used_tracks = set()
        updated_tracks = {}

        for det in detections:
            best_id = None
            best_dist = float("inf")

            for tid, track in tracks.items():
                if track.class_name != det["class"]:
                    continue
                if tid in used_tracks:
                    continue

                pred_center = predictions[tid]
                d = distance(det["center"], pred_center)

                if d < best_dist and d < DIST_THRESHOLD:
                    best_dist = d
                    best_id = tid

            if best_id is not None:
                track = tracks[best_id]
                track.update(det["center"])
                track.bbox = det["bbox"]
                track.missed = 0

                updated_tracks[best_id] = track
                used_tracks.add(best_id)
            else:
                new_track = KalmanTrack(next_id, det["center"], det["class"])
                new_track.bbox = det["bbox"]

                updated_tracks[next_id] = new_track
                next_id += 1


        tracks = {
            tid: t for tid, t in updated_tracks.items()
            if t.missed <= MAX_MISSED
        }


        for tid, track in tracks.items():
            if track.bbox is None:
                continue

            x1, y1, x2, y2 = map(int, track.bbox)
            label = f"{track.class_name} ID:{tid}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Tracking", frame)
        print(f"Tracks: {len(tracks)} | Detections: {len(detections)}")

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    quick_test()