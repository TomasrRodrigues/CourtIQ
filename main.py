import cv2
import numpy as np
from data.video_loader import VideoLoader
from ultralytics import YOLO
import math
from scipy.optimize import linear_sum_assignment

CLASS_MAP = {
    0: "person",
    32: "ball",
}

DIST_THRESHOLD = 120
IOU_THRESHOLD = 0.1
MAX_MISSED = 10
MIN_HITS_TO_CONFIRM = 3


def distance(c1, c2):
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

    return inter / (area1 + area2 - inter + 1e-6)



class KalmanTrack:
    def __init__(self, track_id, center, cls):
        self.id = track_id
        self.class_name = cls
        self.missed = 0

        self.hits = 1
        self.confirmed = False

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

        self.hits += 1
        if self.hits >= MIN_HITS_TO_CONFIRM:
            self.confirmed = True


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
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "center": (cx, cy),
                "class": CLASS_MAP[cls]
            })


        track_ids = list(tracks.keys())
        track_list = [tracks[tid] for tid in track_ids]

        num_tracks = len(track_list)
        num_dets = len(detections)

        predictions = []
        for track in track_list:
            pred_center = track.predict()
            predictions.append(pred_center)
            track.missed += 1


        if num_tracks > 0 and num_dets > 0:
            cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)

            alpha = 0.7
            beta = 0.3

            for i, track in enumerate(track_list):
                for j, det in enumerate(detections):

                    if track.class_name != det["class"]:
                        cost_matrix[i, j] = 1e6
                        continue

                    if track.bbox is not None:
                        iou_score = iou(track.bbox, det["bbox"])
                        if iou_score < IOU_THRESHOLD:
                            cost_matrix[i, j] = 1e6
                            continue
                    else:
                        iou_score = 0

                    dist = distance(predictions[i], det["center"])

                    cost_matrix[i, j] = alpha * dist + beta * (1 - iou_score)


            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            used_tracks = set()
            used_dets = set()
            updated_tracks = {}

            for r, c in zip(row_ind, col_ind):

                if r >= num_tracks or c >= num_dets:
                    continue

                if cost_matrix[r, c] > DIST_THRESHOLD:
                    continue

                track_id = track_ids[r]
                track = tracks[track_id]
                det = detections[c]

                track.update(det["center"])
                track.bbox = det["bbox"]
                track.missed = 0

                updated_tracks[track_id] = track

                used_tracks.add(track_id)
                used_dets.add(c)

            for tid in track_ids:
                if tid not in used_tracks:
                    track = tracks[tid]
                    updated_tracks[tid] = track

            for j, det in enumerate(detections):
                if j in used_dets:
                    continue

                new_track = KalmanTrack(next_id, det["center"], det["class"])
                new_track.bbox = det["bbox"]

                updated_tracks[next_id] = new_track
                next_id += 1

            tracks = {
                tid: t for tid, t in updated_tracks.items()
                if t.missed <= MAX_MISSED
            }

        else:
            for det in detections:
                new_track = KalmanTrack(next_id, det["center"], det["class"])
                new_track.bbox = det["bbox"]
                tracks[next_id] = new_track
                next_id += 1


        for tid, track in tracks.items():

            if not track.confirmed:
                continue

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