import numpy as np
from scipy.optimize import linear_sum_assignment
from tracking.kalman_tracker import KalmanTrack
from detection.utils import distance
from tracking.utils import iou

DIST_THRESHOLD = 120
IOU_THRESHOLD = 0.15
MATCH_THRESHOLD = 1.2
ALPHA = 0.8
BETA = 0.2
MIN_HITS = 3

class Tracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0

    def update(self, detections):
        track_ids = list(self.tracks.keys())
        track_list = [self.tracks[tid] for tid in track_ids]

        predictions = {}
        for tid, track in zip(track_ids, track_list):
            predictions[tid] = track.predict()

        num_tracks = len(track_list)
        num_dets = len(detections)

        if num_tracks == 0:
            for det in detections:
                self._add_track(det)
            return self.tracks

        cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)

        for i, track in enumerate(track_list):
            for j, det in enumerate(detections):

                if track.class_name != det["class"]:
                    cost_matrix[i, j] = 1e6
                    continue

                track_id = track_ids[i]
                dist = distance(predictions[track_id], det["center"])

                # Distance gating
                if dist > DIST_THRESHOLD:
                    cost_matrix[i, j] = 1e6
                    continue

                # IoU gating
                if track.bbox is not None:
                    iou_score = iou(track.bbox, det["bbox"])
                else:
                    iou_score = 0.0

                # Normalize distance
                norm_dist = dist / DIST_THRESHOLD

                # Final cost
                cost_matrix[i, j] = ALPHA * norm_dist + BETA * (1 - iou_score)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        used_tracks = set()
        used_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] > MATCH_THRESHOLD:
                continue

            track = track_list[r]
            det = detections[c]

            track.update(det["center"], det["bbox"])
            track.missed = 0

            used_tracks.add(track.id)
            used_dets.add(c)

        # unmatched tracks
        for tid in track_ids:
            if tid not in used_tracks:
                track = self.tracks[tid]
                track.missed += 1

                px, py = predictions[tid]
                if track.bbox is not None:
                    w = track.bbox[2] - track.bbox[0]
                    h = track.bbox[3] - track.bbox[1]

                    track.bbox = [
                        int(px - w/2),
                        int(py - h/2),
                        int(px + w/2),
                        int(py + h/2)
                    ]

        # new tracks
        for j, det in enumerate(detections):
            if j not in used_dets:
                self._add_track(det)

        # remove old tracks
        self.tracks = {
            tid: t for tid, t in self.tracks.items()
            if t.missed <= 10
        }

        return self.tracks

    def _add_track(self, det):
        track = KalmanTrack(self.next_id, det["center"], det["class"])
        track.bbox = det["bbox"]
        self.tracks[self.next_id] = track
        self.next_id += 1
