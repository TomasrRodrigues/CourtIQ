import numpy as np
from scipy.optimize import linear_sum_assignment
from tracking.kalman_tracker import KalmanTrack
from detection.utils import distance

class Tracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0

    def update(self, detections):
        track_ids = list(self.tracks.keys())
        track_list = [self.tracks[tid] for tid in track_ids]

        predictions = [t.predict() for t in track_list]

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

                dist = distance(predictions[i], det["center"])
                cost_matrix[i, j] = dist

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        used_tracks = set()
        used_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] > 120:
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
                self.tracks[tid].missed += 1

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
