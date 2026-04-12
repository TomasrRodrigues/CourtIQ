"""
Multi-object tracker that combines:

Kalman prediction for motion continuity
Hungarian assignment for global matching
Cost-based association using distance, IoU, and detection confidence
Each update step:

Predict existing tracks
Build track-detection cost matrix
Solve assignment
Update matched tracks
Age unmatched tracks and spawn new ones
Prune stale/weak tracks
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from tracking.kalman_tracker import KalmanTrack
from tracking.utils import iou, distance

DIST_THRESHOLD = 120

ALPHA = 0.50   # distance dominates
BETA  = 0.35   # IoU secondary
GAMMA = 0.15   # confidence tertiary
DELTA = 0.10   # motion consistency (trajectory-based tiebreaker)
MATCH_THRESHOLD = 0.8

# A new track must be seen at least MIN_HITS times before it is
# subject to hits-based pruning. The age guard below enforces this.
MIN_HITS = 3
MAX_MISSED = 10


class Tracker:
    """
    Manages active Kalman-based object tracks across frames.

    Attributes:
        tracks:              dict[int, KalmanTrack], active tracks by id.
        next_id:             int, next unique id for a new track.
        trajectory_builder:  Optional TrajectoryBuilder for motion-consistency
                             scoring inside the cost matrix.
    """
    def __init__(self, trajectory_builder=None):
        self.tracks = {}
        self.next_id = 0
        # Injected from outside so both the tracker and the pipeline share
        # the same instance. The tracker reads history (score); the pipeline
        # writes history (update). No global singleton needed.
        self.trajectory_builder = trajectory_builder

    def update(self, high_dets, low_dets):
        track_ids = list(self.tracks.keys())
        track_list = [self.tracks[tid] for tid in track_ids]

        predictions = {}
        for tid, track in zip(track_ids, track_list):
            predictions[tid] = track.predict()
            track.age += 1


        matches, unmatched_tracks, unmatched_dets = self._match(
            track_list, track_ids, predictions, high_dets
        )

        used_tracks = set()
        used_dets = set()

        for r, c in matches:
            track = track_list[r]
            det = high_dets[c]

            track.update(det["center"], det["bbox"])
            track.missed = 0

            used_tracks.add(track.id)
            used_dets.add(c)

        remaining_tracks = [t for t in track_list if t.id not in used_tracks]
        remaining_ids = [t.id for t in remaining_tracks]

        if len(remaining_tracks) > 0 and len(low_dets) > 0:
            predictions_low = {tid: predictions[tid] for tid in remaining_ids}

            matches_low, _, _ = self._match(
                remaining_tracks, remaining_ids, predictions_low, low_dets
            )

            for r, c in matches_low:
                track = remaining_tracks[r]
                det = low_dets[c]

                track.update(det["center"], det["bbox"])
                track.missed = 0

                used_tracks.add(track.id)


        for tid in track_ids:
            if tid not in used_tracks:
                track = self.tracks[tid]
                track.missed += 1
                track.hits = max(0, track.hits - 1)

                px, py = predictions[tid]

                if track.bbox is not None:
                    w = track.bbox[2] - track.bbox[0]
                    h = track.bbox[3] - track.bbox[1]

                    track.bbox = [
                        int(px - w / 2),
                        int(py - h / 2),
                        int(px + w / 2),
                        int(py + h / 2)
                    ]


        for i, det in enumerate(high_dets):
            if i not in used_dets:
                self._add_track(det)


        # Pruning logic — three ways a track survives:
        #
        #   1. t.confirmed            → already earned its place (≥3 hits).
        #   2. t.hits >= MIN_HITS     → accumulated enough hits.
        #   3. t.age < MIN_HITS       → NEW: age guard.
        #
        # Why the age guard?
        # _add_track() spawns tracks at the END of update(), after age
        # has already been incremented for existing tracks. New tracks
        # therefore have age=0 at pruning time. Without the age guard,
        # a brand-new track (hits=1) fails (1 >= 3) and is pruned
        # immediately — before it ever gets a second detection.
        # The guard gives every track MIN_HITS frames to accumulate
        # evidence before the hits-based gate applies.
        self.tracks = {
            tid: t for tid, t in self.tracks.items()
            if t.missed <= MAX_MISSED
            and (t.confirmed or t.hits >= MIN_HITS or t.age < MIN_HITS)
        }

        return self.tracks

    def _add_track(self, det):
        """
        Creates and registers a new track from one unmatched detection.

        Args:
            det (dict): Detection dictionary with center, class, and bbox.
        """
        track = KalmanTrack(self.next_id, det["center"], det["class"])
        track.bbox = det["bbox"]
        self.tracks[self.next_id] = track
        self.next_id += 1
    
    def _match(self, track_list, track_ids, predictions, detections):
        if len(track_list) == 0 or len(detections) == 0:
            return [], list(range(len(track_list))), list(range(len(detections)))

        cost_matrix = np.zeros((len(track_list), len(detections)), dtype=np.float32)

        for i, track in enumerate(track_list):
            tid = track_ids[i]

            for j, det in enumerate(detections):

                if track.class_name != det["class"]:
                    cost_matrix[i, j] = 1e6
                    continue

                pred = predictions[tid]
                dist = distance(pred, det["center"])

                vel = np.linalg.norm(track.kalman.statePost[2:4])
                adaptive_threshold = DIST_THRESHOLD + vel * 0.5

                if dist > adaptive_threshold:
                    cost_matrix[i, j] = 1e6
                    continue

                if track.bbox is not None:
                    iou_score = iou(track.bbox, det["bbox"])
                else:
                    iou_score = 0.0

                norm_dist = dist / DIST_THRESHOLD
                conf = det.get("conf", 1.0)

                # Motion-consistency score from trajectory history.
                # Returns 0.5 (neutral) when history is too short —
                # so it has no effect until MIN_HITS frames of history exist.
                if self.trajectory_builder is not None:
                    consistency = self.trajectory_builder.motion_consistency_score(
                        track.id, det["center"]
                    )
                else:
                    consistency = 0.5  # neutral fallback

                cost_matrix[i, j] = (
                    ALPHA * norm_dist +
                    BETA  * (1 - iou_score) +
                    GAMMA * (1 - conf) +
                    DELTA * consistency
                )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_tracks = list(range(len(track_list)))
        unmatched_dets = list(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] > MATCH_THRESHOLD:
                continue

            matches.append((r, c))
            unmatched_tracks.remove(r)
            unmatched_dets.remove(c)

        return matches, unmatched_tracks, unmatched_dets