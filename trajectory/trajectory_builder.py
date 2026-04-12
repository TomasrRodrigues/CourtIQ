"""
TrajectoryBuilder — long-horizon temporal record and motion feature extraction.

Separation of concerns
─────────────────────
  KalmanTrack.trajectory   Short rolling window (≤ 50 pts, deque).
                           Lives on the track object. Used by the Kalman
                           cycle and short-horizon visualisation.

  TrajectoryBuilder        Full per-track history (up to max_history frames).
                           Lives here, separate from filter state.
                           Used for motion reasoning, visualisation, and
                           eventually event detection.

Why keep them separate?
  The Kalman filter is optimised for frame-to-frame prediction.
  It does not care about what happened 200 frames ago.
  But reasoning about direction trends, speed profiles, and motion
  consistency — the things that prevent ID switches — requires
  a longer, explicit record. Mixing that into KalmanTrack would
  couple filter state with analytical concerns.

Primary contribution to robustness
───────────────────────────────────
  motion_consistency_score() returns how well a candidate detection
  fits a track's established motion trend. When two tracks cross and
  their plain distance/IoU costs are nearly identical, this score
  breaks the tie using direction evidence — the detection that
  continues each track's natural direction wins.
"""

from collections import deque
import numpy as np


class TrajectoryBuilder:

    def __init__(self, max_history: int = 500, smoothing_window: int = 8):
        """
        Args:
            max_history (int):
                Maximum observations stored per track.
                500 frames ≈ 16 seconds at 30 fps.

            smoothing_window (int):
                Frames used when computing smoothed velocity.
                Larger = more stable direction estimate, slower to react.
                8 frames ≈ 0.27 seconds at 30 fps is a good starting point.
        """
        self.trajectories: dict = {}   # tid -> deque of (frame_idx, center)
        self.max_history = max_history
        self.smoothing_window = smoothing_window

    # ──────────────────────────────────────────────────────────────────
    # Core update
    # ──────────────────────────────────────────────────────────────────

    def update(self, tracks: dict, frame_idx: int) -> None:
        """
        Records each confirmed track's current centre for this frame.

        Only confirmed tracks are stored. Tentative tracks (hits < 3)
        are too unreliable to build motion models from.

        Args:
            tracks:    dict[int, KalmanTrack] from Tracker.update().
            frame_idx: Current frame number (time axis for regression).
        """
        for tid, track in tracks.items():
            if not track.confirmed or not track.trajectory:
                continue

            if tid not in self.trajectories:
                self.trajectories[tid] = deque(maxlen=self.max_history)

            # Kalman-corrected centre — already smoothed at the single-frame level.
            center = track.trajectory[-1]
            self.trajectories[tid].append((frame_idx, center))

    def prune(self, active_ids: set) -> None:
        """
        Removes history for tracks that are no longer active.

        Future extension: archive instead of delete, to enable re-ID
        when a player temporarily leaves and re-enters the frame.

        Args:
            active_ids: set[int] of IDs currently in the tracker.
        """
        dead_ids = [tid for tid in self.trajectories if tid not in active_ids]
        for tid in dead_ids:
            del self.trajectories[tid]

    # ──────────────────────────────────────────────────────────────────
    # Motion features
    # ──────────────────────────────────────────────────────────────────

    def get_smoothed_velocity(self, track_id: int) -> tuple:
        """
        Estimates current velocity via linear regression over the last
        `smoothing_window` observations.

        Why linear regression instead of frame difference?
        ──────────────────────────────────────────────────
        Frame difference (v = pos[t] - pos[t-1]) is maximally sensitive
        to noise — one jittered detection and the estimate flips.
        Linear regression fits a trend across N frames simultaneously.
        Noise in individual frames averages out, giving a stable
        direction estimate.

        Returns:
            (vx, vy) in pixels/frame.  (0.0, 0.0) if insufficient data.
        """
        traj = self.trajectories.get(track_id)
        if traj is None or len(traj) < 2:
            return (0.0, 0.0)

        recent = list(traj)[-self.smoothing_window:]

        frames = np.array([f for f, _ in recent], dtype=np.float32)
        xs     = np.array([c[0] for _, c in recent], dtype=np.float32)
        ys     = np.array([c[1] for _, c in recent], dtype=np.float32)

        frames -= frames[0]    # shift to 0 for numerical stability
        if frames[-1] == 0:
            return (0.0, 0.0)

        # polyfit(x, y, deg=1) -> [slope, intercept]. Slope = velocity.
        vx = float(np.polyfit(frames, xs, 1)[0])
        vy = float(np.polyfit(frames, ys, 1)[0])

        return (vx, vy)

    def get_speed(self, track_id: int) -> float:
        """Scalar speed in pixels/frame from the smoothed velocity."""
        vx, vy = self.get_smoothed_velocity(track_id)
        return float(np.sqrt(vx ** 2 + vy ** 2))

    def motion_consistency_score(self, track_id: int,
                                  candidate_center: tuple) -> float:
        """
        Scores how well a candidate detection fits this track's motion trend.

        This is the core method for reducing ID switches.

        How it works
        ────────────
        1. Compute smoothed velocity (vx, vy) from recent trajectory.
        2. Project one step forward from the last known position.
        3. Measure deviation of the candidate from that projection.
        4. Normalise to [0, 1]:  0 = candidate perfectly follows trend,
                                 1 = candidate maximally contradicts trend.

        Why this breaks ID switches at crossing points
        ───────────────────────────────────────────────
        When players cross, distance and IoU become ambiguous — both
        cross-assignments have nearly equal cost. Direction is still
        strong: player A is going right, player B going left. This score
        makes the tracker prefer the assignment that preserves direction.

        Returns:
            float in [0, 1].
            Returns 0.5 (neutral) when history is too short — meaning
            "no evidence either way, don't penalise anything."
        """
        traj = self.trajectories.get(track_id)
        if traj is None or len(traj) < 3:
            return 0.5  # neutral

        vx, vy = self.get_smoothed_velocity(track_id)
        _, last_center = traj[-1]

        projected_x = last_center[0] + vx
        projected_y = last_center[1] + vy

        deviation = float(np.sqrt(
            (candidate_center[0] - projected_x) ** 2 +
            (candidate_center[1] - projected_y) ** 2
        ))

  
        speed = float(np.sqrt(vx ** 2 + vy ** 2))
        normalised = min(deviation / (2.0 * speed + 50.0), 1.0)

        return normalised



    def get_trajectory(self, track_id: int) -> list:
        """Full [(frame_idx, (cx, cy)), ...] record for a track."""
        traj = self.trajectories.get(track_id)
        return list(traj) if traj else []

    def get_centers(self, track_id: int) -> list:
        """Just the (cx, cy) sequence — convenient for drawing polylines."""
        return [c for _, c in self.get_trajectory(track_id)]