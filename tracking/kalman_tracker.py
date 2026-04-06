import cv2
import numpy as np


class KalmanTrack:
    def __init__(self, track_id, center, cls):
        self.id = track_id
        self.class_name = cls

        self.missed = 0
        self.hits = 1
        self.confirmed = False
        self.age = 0

        self.bbox = None
        self.embedding = None
        self.trajectory = []

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

        self.kalman.statePost = self.kalman.statePre.copy()

    def predict(self):
        """
        Runs one Kalman prediction step and returns predicted center (x, y).
        Also applies lightweight stabilization to velocity:
        - Damps vx and vy by 10% to reduce drift during long gaps.
        - Clamps velocity to +/-50 pixels/frame to suppress outliers.

        Returns:
            tuple[int, int]: Predicted center coordinates.
        """
        pred = self.kalman.predict()

        self.kalman.statePost[2] *= 0.9
        self.kalman.statePost[3] *= 0.9

        max_vel = 50
        self.kalman.statePost[2] = np.clip(self.kalman.statePost[2], -max_vel, max_vel)
        self.kalman.statePost[3] = np.clip(self.kalman.statePost[3], -max_vel, max_vel)

        return int(pred[0][0]), int(pred[1][0])

    def update(self, center, bbox=None, embedding=None):
        """
        Corrects the filter with a new observation and updates track metadata.
        Args:
            center (tuple[float, float] | list[float]): Observed object center (x, y).
            bbox (list[int] | None): Optional detection bbox [x1, y1, x2, y2].
            embedding (np.ndarray | None): Optional appearance vector.

        Behavior:
        - Stores embedding when provided.
        - Runs Kalman correction from the new center measurement.
        - Appends center to trajectory history.
        - Increments hits and marks track as confirmed after 3 hits.
        - Smooths bbox with exponential moving average (alpha=0.7) when previous bbox exists.
        """
        if embedding is not None:
            self.embedding = embedding

        measurement = np.array([[center[0]], [center[1]]], dtype=np.float32)
        self.kalman.correct(measurement)

        self.trajectory.append(center)

        self.hits += 1
        if self.hits >= 3:
            self.confirmed = True

        if bbox is not None:
            if self.bbox is not None:
                alpha = 0.7
                self.bbox = [
                    int(alpha * self.bbox[i] + (1 - alpha) * bbox[i])
                    for i in range(4)
                ]
            else:
                self.bbox = bbox