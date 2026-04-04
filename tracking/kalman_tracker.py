import cv2
import numpy as np

class KalmanTrack:
    def __init__(self, track_id, center, cls):
        self.id = track_id
        self.class_name = cls
        self.missed = 0
        self.hits = 1
        self.confirmed = False
        self.bbox = None
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
        pred = self.kalman.predict()
        self.kalman.statePost[2] *= 0.9
        self.kalman.statePost[3] *= 0.9
        return int(pred[0][0]), int(pred[1][0])

    def update(self, center, bbox=None):
        measurement = np.array([[center[0]], [center[1]]], dtype=np.float32)
        self.kalman.correct(measurement)

        self.trajectory.append(center)

        self.hits += 1
        if self.hits >= 3:
            self.confirmed = True

        self.bbox = bbox
