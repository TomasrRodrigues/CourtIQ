import cv2

from detection.yolo_detector import YOLODetector
from tracking.tracker import Tracker
from trajectory.trajectory_builder import TrajectoryBuilder
from pipeline.pipeline import CourtIQPipeline


def court_filter(detections, frame):
    """
    Scene-level filter applied after detection, before tracking.
    Removes detections that violate court geometry priors.

    Note: low-level geometric gates (area, aspect ratio, vertical
    region) already run inside YOLODetector. This filter exists for
    court-specific logic that the detector cannot know about.
    Currently it replicates those gates — a future version should
    add court-boundary polygon checks and role-based filters.
    """
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


def draw_trajectory(frame, centers, color=(0, 200, 255), max_pts=30):
    """
    Draws a fading polyline for a track's recent history.

    Args:
        centers:  List of (cx, cy) tuples, oldest first.
        max_pts:  How many recent points to draw (avoids visual clutter).
    """
    pts = centers[-max_pts:]
    for i in range(1, len(pts)):
        # Fade older segments by reducing alpha linearly
        alpha = i / len(pts)
        faded = tuple(int(c * alpha) for c in color)
        cv2.line(frame, tuple(map(int, pts[i - 1])), tuple(map(int, pts[i])), faded, 2)


def main():
    cap = cv2.VideoCapture("data/clip_01.mp4")

    detector     = YOLODetector()
    traj_builder = TrajectoryBuilder(max_history=500, smoothing_window=8)
    tracker      = Tracker(trajectory_builder=traj_builder)

    pipeline = CourtIQPipeline(
        detector,
        tracker,
        filter_fn=court_filter,
        trajectory_builder=traj_builder,
    )

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

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw trajectory trail from the builder's long-horizon record.
            centers = traj_builder.get_centers(track.id)
            if len(centers) > 1:
                draw_trajectory(frame, centers)

        cv2.imshow("CourtIQ", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()