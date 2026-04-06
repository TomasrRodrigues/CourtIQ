from ultralytics import YOLO


CLASS_MAP = {
    0: "person",
    32: "ball",
}


class YOLODetector:
    """
    YOLOv8-based detector with task-specific post-processing for court scenes.

    The detector:
    - Runs YOLO inference on the input frame.
    - Keeps only classes defined in CLASS_MAP.
    - Applies class-specific confidence thresholds.
    - Filters by geometry and scene priors (area, aspect ratio, vertical region).
    """

    def __init__(self, model_path="yolov8m.pt", device=0, imgsz=1280):
        """
        Args:
            model_path (str): Path to YOLO model weights.
            device (int | str): Inference device (GPU id or "cpu").
            imgsz (int): Inference image size passed to YOLO predict.
        """
        self.model = YOLO(model_path)
        self.model.fuse()
        self.device = device
        self.imgsz = imgsz

    def detect(self, frame):
        """
        Runs detection on one frame and returns filtered detections.

        Args:
            frame (np.ndarray): Input image in HxWxC format.

        Returns:
            list[dict]: Detection items with:
                - bbox: [x1, y1, x2, y2]
                - center: (cx, cy)
                - class: mapped class label (for example "person", "ball")
                - conf: confidence score
        """
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            device=self.device,
            conf=0.1,
            iou=0.5,
            verbose=False
        )[0]

        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Ignore classes not explicitly tracked by this pipeline.
            if cls not in CLASS_MAP:
                continue

            # Class-specific confidence gates tuned for person/ball behavior.
            if cls == 0 and conf < 0.6:
                continue
            if cls == 32 and conf < 0.1:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            w = x2 - x1
            h = y2 - y1
            area = w * h

            # Remove tiny noise boxes and implausibly large regions.
            if area < 100 or area > 50000:
                continue

            aspect_ratio = h / (w + 1e-6)

            # Person boxes are expected to be vertically elongated.
            if cls == 0 and not (1.2 < aspect_ratio < 6):
                continue

            H, W = frame.shape[:2]

            # Ignore detections too close to the top of frame.
            if cy < H * 0.2:
                continue

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "center": (cx, cy),
                "class": CLASS_MAP[cls],
                "conf": conf
            })

        return detections