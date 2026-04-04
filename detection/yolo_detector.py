from ultralytics import YOLO


CLASS_MAP = {
    0: "person",
    32: "ball",
}

class YOLODetector:
    def __init__(self, model_path="yolov8m.pt", device=0):
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, frame, imgsz=1280):
        results = self.model.predict(frame, imgsz=imgsz, device=self.device)[0]

        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls not in CLASS_MAP:
                continue

            # class-specific thresholds
            if cls == 0 and conf < 0.65:
                continue
            if cls == 32 and conf < 0.35:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "center": (cx, cy),
                "class": CLASS_MAP[cls],
                "conf": conf
            })

        return detections
