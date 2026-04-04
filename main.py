import cv2

from data.video_loader import VideoLoader
from ultralytics import YOLO

CLASS_MAP = {
    0: "person",
    32: "ball",
}

def quick_test():
    video_path = "data/clip_01.mp4"
    video = VideoLoader(video_path)
    model = YOLO("yolov8m.pt")

    next_id = 0
    tracks = {}

    for i, frame in enumerate(video):

        if i > 300:   # limit for now
            break
        detections = []

        results = model(frame)[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls not in CLASS_MAP:
                continue

            if conf < 0.4:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            label = f"{CLASS_MAP[cls]} {conf:.2f}"
            
            
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "center": (cx, cy),
                "score": conf,
                "class": CLASS_MAP[cls]
            })



        new_tracks = {}
        used_ids = set()

        for det in detections:
            best_id = None
            best_dist = float('inf')

            cx, cy = det["center"]
            for track_id, track in tracks.items():
                if track["class"] != det["class"]:
                    continue
                if track_id in used_ids:
                    continue
                tx, ty = track["center"]
                dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5

                if dist < best_dist and dist < 50:  # distance threshold
                    best_dist = dist
                    best_id = track_id
            
            if best_dist < 50:
                new_tracks[best_id] = det
                used_ids.add(best_id)
            else:
                new_tracks[next_id] = det
                next_id += 1
            
            for track_id, det in new_tracks.items():
                x1, y1, x2, y2 = map(int, det["bbox"])

                label = f"{det['class']} ID:{track_id}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        tracks = new_tracks
        cv2.imshow("Detections", frame)
        print(len(detections))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Show result
    cv2.imshow("Detections", frame)
    cv2.waitKey(0)

    video.release()
    cv2.destroyAllWindows()


def main():
    print("\n\nLoading videos...\n")
    i=1
    while True:
        try:
            if i<=9:
                video_path = f"data/clip_0{i}.mp4"
            else:
                video_path = f"data/clip_{i}.mp4"
            video = VideoLoader(video_path)
            i+=1
        except Exception as e:
            break
            
    print("\nAll videos loaded successfully.\n")

if __name__ == "__main__":
    quick_test()
