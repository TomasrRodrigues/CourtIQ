import cv2

class VideoLoader:
    def __init__(self, video_url):
        self.video_url = video_url

        print(f"Loading video: {self.video_url}")
        self.cap = cv2.VideoCapture(self.video_url)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_url}")
        
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.duration = self.frame_count / self.fps
        print(f"Video metadata - Frame count: {self.frame_count}, FPS: {self.fps}, Duration: {self.duration:.2f} seconds")
    
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.cap.isOpened():
            raise StopIteration
        
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        
        return frame

    def release(self):
        self.cap.release()
    