from data.video_loader import VideoLoader

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
    main()
