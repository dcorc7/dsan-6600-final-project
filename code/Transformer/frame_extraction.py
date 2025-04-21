import os
import cv2
from pathlib import Path
import shutil

VIDEO_DIR = "../data/raw_videos"
FRAME_OUTPUT_DIR = "./cleaned_frames"
FPS = 10
DURATION = 8

os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)

def extract_last_n_seconds(video_path, output_dir, fps_target=10, seconds=8):
    cap = cv2.VideoCapture(str(video_path))
    fps_actual = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps_actual

    print(f"[INFO] Video: {video_path.name}, FPS: {fps_actual:.2f}, Duration: {duration_sec:.2f}s")

    frames_to_keep = int(fps_target * seconds)
    frame_interval = int(fps_actual / fps_target) if fps_actual >= fps_target else 1
    start_frame = max(0, total_frames - (frames_to_keep * frame_interval))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    count = 0
    saved = 0

    while cap.isOpened() and saved < frames_to_keep:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_name = f"frame_{saved:04d}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            saved += 1
        count += 1
    cap.release()


def process_video_file(video_path):
    video_name = Path(video_path).stem
    # Get the relative path from VIDEO_DIR to maintain train/test structure
    rel_path = os.path.relpath(video_path.parent, VIDEO_DIR)
    out_dir = os.path.join(FRAME_OUTPUT_DIR, rel_path, video_name)
    
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
        print(f"[SKIP] Already extracted frames for {video_name}")
        return
    os.makedirs(out_dir, exist_ok=True)
    print(f"[PROCESS] {video_name}")
    extract_last_n_seconds(video_path, out_dir, FPS, DURATION)

if __name__ == "__main__":
    video_paths = list(Path(VIDEO_DIR).rglob("*.mp4"))  # include both train/ and test/
    for vp in video_paths:
        process_video_file(vp)

    print("Done")