import os
import cv2
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

if __name__ == "__main__":
    #load pretrained drone detection model
    model = YOLO('runs/detect/train/weights/best.pt')

    video_directory = Path(".") # directory of .mp4 files
    detections_directory = Path("detections")
    detections_directory.mkdir(exist_ok=True)
    output_file = "detections.parquet"

    rows = []

    for video_path in sorted(video_directory.glob("*.mp4")):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps/5) # rate of 5 fps
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % frame_interval == 0:
                results = model.predict(frame, conf=0.3, verbose=False)
                for result in results:
                    if len(result.boxes) > 0:
                        out_name = f"{video_path.stem}_frame{frame_index:05d}.jpg"
                        cv2.imwrite(str(detections_directory / out_name), frame)
                        for box in result.boxes:
                            confidence = box.conf.item()
                            class_label = result.names[int(box.cls.item())]
                            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

                            rows.append({
                                "video_id": video_path.stem, 
                                "frame_index": frame_index,
                                "timestamp_sec": int(frame_index / fps),
                                "class_label": result.names[int(box.cls.item())],
                                "confidence": box.conf.item(),
                                "x_min": box.xyxy[0][0].item(),
                                "y_min": box.xyxy[0][1].item(),
                                "x_max": box.xyxy[0][2].item(),
                                "y_max": box.xyxy[0][3].item(),
                            })
            frame_index += 1
        cap.release()
    df = pd.DataFrame(rows)
    df.to_parquet(output_file)
    print(f"{len(df)} detections saved to {output_file}")