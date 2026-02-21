#detect.py
import os
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

if __name__ == "__main__":
    #load pretrained car parts model
    model = YOLO('runs/segment/train2/weights/best.pt')

    frames_dir = Path("frames")
    output_file = "detections.parquet"

    rows = []

    frame_files = sorted(frames_dir.glob("*.jpg"))
    print(f"frames to process: {len(frame_files)}")


    for frame_path in frame_files:
        frame_index = int(frame_path.stem.split("_")[1])
        timestamp_sec = frame_index * 5

        results = model.predict(frame_path)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf.item()
                class_label = result.names[int(box.cls.item())]
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

                rows.append({
                    "video_id": "YcvECxtXoxQ", # yt video: rav4_review
                    "frame_index": frame_index,
                    "timestamp_sec": timestamp_sec,
                    "class_label": class_label,
                    "confidence": confidence,
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max
                })
    df = pd.DataFrame(rows)
    df.to_parquet(output_file)
    print(f"{len(df)}Detections saved to {output_file}")