#retrieve.py
import os
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
from datasets import load_dataset
VIDEO_ID = "YcvECxtXoxQ"
if __name__ == "__main__":
    hf_dataset = load_dataset("aegean-ai/rav4-exterior-images", split="train")
    yt_dataset = pd.read_parquet("detections.parquet")

    model = YOLO('runs/segment/train2/weights/best.pt')
    segments_rows = []

    for image in hf_dataset:
        results = model.predict(image["image"])
        query_labels = set() # store detected labels for this image 
        for result in results:
            boxes = result.boxes
            for box in boxes:
                query_labels.add(result.names[box.cls.item()])
        matches = yt_dataset[yt_dataset["class_label"].isin(query_labels)]    
        print(f"{image['timestamp_sec']}: query labels: {query_labels}, {len(matches)} matches")   

        timestamps = sorted(matches["timestamp_sec"].unique())
        intervals = []
        if timestamps:
            start = timestamps[0]
            end = timestamps[0]
            for t in timestamps[1:]:
                if t - end <= 120: # 120 second gap allowed between detections
                    end = t
                else:
                    intervals.append((start, end))
                    start = t
                    end = t
            intervals.append((start, end))

        for start, end in intervals:
            count = len(matches[(matches["timestamp_sec"] >= start) & (matches["timestamp_sec"] <= end)])
            print(f" [{start} - {end}] {count} detections")
            segments_rows.append({
                "query_image": image["timestamp_sec"],
                "query_labels": ", ".join(sorted(query_labels)),
                "start_sec": start,
                "end_sec": end,
                "detection_count": count,
                "clip_url": f"https://www.youtube.com/watch?v={VIDEO_ID}&t={start}s"
            })

    df_segments = pd.DataFrame(segments_rows)
    df_segments.to_parquet("segments.parquet", index=False)
    print(f"Saved {len(df_segments)} segments to segments.parquet")
