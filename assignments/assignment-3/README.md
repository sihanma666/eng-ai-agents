Video 1
[![Drone Tracking Video 1](https://img.youtube.com/vi/KOPEBPMGO7g/0.jpg)](https://youtu.be/KOPEBPMGO7g)

Video 2
[![Drone Tracking Video 2](https://img.youtube.com/vi/tz-vkLdm2Ac/0.jpg)](https://youtu.be/tz-vkLdm2Ac)

Dataset: 
[mamasihan/drone-detections](https://huggingface.co/datasets/mamasihan/drone-detections)

## Detector
- Model: YOLOv8n fine-tuned on a drone detection dataset from Roboflow Universe ([drone-detection-a1tsf v6](https://universe.roboflow.com/drone-detection-g4d3g/drone-detection-a1tsf/dataset/6), CC BY 4.0).
The base model (`yolov8n.pt`) was trained on COCO and has no drone class, so fine-tuning on a labeled drone dataset was necessary. Training ran for 30 epochs at 640px image size. Inference uses a confidence threshold of 0.3.

- `part_A.py` handles training. `part_B.py` runs inference on all `.mp4` files in a directory, saves frames containing detections to `detections/`, and writes `detections.parquet`.

- On each sampled frame (5 fps), if a detection is found: `predict()` then `update()` with measured center. If no detection, `predict()` only, increment missed counter
And if missed frames exceed `MAX_MISSED = 10`, tracker resets

- `part_C.py` outputs one video per input with the detector bounding box (green) and Kalman-estimated trajectory polyline (red) overlaid, containing only frames where the drone is present.

## Failure Modes
- Occlusion: if the drone passes behind an object for more than 10 frames the tracker resets and loses the trajectory history
- Multiple drones: the pipeline only tracks the highest-confidence detection per frame, a second drone would be ignored
- Fast motion: the constant velocity model struggles with sharp direction changes; the tracker may lag behind or diverge briefly
- False positives: small birds or aircraft at distance can trigger detections, introducing noise into the trajectory
- Output codec compatibility: OpenCV's `mp4v` codec produces files not playable in most browsers or modern players; re-encoding with ffmpeg (`libx264`) is required for H.264 compatible output
