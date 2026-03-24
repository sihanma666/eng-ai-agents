#kalman filter tracking
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from filterpy.kalman import KalmanFilter

def make_kalman_filter() -> KalmanFilter:
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=float)
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]], dtype=float)
    kf.R *= 10 # measurement noise
    kf.P *= 100 # initial uncertainty
    kf.Q *= 0.1 # process noise
    return kf

if __name__ == "__main__":
    model = YOLO('runs/detect/train/weights/best.pt')
    video_directory = Path(".")
    output_directory = Path("tracked")
    output_directory.mkdir(exist_ok=True)
    MAX_MISSED = 10 # frames before dropping tracker

    for video_path in sorted(video_directory.glob("*.mp4")):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_interval = max(1, int(fps/5)) # rate of 5 fps
        frame_index = 0
        kf = make_kalman_filter()

        out_path = output_directory / f"{video_path.stem}_tracked.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))
        initialized = False
        missed = 0
        trajectory: list[tuple[int, int]] = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_index % frame_interval == 0:
                results = model.predict(frame, conf=0.3, verbose=False)
                bbox = None

                for result in results:
                    if len(result.boxes) > 0:
                        box = result.boxes[0]
                        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                        cx = (x_min + x_max) / 2
                        cy = (y_min + y_max) / 2
                        bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
                        if not initialized:
                            kf.x = np.array([[cx], [cy], [0.0], [0.0]])
                            initialized = True
                        else:
                            kf.predict()
                            kf.update(np.array([[cx], [cy]]))
                        missed = 0
                        break

                if bbox is None and initialized:
                    kf.predict()
                    missed += 1
                    
                if initialized and missed < MAX_MISSED:
                    cx_pred = int(kf.x[0, 0])
                    cy_pred = int(kf.x[1, 0])
                    trajectory.append((cx_pred, cy_pred))
                    if bbox:
                        cv2.rectangle(frame, bbox[:2], bbox[2:], (0, 255, 0), 2)

                    for i in range(1, len(trajectory)):
                        cv2.line(frame, trajectory[i-1], trajectory[i], (0, 0, 255), 2)
                    writer.write(frame)

                if missed >= MAX_MISSED:
                    initialized = False
                    trajectory.clear()

            frame_index += 1

        cap.release()
        writer.release()
        print(f"Saved: {out_path}")
           