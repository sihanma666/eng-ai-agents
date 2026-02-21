#train.py
from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO('yolo26n-seg.pt')

    # Train the model on the Carparts Segmentation dataset
    results = model.train(data="carparts-seg.yaml", epochs=100, imgsz=640, workers=0)
