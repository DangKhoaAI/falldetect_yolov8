python
from ultralytics import YOLO

def train_yolov8():
    # Load pretrained model
    model = YOLO("yolov8s.pt")

    # Train
    model.train(
        data="configs/mydata.yaml",
        epochs=100,
        imgsz=640,
        batch=16
    )

if __name__ == "__main__":
    train_yolov8()