from ultralytics import YOLO
import cv2

def run_inference(weights="runs/detect/train/weights/best.pt", source="data/images/val"):
    model = YOLO(weights)
    results = model.predict(source=source, save=True)
    for r in results:
        r.show()  # hiển thị
        # r.save() # lưu kết quả

if __name__ == "__main__":
    run_inference()
