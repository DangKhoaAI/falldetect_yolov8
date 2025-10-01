from ultralytics import YOLO
from pathlib import Path
import sys


MODEL_PATH = Path(__file__).parent.parent / "model" / "best.pt"
IMAGE_PATH = Path(__file__).parent.parent / "data" / "test1.jpg"  # Change as needed


def run_inference(image_path):
    # Load YOLOv8 model
    model = YOLO(str(MODEL_PATH))
    # Run inference
    results = model.predict(source=str(image_path), save=True, project=str(Path(__file__).parent.parent / "runs" / "inference"))
    # Print results
    for result in results:
        print(result.boxes)

if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    run_inference(img_path)