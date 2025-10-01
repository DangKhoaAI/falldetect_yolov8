from ultralytics import YOLO
from pathlib import Path
import sys

# Đường dẫn model và video
MODEL_PATH = Path(__file__).parent.parent / "model" / "best.pt"
VIDEO_PATH = Path(__file__).parent.parent / "data" / "test.mp4"

def run_inference(video_path):
    # Load YOLOv8 model
    model = YOLO(str(MODEL_PATH))

    # Run inference trực tiếp trên video
    results = model.predict(
        source=str(video_path),
        save=True,   # lưu video kết quả
        project=str(Path(__file__).parent.parent / "runs" / "inference"),
        name="video_test",  # tên thư mục con
        exist_ok=True       # không tạo thư mục mới nếu đã tồn tại
    )

    # In ra kết quả (boxes, classes, conf)
    for result in results:
        print(result.boxes)

if __name__ == "__main__":
    vid_path = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH
    run_inference(vid_path)
