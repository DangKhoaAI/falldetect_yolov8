import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "model" / "best.pt"
model = YOLO(str(MODEL_PATH))
# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Run inference
    results = model(frame, verbose=False)  # verbose=False để tắt log

    # Lấy frame có vẽ bounding box
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow("Real-time Hand Gesture Detection (YOLOv8)", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
