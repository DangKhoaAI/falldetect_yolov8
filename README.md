# Fall Detection with YOLOv8

This project implements a fall detection system using YOLOv8, fine-tuned on custom datasets.

## Project Structure

fall-detection-yolov8/
├── data/ # dataset 
├── configs/ # file cấu hình (dataset, train hyperparams)
├── notebooks/ # Jupyter notebooks (train, eval, demo)
├── src/ # Python source code
├── runs/ # YOLO outputs (auto-generated)
├── requirements.txt # dependencies
├── README.md # giới thiệu
└── .gitignore # ignore các file không cần push


## Quick Start

### Install
```bash
pip install -r requirements.txt

Train

yolo task=detect mode=train model=yolov8s.pt data=configs/mydata.yaml epochs=100 imgsz=640

or via Python:

from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(data="configs/mydata.yaml", epochs=100, imgsz=640)