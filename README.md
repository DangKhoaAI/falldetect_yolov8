# Fall Detection with YOLOv8

This project implements a fall detection system using YOLOv8, fine-tuned on custom datasets.

## Project Structure
```
fall-detection-yolov8/
├── data/ # dataset 
├── configs/ #  (dataset, train hyperparams)
├── notebooks/ # Jupyter notebooks (train, eval, demo)
├── src/ # Python source code
├── requirements.txt # dependencies
├── models/ #Trained models 
├── README.md 
└── .gitignore 
```
## How to run

### Installation
```bash
pip install -r requirements.txt
```

### Training
To train the model, run the following command:
```bash
python src/train.py
```
This will train the model using the configuration in `configs/mydata.yaml`.

### Inference
You can run inference on a video, an image, or a webcam feed.

#### Video
To run inference on a video, use the following command:
```bash
python src/inference/video_test.py --video_path [path_to_video]
```
If you don't provide a path to a video, it will use the default video file.

#### Picture
To run inference on a picture, use the following command:
```bash
python src/inference/picture_test.py --image_path [path_to_picture]
```
If you don't provide a path to a picture, it will use the default picture file.

#### Camera
To run inference on a webcam feed, use the following command:
```bash
python src/inference/camera_test.py
```

## Model
This project utilizes a YOLOv8m model, which was trained for 20 epochs. The model configuration can be found in `models/yolov8m/args.yaml`.

## Dataset
The model was trained on a custom dataset with 2 classes: `fall` and `normal`. The dataset configuration is defined in `configs/mydata.yaml`, with training images located in `data/images/train` and validation images in `data/images/val`.

## Results
Training results and metrics are stored in the `models/yolov8m/` directory
