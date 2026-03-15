# Real-Time Body Detection
Detects hands and body parts in real time using YOLOv8 and OpenCV.

## Setup
1. Create a virtual environment and activate it
2. pip install -r requirements.txt

## Model weights
Download hand detector from https://drive.google.com/drive/folders/1xrgut_eCsRZECjwfp89ar9UlxHuSgz2J
Download body detector from https://drive.google.com/drive/folders/1W7ISidhyQXdpGb_-L6o5CRM5PATzOT2k

Or retrain from scratch:
1. Hand dataset from Roboflow (https://universe.roboflow.com/catwithawand/hand-detection-fuao9)
   Body dataset from Roboflow (https://universe.roboflow.com/dataregenerator/human-body-kcosm)
2. Run python train.py