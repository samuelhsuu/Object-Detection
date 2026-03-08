import cv2
from ultralytics import YOLO
import time

#load model weights (small)
model = YOLO("yolov8s.pt")
#open camera, can also use filepath for video
cap = cv2.VideoCapture(0)

if not cap.isOpened():
  print("Error: could not open camera")
  exit()

#objects with confidence <0.4 are filtered out
CONFIDENCE_THRESHOLD = 0.4
prev_time = time.time()

SKIP_FRAMES = 2
frame_count = 0
last_results = [] # previous frame detections

while True:
  #frame is a 3d array containing image data
  ret, frame = cap.read()

  frame_count+=1

  if frame_count % SKIP_FRAMES == 0:
    # run inference on every other frame
    last_results = model(frame, conf=CONFIDENCE_THRESHOLD, imgsz = 320, verbose=False)

  if not ret:
    print("Error: failed to read frame")
    break
  #model is called like a function to analyze the current frame, specify imgsz for optimization
  # results = model(frame, conf=CONFIDENCE_THRESHOLD, imgsz = 320, verbose=False)

  #results only really has one item (the current frame)
  for result in last_results:
    #each box has all detections >0.4 confidence
    for box in result.boxes:
      x1, y1, x2, y2 = map(int, box.xyxy[0])
      confidence = float(box.conf[0])
      class_id = int(box.cls[0])
      label = model.names[class_id]

      cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255,100), 1)

      text = f"{label} {confidence:.0%}"
      (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
      cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+6,y1), (0,255,100), -1)

      cv2.putText(frame, text, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

  curr_time = time.time()
  fps = 1.0/(curr_time-prev_time)
  prev_time = curr_time
  cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 200, 255), 2)

  cv2.imshow("Object Detection", frame)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break
cap.release()
cv2.destroyAllWindows()