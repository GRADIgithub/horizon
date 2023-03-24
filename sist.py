import cvzone
from ultralytics import YOLO
import cv2
import numpy as np
import math
from sort import *

# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

cap = cv2.VideoCapture("sist_test.mp4")

model = YOLO('../Yolo-Files/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
               "teddy bear", "hair drier", "toothbrush"]

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

mask = cv2.imread('test_mask.png')

limits = [110, 580, 310, 683]

total_count = []


while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2-x1, y2-y1

            conf = math.ceil((box.conf[0]*100))/100

            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.4:

                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(0, y1)), scale=1, thickness=1)
                # cvzone.cornerRect(img, (x1, y1, w, h))

                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255))

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        w, h = x2-x1, y2-y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=8, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3)

        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx, cy), 3, (255, 255, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] < cy < limits[3]:
            if total_count.count(id) == 0:
                total_count.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0))

    cvzone.putTextRect(img, f' Count: {len(total_count)}', (50, 50))

    cv2.imshow("Images", img)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
