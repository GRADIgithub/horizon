import cvzone
from ultralytics import YOLO
import cv2
import numpy as np
import math
from sort import *

# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

cap = cv2.VideoCapture("people.mp4")

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

mask = cv2.imread('sample.png')

limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]

total_countUp = []
total_countDown = []


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
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255))
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255))

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        w, h = x2-x1, y2-y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=8, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3)

        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx, cy), 3, (255, 255, 255), cv2.FILLED)

        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1]-15 < cy < limitsUp[3]+15:
            if total_countUp.count(id) == 0:
                total_countUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0))

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1]-15 < cy < limitsDown[3]+15:
            if total_countDown.count(id) == 0:
                total_countDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0))

    cvzone.putTextRect(img, f' UP: {len(total_countUp)}', (929, 345), colorR=(139, 195, 75))
    cvzone.putTextRect(img, f' Down: {len(total_countDown)}', (929, 425), colorR=(50, 50, 230))

    cv2.imshow("Images", img)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
