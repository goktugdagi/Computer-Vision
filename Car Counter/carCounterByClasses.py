from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import numpy as np

cap = cv2.VideoCapture("Videos/cars.mp4")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

model = YOLO("Yolo-Weights/yolov8l.pt")
mask = cv2.imread("mask.png")

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [400, 297, 673, 297]

# Her sınıf için ID listesi
totalCar, totalTruck, totalBus, totalMotorbike = [], [], [], []

while True:
    success, img = cap.read()
    if not success:
        break

    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))
    classes = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                # Nesne bilgisi metni kutudan biraz yukarı yerleştirildi
                cvzone.putTextRect(img, f"{currentClass} {conf}", (x1, y1 - 10), scale=0.6, thickness=1, offset=3)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                classes.append(currentClass)

    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), color=(0, 0, 255), thickness=5)

    for result, cls in zip(resultsTracker, classes):
        x1, y1, x2, y2, id = map(int, result)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # ID metni de üstte gösterilecek şekilde ayarlandı
        cvzone.putTextRect(img, f"{id}", (x1, y1 - 35), scale=1, thickness=2, offset=4)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if cls == "car" and id not in totalCar:
                totalCar.append(id)
            elif cls == "truck" and id not in totalTruck:
                totalTruck.append(id)
            elif cls == "bus" and id not in totalBus:
                totalBus.append(id)
            elif cls == "motorbike" and id not in totalMotorbike:
                totalMotorbike.append(id)
            cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # Sayaç bilgileri sabit konumlarda ayrı ayrı yazdırıldı
    cvzone.putTextRect(img, f"Cars: {len(totalCar)}", (50, 40), scale=1.5, thickness=2, offset=4, colorR=(255, 0, 255))
    cvzone.putTextRect(img, f"Trucks: {len(totalTruck)}", (50, 90), scale=1.5, thickness=2, offset=4, colorR=(255, 0, 255))
    cvzone.putTextRect(img, f"Buses: {len(totalBus)}", (50, 140), scale=1.5, thickness=2, offset=4, colorR=(255, 0, 255))
    cvzone.putTextRect(img, f"Motorbikes: {len(totalMotorbike)}", (50, 190), scale=1.5, thickness=2, offset=4, colorR=(255, 0, 255))

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
