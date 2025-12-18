from ultralytics import YOLO        # YOLOv8 object detection model
import cv2                          # OpenCV for video processing
import cvzone                       # Utility library for drawing UI elements
import numpy as np                  # Numerical operations
from sort import *                  # SORT tracker for multi-object tracking

# -------------------------------------------------------
# VIDEO & MODEL SETUP
# -------------------------------------------------------

cap = cv2.VideoCapture("../Videos/cars.mp4")     # Load input video file
model = YOLO("../Yolo-Weights/yolov8l.pt")       # Load pretrained YOLOv8 model
classNames = model.names                         # Mapping: class_id -> class_name

# -------------------------------------------------------
# COUNTING LINE (USER DEFINED - DO NOT MODIFY)
# -------------------------------------------------------

limits = [400, 297, 673, 297]   # (x1, y1, x2, y2)
# Vehicles are counted ONLY when crossing this exact line

# -------------------------------------------------------
# DETECTION & TRACKING CONFIGURATION
# -------------------------------------------------------

conf_thresh = 0.3                                   # Minimum detection confidence
vehicle_classes = {"car", "truck", "bus", "motorbike"}  # Classes to be tracked

tracker = Sort(
    max_age=20,        # Frames to keep track alive without detection
    min_hits=3,        # Minimum detections before track is confirmed
    iou_threshold=0.3  # IOU threshold for track association
)

# -------------------------------------------------------
# ROI (REGION OF INTEREST)
# -------------------------------------------------------

area = [
    (470, 220),
    (700, 220),
    (700, 470),
    (50, 446),
    (375, 310)
]
poly = cv2.convexHull(np.array(area, np.int32))  # Convert ROI to convex polygon

# -------------------------------------------------------
# COUNT STORAGE (TRACK IDS)
# -------------------------------------------------------

totalCar = []         # Stores unique car IDs
totalTruck = []      # Stores unique truck IDs
totalBus = []        # Stores unique bus IDs
totalMotorbike = []  # Stores unique motorbike IDs

# -------------------------------------------------------
# TEXT DRAWING CONFIG
# -------------------------------------------------------

TEXT_SCALE = 0.9
TEXT_THICKNESS = 2
TEXT_OFFSET = 6

# -------------------------------------------------------
# IOU FUNCTION (TRACK ↔ DETECTION CLASS MATCHING)
# -------------------------------------------------------

def iou_xyxy(a, b):
    # Bounding box A
    ax1, ay1, ax2, ay2 = a
    # Bounding box B
    bx1, by1, bx2, by2 = b

    # Intersection coordinates
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    # Intersection area
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Areas of both boxes
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    # Union area
    union = area_a + area_b - inter_area + 1e-6

    return inter_area / union

# -------------------------------------------------------
# MAIN PROCESSING LOOP
# -------------------------------------------------------

while True:
    success, img = cap.read()   # Read next frame
    if not success:
        break                  # Stop if video ends

    detections = []             # Stores detections for SORT
    det_classes = []            # Stores class for each detection

    # ---------------- YOLO DETECTION --------------------

    for r in model(img, stream=True, verbose=False):
        for box in r.boxes:

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            conf = float(box.conf[0])       # Detection confidence
            cls = int(box.cls[0])           # Class index
            currentClass = classNames.get(cls, "obj")

            # Filter by class and confidence
            if currentClass in vehicle_classes and conf >= conf_thresh:

                # Compute center point
                cx = x1 + w // 2
                cy = y1 + h // 2

                # Check if center is inside ROI
                if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
                    detections.append([x1, y1, x2, y2, conf])
                    det_classes.append(currentClass)

                    # Draw detection bounding box
                    cvzone.cornerRect(img, (x1, y1, w, h), l=8)

    # ---------------- SORT TRACKING ---------------------

    detections_np = (
        np.array(detections, dtype=np.float32)
        if detections else
        np.empty((0, 5), dtype=np.float32)
    )

    resultsTracker = tracker.update(detections_np)

    # -------- TRACK → CLASS MATCHING --------------------

    det_boxes = [d[:4] for d in detections]
    track_to_class = {}

    for trk in resultsTracker:
        tx1, ty1, tx2, ty2, track_id = map(int, trk)

        best_iou = 0
        best_idx = -1

        # Match track box with detection box
        for j, db in enumerate(det_boxes):
            score = iou_xyxy((tx1, ty1, tx2, ty2), db)
            if score > best_iou:
                best_iou = score
                best_idx = j

        if best_idx != -1 and best_iou >= 0.2:
            track_to_class[track_id] = det_classes[best_idx]
        else:
            track_to_class[track_id] = "obj"

    # ---------------- DRAW & COUNT ----------------------

    for trk in resultsTracker:
        x1, y1, x2, y2, track_id = map(int, trk)

        w, h = x2 - x1, y2 - y1
        cx = x1 + w // 2
        cy = y1 + h // 2

        obj_class = track_to_class.get(track_id, "obj")

        # Draw label
        cvzone.putTextRect(
            img,
            f"{obj_class} | ID:{track_id}",
            (max(0, x1), max(20, y1 - 10)),
            scale=TEXT_SCALE,
            thickness=TEXT_THICKNESS,
            offset=TEXT_OFFSET
        )

        # Draw center point
        cv2.circle(img, (cx, cy), 4, (255, 0, 255), cv2.FILLED)

        # COUNTING LOGIC (STRICTLY BASED ON USER LIMITS)
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if obj_class == "car" and track_id not in totalCar:
                totalCar.append(track_id)
            elif obj_class == "truck" and track_id not in totalTruck:
                totalTruck.append(track_id)
            elif obj_class == "bus" and track_id not in totalBus:
                totalBus.append(track_id)
            elif obj_class == "motorbike" and track_id not in totalMotorbike:
                totalMotorbike.append(track_id)

    # ---------------- DRAW COUNTING LINE ----------------

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 255), 4)
    cv2.circle(img, (limits[0], limits[1]), 6, (0, 0, 255), cv2.FILLED)
    cv2.circle(img, (limits[2], limits[3]), 6, (0, 0, 255), cv2.FILLED)

    # ---------------- UI COUNTERS -----------------------

    cvzone.putTextRect(img, f"Cars: {len(totalCar)}", (50, 40), scale=1.5)
    cvzone.putTextRect(img, f"Trucks: {len(totalTruck)}", (50, 90), scale=1.5)
    cvzone.putTextRect(img, f"Buses: {len(totalBus)}", (50, 140), scale=1.5)
    cvzone.putTextRect(img, f"Motorbikes: {len(totalMotorbike)}", (50, 190), scale=1.5)

    cv2.imshow("Vehicle Counter", img)

    # Exit when ESC is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# -------------------------------------------------------
# CLEANUP
# -------------------------------------------------------

cap.release()
cv2.destroyAllWindows()
