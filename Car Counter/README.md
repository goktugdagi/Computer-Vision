
# 🚗 Vehicle Counter with YOLOv8 + SORT

This project performs **vehicle detection, tracking, and counting** on a video stream using **YOLOv8** for object detection and **SORT** for multi-object tracking.

Vehicles are counted **only when they cross a manually defined counting line (`limits`)**, exactly as specified by the user — no automatic repositioning or dynamic scaling is applied.

---

## 📌 Features

- YOLOv8-based vehicle detection
- SORT multi-object tracking (ID-based)
- Class-based counting:
  - Car
  - Truck
  - Bus
  - Motorbike
- ROI (Region of Interest) filtering
- Exact, user-defined counting line
- Each vehicle counted **only once**
- Real-time visualization with OpenCV

---

## 🧠 How the Pipeline Works

1. **Read video frame**
2. **YOLOv8 detects objects**
3. Filter detections:
   - Confidence threshold
   - Vehicle class
   - Inside ROI polygon
4. **SORT tracker** assigns unique IDs
5. Track boxes are matched with detection classes using **IoU**
6. Vehicles are counted when their center crosses the defined line
7. Counters are displayed on screen

---

## 📐 Counting Logic

Counting line definition:

```python
limits = [400, 297, 673, 297]
```

A vehicle is counted if:

```python
limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15
```

This ensures:
- Counting happens **only** at the specified line
- No assumptions or auto-adjustments are made

---

## 📍 ROI (Region of Interest)

Only vehicles whose **center point** lies inside the ROI polygon are tracked:

```python
area = [(470, 220), (700, 220), (700, 470), (50, 446), (375, 310)]
```

This reduces false positives and irrelevant detections.

---

## 🧩 Supported Vehicle Classes

```python
vehicle_classes = {"car", "truck", "bus", "motorbike"}
```

Each class has its own independent counter.

---

## 🛠️ Requirements

- Python 3.8+
- ultralytics
- opencv-python
- numpy
- cvzone

Install dependencies:

```bash
pip install ultralytics opencv-python numpy cvzone
```

> **Important:**  
> `sort.py` must be present in the project directory.  
> Source: https://github.com/abewley/sort

---

## 📂 Project Structure

```
Vehicle-Counter/
│
├── Videos/
│   └── cars.mp4
│
├── Yolo-Weights/
│   └── yolov8l.pt
│
├── sort.py
├── vehicle_counter.py
└── README.md
```

---

## ▶️ Run

```bash
python vehicle_counter.py
```

Press **ESC** to exit.

---

## 📊 Output

- Bounding boxes with class and ID
- Exact counting line drawn from `limits`
- Live counters:
  - Cars
  - Trucks
  - Buses
  - Motorbikes

---

## 🚀 Possible Extensions

- Direction-based counting
- Multiple counting lines
- CSV export
- FPS monitoring
- Mouse-based line selection
- Config via YAML

---

## 📜 License

For educational and research purposes.
