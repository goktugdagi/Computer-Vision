
# 🎯 Face App V2 — Real-Time Face Recognition

A production-style face recognition system built with:

- 🚀 FastAPI (backend API + camera control)
- 🎥 OpenCV (live video processing + tracking)
- 🧠 face_recognition (face embeddings)
- 🖥 Streamlit (frontend UI)
- 📦 MJPEG streaming (real-time browser video)

---

## 🔥 Key Features

✅ Backend owns the camera (no multi-process conflicts)  
✅ Real-time MJPEG streaming via `/video_feed`  
✅ IoU-based tracking reconciliation (reduced flicker)  
✅ Snapshot-based enrollment (live preview + capture)  
✅ Image upload enrollment  
✅ People list & delete  
✅ Cache-safe live stream refresh  

---

## 🏗 Architecture

Streamlit UI  
⬇  
FastAPI Backend  
⬇  
OpenCV Camera  
⬇  
Face Detection + Embedding  
⬇  
Tracking + Recognition  

Camera is accessed **only once** (backend).

---

## 🚀 How To Run

### 1️⃣ Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Start backend

```bash
uvicorn app_fastapi:app --reload
```

### 4️⃣ Start frontend

```bash
streamlit run streamlit_app.py
```

Open in browser:
```
http://localhost:8501
```

---

## 📂 Project Structure

```
face_app_v2/
│
├── app_fastapi.py
├── streamlit_app.py
├── core/
│   ├── settings.py
│   ├── vision.py
│   ├── recognizer.py
│   ├── camera.py
│   ├── tracking.py
│   └── storage.py
│
├── known_faces/
├── requirements.txt
└── README.md
```

---

## 🧠 Technical Highlights

- Detection runs periodically
- Tracking (CSRT/KCF) between detections
- IoU matching prevents label flickering
- Snapshot endpoint for controlled enrollment
- MJPEG stream cache-busting for stable UI

---

## ⚡ Future Roadmap

- Replace face_recognition with ArcFace (InsightFace)
- FAISS integration for scalable search
- RetinaFace / YOLO-face detector
- Docker packaging
- Authentication layer

---

## 👨‍💻 Author

Built as part of an advanced Computer Vision learning roadmap.
