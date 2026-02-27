# 🚀 Face Recognition App (FastAPI + Streamlit)

A modular, production-oriented Face Recognition system built with:

-   🔹 **FastAPI** (Backend API)
-   🔹 **Streamlit** (Interactive Frontend)
-   🔹 **face_recognition / dlib** (Face embeddings)
-   🔹 **OpenCV** (Image processing)

This project demonstrates a clean separation between business logic
(core), API layer, and UI layer --- designed to be portfolio-ready and
extensible.

------------------------------------------------------------------------

## 🧠 Project Architecture

User (Browser) ↓\
Streamlit UI\
↓\
FastAPI Backend API\
↓\
Core Face Recognition Logic\
↓\
Stored Encodings (known_faces/)

------------------------------------------------------------------------

## ✨ Features

### ✅ Enroll (Register a Person)

-   Upload image OR use camera snapshot
-   Requires exactly one face
-   Saves 128-d face embedding

### ✅ Recognize (Snapshot Mode)

-   Upload image OR camera snapshot
-   Returns name + similarity distance

### ✅ Live Video Recognition

-   Real-time MJPEG stream
-   Face detection on live webcam
-   Bounding box + name label overlay
-   Automatic DB refresh every few seconds

### ✅ People Management

-   List enrolled users
-   Delete users from database

------------------------------------------------------------------------

## 📂 Project Structure

face_app/ │ ├── app_fastapi.py ├── streamlit_app.py ├── requirements.txt
├── README.md │ └── core/ ├── settings.py ├── storage.py ├── vision.py
└── recognizer.py

------------------------------------------------------------------------

## ⚙️ Installation

### 1️⃣ Create virtual environment

``` bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 2️⃣ Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ Run the Application

### Start Backend (FastAPI)

``` bash
uvicorn app_fastapi:app --reload
```

API Docs: http://127.0.0.1:8000/docs

### Start Frontend (Streamlit)

``` bash
streamlit run streamlit_app.py
```

------------------------------------------------------------------------

## 🔬 Performance Notes

-   Uses HOG model (CPU-friendly)
-   Frame resizing for speed optimization
-   Detection runs every N frames
-   JPEG compression tuned for streaming

⚠ Currently optimized for CPU execution.

------------------------------------------------------------------------

## 🔮 Future Improvements

-   🔥 Switch to GPU acceleration (InsightFace / DeepFace with CUDA)
-   🎯 Replace dlib detector with faster alternatives (MediaPipe /
    RetinaFace)
-   📈 Add FAISS index for scalable face search
-   🌍 Deploy backend to cloud (Docker + Nginx)
-   🧩 Add authentication & user roles

------------------------------------------------------------------------

## 🏷 Tech Stack

-   Python 3.10+
-   FastAPI
-   Streamlit
-   OpenCV
-   face_recognition (dlib)

------------------------------------------------------------------------

## 📌 Why This Project?

This project was built to:

-   Improve understanding of computer vision pipelines
-   Practice API + UI separation
-   Learn real-time streaming over HTTP (MJPEG)
-   Prepare a production-style portfolio project

------------------------------------------------------------------------

## 📎 LinkedIn Description (Copy & Paste)

I built a modular Face Recognition system using FastAPI and Streamlit.\
The project supports live video recognition with bounding boxes and name
labels,\
snapshot-based recognition, and dynamic face enrollment.

It demonstrates clean architecture separation between backend, core
logic, and frontend UI.

Tech: Python, FastAPI, Streamlit, OpenCV, dlib.

------------------------------------------------------------------------

## 📄 License

MIT License

------------------------------------------------------------------------

⭐ If you find this useful, feel free to star the repository!
