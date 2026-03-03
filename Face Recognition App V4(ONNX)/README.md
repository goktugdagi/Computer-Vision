# 🚀 Face App V4 --- SCRFD + ArcFace (ONNX) + FastAPI + Streamlit

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange)
![InsightFace](https://img.shields.io/badge/InsightFace-SCRFD%20%2B%20ArcFace-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

------------------------------------------------------------------------

## 📌 Overview

Face App V4 is a production-ready real-time face recognition system
built using:

-   SCRFD (Face Detection)
-   ArcFace (Face Embedding)
-   ONNX Runtime
-   FastAPI (Backend API)
-   Streamlit (Frontend UI)

This version removes TensorFlow / DeepFace and runs entirely on
ONNX-based inference (CPU or GPU).

------------------------------------------------------------------------

## 🏗 Architecture

Camera (OpenCV) ↓ SCRFD Detector (ONNX) ↓ ArcFace Embedder (ONNX) ↓ L2
Normalization ↓ Cosine Similarity Matching ↓ Disk-Based Embedding Store
(JSON + .npy)

------------------------------------------------------------------------

## ⚙️ Features

-   Real-time live recognition
-   Face enrollment (camera or upload)
-   Multi-sample per person
-   Disk-based embedding storage
-   Cosine similarity matching
-   Throttled live inference
-   CPU and GPU support
-   Clean Streamlit UI
-   REST API endpoints

------------------------------------------------------------------------

## 📂 Project Structure

face_app_v4/ ├─ app_fastapi.py ├─ streamlit_app.py ├─ requirements.txt
└─ core/ ├─ camera.py ├─ settings.py ├─ storage.py ├─ matching.py ├─
vision.py └─ engine_insightface_v4.py

------------------------------------------------------------------------

## 🧠 Technology Stack

  Component          Technology
  ------------------ ---------------------
  Face Detection     SCRFD
  Face Embedding     ArcFace
  Inference Engine   ONNX Runtime
  Backend            FastAPI
  Frontend           Streamlit
  Storage            JSON + NumPy (.npy)
  Similarity         Cosine Distance
  Language           Python 3.10+

------------------------------------------------------------------------

## 🖥 Installation (Windows)

python -m venv venv-faceappV4
venv-faceappV4`\Scripts`{=tex}`\activate`{=tex}

pip install -r requirements.txt

------------------------------------------------------------------------

## ▶️ Run Backend

python -m uvicorn app_fastapi:app --reload

------------------------------------------------------------------------

## 🎛 Run Frontend

venv-faceappV4`\Scripts`{=tex}`\activate`{=tex} streamlit run
streamlit_app.py

------------------------------------------------------------------------

## ⚡ GPU Acceleration

pip uninstall onnxruntime pip install onnxruntime-gpu

Update settings.py:

ctx_id = 0

------------------------------------------------------------------------

## 📊 Performance Benchmark (Approximate)

  Mode   Detection + Embedding
  ------ -----------------------
  CPU    \~150--300 ms
  GPU    \~20--60 ms

------------------------------------------------------------------------

## 🔐 API Endpoints

  Endpoint              Method   Description
  --------------------- -------- -------------------------
  /                     GET      System info
  /preview              GET      Camera preview
  /live                 GET      Live recognition stream
  /api/snapshot         GET      Capture frame
  /api/people           GET      List enrolled
  /api/enroll_file      POST     Enroll face
  /api/recognize_file   POST     Recognize image
  /api/delete_person    POST     Delete person

------------------------------------------------------------------------

## 🧮 Matching Logic

-   All embeddings are L2 normalized
-   Matching uses Cosine Distance
-   Threshold configurable via match_threshold in settings.py

------------------------------------------------------------------------

## 📦 Storage Format

known_faces_v4/ index.json person\_\_0.npy person\_\_1.npy

------------------------------------------------------------------------

## 📄 License

MIT License

------------------------------------------------------------------------

Built as part of an advanced face recognition engineering roadmap (V1 →
V5).
