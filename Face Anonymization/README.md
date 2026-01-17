# BlurGuard Studio 🛡️

**BlurGuard Studio** is a **privacy-first computer vision application** for real-time and batch **face anonymization**.  
It provides a **modern, colorful Streamlit interface** combined with a **FastAPI backend** to blur faces in:

- 🎥 **Live webcam streams (real time)**
- 📁 **Uploaded video files (batch processing)**

The project is designed as a **portfolio-quality, production-style prototype** with a strong focus on **usability, privacy, and clean architecture**.

---

## 🎨 User Interface Highlights

BlurGuard Studio is not a basic Streamlit demo. The UI includes:

- Gradient **hero header** with product-style branding  
- Colorful **cards and panels** for inputs, outputs, and settings  
- Clear **Control Panel sidebar**  
- Tab-based navigation:
  - **Webcam Live Blur**
  - **Video Upload**
- Visual backend **connection status indicator**

The goal is to feel like a real product, not a notebook wrapper.

---

## 🧠 How It Works (High Level)

```
Browser (Streamlit UI)
        │
        ▼
FastAPI Backend
        │
        ▼
MediaPipe FaceMesh → Face Mask → Adaptive Blur → MP4 Output
```

---

## 📁 Project Structure

```
blurguard-studio/
│
├── app_streamlit.py        # Colorful Streamlit frontend (UI + live preview)
├── api.py                  # FastAPI backend (video processing)
├── facial_landmarks.py     # Face detection & landmark extraction (MediaPipe)
├── core_blur.py             # Mask generation & adaptive blur logic
├── requirements.txt
└── README.md
```

---

## ✨ Core Features

- ✅ Multi-face support
- ✅ Adaptive blur strength (scales with face size)
- ✅ Real-time webcam anonymization
- ✅ Video upload & batch processing
- ✅ HTML5 / browser-compatible MP4 output
- ✅ Clear separation of frontend & backend
- ✅ Portfolio-ready UI and code structure

---

## 🧩 Core Components Explained

### `app_streamlit.py`
- Modern, colorful **frontend**
- Webcam live blur using **streamlit-webrtc**
- Video upload, preview, processing, and download
- Sidebar controls for blur strength and performance tuning

### `api.py`
- **FastAPI backend**
- `/health` endpoint for connection checks
- `/blur-video` endpoint for processing uploaded videos
- Uses FFmpeg to ensure browser-compatible H.264 output

### `facial_landmarks.py`
- Wrapper around **MediaPipe FaceMesh**
- Extracts **468 landmarks per face**
- Supports multiple faces per frame

### `core_blur.py`
- Builds face masks using convex hulls
- Expands masks slightly for better edge coverage
- Applies **adaptive Gaussian blur** based on face size

---

## 🖥️ Requirements

### Python Version
```
Python 3.9 – 3.11 (recommended)
```

### Python Dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```txt
fastapi
uvicorn[standard]
python-multipart

streamlit
streamlit-webrtc
av
requests

opencv-python
numpy
mediapipe
```

---

## ⚙️ System Dependency (Required)

### FFmpeg
FFmpeg **must** be installed and accessible via PATH.

- Required for browser-playable MP4 (H.264) output
- Verify installation:

```bash
ffmpeg -version
```

---

## 🚀 Running the Application

### 1️⃣ Start the FastAPI Backend

```bash
uvicorn api:app --reload
```

Check health:

```
http://localhost:8000/health
```

Expected response:
```json
{"status":"ok"}
```

---

### 2️⃣ Start the Streamlit Frontend

```bash
streamlit run app_streamlit.py
```

Open in browser:
```
http://localhost:8501
```

---

## 🧪 Usage Guide

### Webcam Live Blur
1. Open **Webcam Live Blur** tab
2. Grant camera permissions
3. Faces are blurred in real time

### Video Upload
1. Open **Video Upload** tab
2. Upload a supported video file
3. Adjust blur and performance settings
4. Click **Process Video**
5. Preview and download the blurred video

---

## 🎛️ Configurable Controls

- **Blur Strength** – Base kernel size (adaptive scaling applied)
- **Max Faces** – Webcam face limit
- **Detect Every N Frames** – Performance tuning for uploads
- **Frame Limit (Testing)** – Useful for debugging

---

## ⚡ Performance Notes

- MediaPipe FaceMesh is accurate but CPU-intensive
- Designed for clarity and correctness over raw speed
- Future optimizations can include:
  - GPU-based face detection
  - Frame skipping
  - ROI-only blur
  - NVENC video encoding

---

## ⚖️ Ethical Use

This project is intended **only** for:

- Privacy protection
- Anonymization
- Compliance with data protection regulations (GDPR, KVKK, etc.)

🚫 Do **not** use for surveillance, tracking, or identity inference.

---

## 🧭 Roadmap / Future Ideas

- GPU acceleration
- Object blurring (license plates, screens)
- Preset modes (Light / Medium / Strong blur)
- Progress bar during processing
- Docker & deployment support

---

## 📌 License

MIT License (recommended)

---

## 👤 Author

Built as a **computer vision & privacy-focused portfolio project**.

If you want to evolve this into a production system or research prototype, the architecture is designed to scale.
