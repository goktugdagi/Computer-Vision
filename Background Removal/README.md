# Real-Time Camera Background Removal

A demo application that performs **real-time person segmentation**, **background removal**, and **background replacement** using a webcam stream.  
Built with **cvzone.SelfiSegmentation (MediaPipe backend)** and **Streamlit** for an interactive and tunable UI.

## Features

- Real-time human segmentation and background removal
- Background replacement from:
  - Local `./images` folder
  - Streamlit drag-and-drop upload
  - Direct image URL input
- Optional **Blur Background Mode** toggle for more stable segmentation
- UI parameter controls:
  - `cutThreshold` (segmentation confidence threshold)
  - FPS limit for UI refresh pacing
  - Padding value for letterbox blank areas
  - Camera index selection
  - Start/Stop webcam stream

## Installation & Setup (Optimized for Windows 10/11 + VS Code)

1. **Create and activate a virtual environment**
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install required dependencies**
   ```powershell
   pip install opencv-python numpy streamlit cvzone mediapipe
   ```

3. **Verify project structure**
   ```
   repo-root/
   │── app.py
   │── images/        # Put your background images here (jpg/png/webp)
   │── README.md      # This file
   │── requirements.txt (optional)
   ```

4. **Run the Streamlit app**
   ```powershell
   streamlit run app.py
   ```

## Notes

- The webcam runs at its **native resolution** (no forced size set).
- All background images are resized to match the first captured frame **without cropping or stretching**.
- Camera cleanup is handled safely using `st.session_state.cap.release()` inside Streamlit logic.

