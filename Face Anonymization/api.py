# python -m uvicorn api:app --reload
###########################################################

import os                     # File system operations
import uuid                     # Unique ID generation
import tempfile                 # Temporary file handling
import shutil                   # Shell utilities (PATH lookup)
import subprocess               # External process execution
from typing import Optional     # Optional type hints

import cv2                      # OpenCV for video/image processing
import numpy as np              # NumPy for array handling
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, Response

from facial_landmarks import FaceLandmarks
from core_blur import blur_faces


# Initialize FastAPI application
app = FastAPI(title="Face Blur API", version="1.0.0")

# FaceMesh instances for video and image processing
FL_VIDEO = FaceLandmarks(max_num_faces=10, static_image_mode=False)
FL_IMAGE = FaceLandmarks(max_num_faces=10, static_image_mode=True)


@app.get("/health")
def health():
    # Simple health check endpoint
    return {"status": "ok"}
    


@app.post("/blur-image")
async def blur_image(
    file: UploadFile = File(...),                 # Uploaded image file
    blur_ksize: int = Query(27, ge=1, le=99),     # Blur kernel size
):
    # Read uploaded file content
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")

    # Decode image from bytes
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image or unsupported format.")

    # Detect facial landmarks
    all_faces = FL_IMAGE.get_facial_landmarks(img)

    # Apply face blur
    out = blur_faces(img, all_faces, blur_ksize=blur_ksize)
    if out is None:
        raise HTTPException(status_code=500, detail="Processing failed.")

    # Determine output image format
    ext = (os.path.splitext(file.filename or "")[1] or ".jpg").lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".jpg"

    # Encode output image
    encode_ok, buf = cv2.imencode(ext, out)
    if not encode_ok:
        raise HTTPException(status_code=500, detail="Failed to encode output image.")
    
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }[ext]


    # Return image response
    return Response(content=buf.tobytes(), media_type=media_type)

def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise HTTPException(
            status_code=400,
            detail=(
                "FFmpeg not found in PATH. Browser playback requires H.264 MP4. "
                "Install FFmpeg and add it to PATH, then restart the API."
            ),
        )
    return ffmpeg


def _transcode_h264(ffmpeg: str, in_mp4_path: str) -> str:
    out_path = os.path.join(tempfile.gettempdir(), f"h264_{uuid.uuid4().hex}.mp4")

    cmd = [
        ffmpeg, "-y",
        "-i", in_mp4_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "baseline",
        "-level", "3.0",
        "-movflags", "+faststart",
        "-an",
        out_path,
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return out_path
    except subprocess.CalledProcessError:
        raise HTTPException(
            status_code=500,
            detail="FFmpeg transcoding failed. Check FFmpeg installation and input video integrity.",
        )


@app.post("/blur-video")
async def blur_video(
    file: UploadFile = File(...),
    blur_ksize: int = Query(27, ge=1, le=99),
    detect_every_n: int = Query(1, ge=1, le=30),
    max_frames: Optional[int] = Query(None, ge=1),
):
    ffmpeg = _require_ffmpeg()

    suffix = os.path.splitext(file.filename or "")[1]
    if suffix.lower() not in [".mp4", ".mov", ".avi", ".mkv"]:
        suffix = ".mp4"

    in_path = os.path.join(tempfile.gettempdir(), f"in_{uuid.uuid4().hex}{suffix}")
    out_raw_path = os.path.join(tempfile.gettempdir(), f"out_{uuid.uuid4().hex}.mp4")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")

    with open(in_path, "wb") as f:
        f.write(content)

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Video could not be opened.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 25.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        cap.release()
        raise HTTPException(status_code=400, detail="Invalid video dimensions.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_raw_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise HTTPException(status_code=500, detail="VideoWriter could not be opened.")

    processed = 0
    cached_faces = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run FaceMesh every N frames; reuse cached landmarks in between
            if processed % detect_every_n == 0:
                cached_faces = FL_VIDEO.get_facial_landmarks(frame)

            out = blur_faces(frame, cached_faces, blur_ksize=blur_ksize)
            writer.write(out)

            processed += 1
            if max_frames is not None and processed >= max_frames:
                break
    finally:
        cap.release()
        writer.release()
        try:
            os.remove(in_path)
        except OSError:
            pass

    out_h264_path = _transcode_h264(ffmpeg, out_raw_path)

    try:
        os.remove(out_raw_path)
    except OSError:
        pass

    return FileResponse(out_h264_path, media_type="video/mp4", filename="blurred.mp4")
