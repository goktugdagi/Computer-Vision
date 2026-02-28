# Run:
#   python -m uvicorn app_fastapi:app --reload

import time
from typing import Generator, List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from core.camera import CameraManager
from core.engine_insightface import InsightFaceEngine
from core.matching import match_one
from core.settings import SETTINGS
from core.storage import delete_encoding, load_database, list_people, sanitize_name, save_encoding

app = FastAPI(title="Face App V3 (InsightFace/ArcFace)", version="3.0.0")

camera = CameraManager(index=SETTINGS.camera_index)
engine = InsightFaceEngine()


@app.on_event("startup")
def _startup():
    camera.start()


@app.on_event("shutdown")
def _shutdown():
    camera.stop()


def _encode_jpeg(bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode JPEG.")
    return buf.tobytes()


def _draw_label(bgr: np.ndarray, bbox: Tuple[int, int, int, int], label: str) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    y0 = max(0, y1 - th - 8)
    cv2.rectangle(bgr, (x1, y0), (x1 + tw + 8, y0 + th + 8), (0, 255, 0), -1)
    cv2.putText(bgr, label, (x1 + 4, y0 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def _mjpeg_generator(mode: str) -> Generator[bytes, None, None]:
    """
    mode:
      - "preview": raw frames
      - "live": recognition overlay
    """
    last_db_load = 0.0
    known_names: List[str] = []
    known_embs = np.empty((0, 512), dtype=np.float32)

    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.02)
            continue

        bgr = frame

        if mode == "live":
            now = time.time()
            if now - last_db_load > 1.0:
                known_names, known_embs = load_database(SETTINGS.data_dir)
                last_db_load = now

            faces = engine.detect_and_embed(bgr)
            for f in faces:
                name, dist = match_one(f.embedding, known_names, known_embs, SETTINGS.match_threshold)
                label = f"{name} ({dist:.2f})" if name != "Unknown" else "Unknown"
                _draw_label(bgr, f.bbox, label)

        jpg = _encode_jpeg(bgr)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        )


@app.get("/health")
def health():
    return {"status": "ok", "engine_backend": engine.backend}


@app.get("/people")
def get_people():
    return {"people": list_people(SETTINGS.data_dir)}


@app.delete("/people/{name}")
def remove_person(name: str):
    name = sanitize_name(name)
    ok = delete_encoding(SETTINGS.data_dir, name)
    if not ok:
        raise HTTPException(status_code=404, detail="Person not found.")
    return {"deleted": name}


@app.get("/preview_feed")
def preview_feed():
    return StreamingResponse(
        _mjpeg_generator(mode="preview"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/live_feed")
def live_feed():
    return StreamingResponse(
        _mjpeg_generator(mode="live"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/snapshot")
def snapshot():
    frame = camera.snapshot()
    jpg = _encode_jpeg(frame)
    return Response(content=jpg, media_type="image/jpeg")


@app.post("/enroll_file")
async def enroll_file(name: str, image: UploadFile = File(...)):
    if not name.strip():
        raise HTTPException(status_code=400, detail="Query param 'name' is required.")

    data = await image.read()
    npbuf = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    faces = engine.detect_and_embed(bgr)
    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="No face found.")
    if len(faces) > 1:
        raise HTTPException(status_code=400, detail="Multiple faces found. Use a single-person image.")

    safe = sanitize_name(name)
    save_encoding(SETTINGS.data_dir, safe, faces[0].embedding, overwrite=True)
    return {"saved": safe, "engine_backend": engine.backend}


# ✅ Recognize from an uploaded image.
# Returns:
#   - annotated image (JPEG) with boxes + labels
#   - plus JSON-like metadata in headers is not ideal; so Streamlit will call /recognize_json too.
@app.post("/recognize_file")
async def recognize_file(image: UploadFile = File(...)):
    data = await image.read()
    npbuf = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    known_names, known_embs = load_database(SETTINGS.data_dir)

    results = []
    faces = engine.detect_and_embed(bgr)
    for f in faces:
        name, dist = match_one(f.embedding, known_names, known_embs, SETTINGS.match_threshold)
        label = f"{name} ({dist:.2f})" if name != "Unknown" else "Unknown"
        _draw_label(bgr, f.bbox, label)
        results.append({"bbox": f.bbox, "name": name, "distance": float(dist), "det_score": float(f.det_score)})

    # We return the annotated image bytes.
    # Streamlit will show it, and we will also expose JSON via another endpoint:
    # (keeps things simple)
    jpg = _encode_jpeg(bgr, quality=90)

    # Store JSON in a custom header is not recommended; so we don't.
    # Use /recognize_json endpoint below.
    return Response(content=jpg, media_type="image/jpeg")


@app.post("/recognize_json")
async def recognize_json(image: UploadFile = File(...)):
    data = await image.read()
    npbuf = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    known_names, known_embs = load_database(SETTINGS.data_dir)

    out = []
    faces = engine.detect_and_embed(bgr)
    for f in faces:
        name, dist = match_one(f.embedding, known_names, known_embs, SETTINGS.match_threshold)
        out.append({"bbox": f.bbox, "name": name, "distance": float(dist), "det_score": float(f.det_score)})

    return {"results": out, "engine_backend": engine.backend, "threshold": SETTINGS.match_threshold}