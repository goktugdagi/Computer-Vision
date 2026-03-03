# Run:
#   python -m uvicorn app_fastapi:app --reload

import time
from typing import Generator

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse

from core.settings import SETTINGS, DB_PATH
from core.camera import Camera
from core.storage import FaceStore
from core.matching import find_best_match
from core.vision import encode_jpg, draw_box_label
from core.engine_insightface_v4 import InsightFaceEngineV4

app = FastAPI(title="Face App V4 (SCRFD+ArcFace ONNX)")

camera: Camera | None = None
engine = InsightFaceEngineV4()
store = FaceStore(DB_PATH)


def mjpeg(frames: Generator[np.ndarray, None, None]) -> Generator[bytes, None, None]:
    for frame in frames:
        jpg = encode_jpg(frame, quality=80)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")


@app.on_event("startup")
def on_startup():
    global camera
    camera = Camera(SETTINGS.camera_index, (SETTINGS.camera_width, SETTINGS.camera_height)).start()


@app.on_event("shutdown")
def on_shutdown():
    global camera
    if camera:
        camera.stop()


@app.get("/")
def root():
    return {
        "status": "ok",
        "version": "V4",
        "engine": engine.backend,
        "people": store.count(),
        "ctx_id": SETTINGS.ctx_id,
        "det_size": [SETTINGS.det_size_w, SETTINGS.det_size_h],
        "det_thresh": SETTINGS.det_thresh,
        "match_threshold": SETTINGS.match_threshold,
    }


@app.get("/preview")
def preview():
    def gen():
        while True:
            frame = camera.read() if camera else None
            if frame is None:
                time.sleep(0.01)
                continue
            yield frame
            time.sleep(0.03)

    return StreamingResponse(mjpeg(gen()), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/live")
def live():
    """
    Live recognition with throttling to keep it stable.
    """
    def gen():
        infer_every_sec = 0.5
        last_infer_t = 0.0
        cached = []  # list[(bbox,label)]
        while True:
            frame = camera.read() if camera else None
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            if (now - last_infer_t) >= infer_every_sec:
                last_infer_t = now
                try:
                    db = store.load_all()
                    faces = engine.detect_and_embed(frame)

                    new_cache = []
                    for f in faces:
                        if not db:
                            label = "Unknown"
                        else:
                            name, dist = find_best_match(f.embedding, db)
                            label = name if (name is not None and dist <= SETTINGS.match_threshold) else "Unknown"
                        new_cache.append((f.bbox, label))
                    cached = new_cache
                except Exception:
                    cached = []

            for bbox, label in cached:
                draw_box_label(frame, bbox, label)

            yield frame
            time.sleep(0.03)

    return StreamingResponse(mjpeg(gen()), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/snapshot")
def snapshot():
    frame = camera.read() if camera else None
    if frame is None:
        return JSONResponse({"ok": False, "error": "No camera frame"}, status_code=500)
    return StreamingResponse(iter([encode_jpg(frame, quality=90)]), media_type="image/jpeg")


@app.get("/api/people")
def people():
    return {"ok": True, "people": store.list_people()}


@app.post("/api/delete_person")
def delete_person(name: str = Form(...)):
    ok = store.delete_person(name)
    return {"ok": ok}


@app.post("/api/enroll_file")
async def enroll_file(name: str = Form(...), file: UploadFile = File(...)):
    try:
        content = await file.read()
        arr = np.frombuffer(content, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return JSONResponse({"ok": False, "error": "Could not decode image"}, status_code=400)

        faces = engine.detect_and_embed(bgr)
        if not faces:
            return JSONResponse({"ok": False, "error": "No face detected"}, status_code=400)

        def area(bb):
            x1, y1, x2, y2 = bb
            return max(0, x2 - x1) * max(0, y2 - y1)

        best = sorted(faces, key=lambda f: area(f.bbox), reverse=True)[0]
        sample_id = store.add_embedding(name, best.embedding)

        return {"ok": True, "name": name, "sample_id": sample_id, "bbox": best.bbox, "det_score": best.det_score}

    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Enroll failed: {repr(e)}"}, status_code=500)


@app.post("/api/recognize_file")
async def recognize_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        arr = np.frombuffer(content, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return JSONResponse({"ok": False, "error": "Could not decode image"}, status_code=400)

        db = store.load_all()
        faces = engine.detect_and_embed(bgr)

        for f in faces:
            if not db:
                label = "Unknown"
            else:
                name, dist = find_best_match(f.embedding, db)
                label = name if (name is not None and dist <= SETTINGS.match_threshold) else "Unknown"
            draw_box_label(bgr, f.bbox, label)

        jpg = encode_jpg(bgr, quality=90)
        return StreamingResponse(iter([jpg]), media_type="image/jpeg")

    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Recognize failed: {repr(e)}"}, status_code=500)