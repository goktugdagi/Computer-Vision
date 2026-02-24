# python -m uvicorn app_fastapi:app --reload
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse
import cv2
import time
import numpy as np  # <-- NEW

from core.settings import SETTINGS
from core.storage import load_database, save_encoding, delete_encoding, list_people, sanitize_name
from core.vision import decode_image_bytes_to_bgr, bgr_to_rgb, detect_faces, encode_faces
from core.recognizer import recognize_one

app = FastAPI(title="Face Recognition API", version="1.1.1")


# ----------------------------
# Existing endpoints (people/enroll/recognize)
# ----------------------------
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


@app.post("/enroll_file")
async def enroll_file(name: str, image: UploadFile = File(...)):
    if not name.strip():
        raise HTTPException(status_code=400, detail="Query param 'name' is required.")

    data = await image.read()
    try:
        bgr = decode_image_bytes_to_bgr(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    rgb = bgr_to_rgb(bgr)
    locations = detect_faces(rgb, model=SETTINGS.model)

    if len(locations) == 0:
        raise HTTPException(status_code=400, detail="No face found.")
    if len(locations) > 1:
        raise HTTPException(status_code=400, detail="Multiple faces found. Use a single-person image.")

    enc = encode_faces(rgb, locations)[0]
    safe_name = sanitize_name(name)
    save_encoding(SETTINGS.data_dir, safe_name, enc, overwrite=True)

    return {"saved": safe_name}


@app.post("/recognize_file")
async def recognize_file(image: UploadFile = File(...)):
    data = await image.read()
    try:
        bgr = decode_image_bytes_to_bgr(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    rgb = bgr_to_rgb(bgr)
    locations = detect_faces(rgb, model=SETTINGS.model)

    if len(locations) == 0:
        return {"faces": []}

    encs = encode_faces(rgb, locations)
    known_names, known_encs = load_database(SETTINGS.data_dir)

    results = []
    for enc in encs:
        match = recognize_one(known_names, known_encs, enc, tolerance=SETTINGS.tolerance)
        if match:
            results.append({"name": match.name, "distance": match.distance})
        else:
            results.append({"name": "Unknown", "distance": None})

    return {"faces": results}


# ----------------------------
# Live video stream with boxes + labels
# ----------------------------
def draw_label_box(frame_bgr, top, right, bottom, left, label: str):
    cv2.rectangle(frame_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    y1 = max(top - th - baseline - 6, 0)
    y2 = top
    x1 = left
    x2 = left + tw + 10

    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), cv2.FILLED)
    cv2.putText(frame_bgr, label, (left + 5, top - 6), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def _error_frame_jpeg(msg: str) -> bytes:
    """
    Create a visible JPEG frame that shows an error message.
    This prevents the stream from appearing 'blank'.
    """
    h, w = 480, 800
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(frame, "VIDEO FEED ERROR", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
    y = 150
    for line in msg.split("\n"):
        cv2.putText(frame, line[:80], (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        y += 40

    ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return jpeg.tobytes() if ok else b""


def generate_mjpeg_stream(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows
    if not cap.isOpened():
        # IMPORTANT FIX: Don't silently return. Emit an error frame repeatedly.
        msg = (
            f"Camera open failed (cam={camera_index}).\n"
            "Close apps using the camera (Chrome camera_input/Teams/Discord).\n"
            "Try another index: /video_feed?cam=1 or cam=2."
        )
        err_jpeg = _error_frame_jpeg(msg)
        while True:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + err_jpeg + b"\r\n")
            time.sleep(1.0)

    known_names, known_encs = load_database(SETTINGS.data_dir)

    resize_scale = 0.25
    detect_every_n_frames = 8

    frame_idx = 0
    last_locations_full = []
    last_labels = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                err_jpeg = _error_frame_jpeg("Camera read failed.\nTry restarting backend.")
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + err_jpeg + b"\r\n")
                time.sleep(0.5)
                continue

            frame_idx += 1

            # refresh DB periodically so new enrolls appear
            if frame_idx % 90 == 0:
                known_names, known_encs = load_database(SETTINGS.data_dir)

            small = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            if frame_idx % detect_every_n_frames == 0:
                loc_small = detect_faces(rgb_small, model=SETTINGS.model)
                enc_small = encode_faces(rgb_small, loc_small)

                inv = 1.0 / resize_scale
                last_locations_full = []
                last_labels = []

                for (t, r, b, l), enc in zip(loc_small, enc_small):
                    top = int(t * inv)
                    right = int(r * inv)
                    bottom = int(b * inv)
                    left = int(l * inv)

                    match = recognize_one(known_names, known_encs, enc, tolerance=SETTINGS.tolerance)
                    label = match.name if match else "Unknown"

                    last_locations_full.append((top, right, bottom, left))
                    last_labels.append(label)

            for (top, right, bottom, left), label in zip(last_locations_full, last_labels):
                draw_label_box(frame, top, right, bottom, left, label)

            ok2, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            if not ok2:
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

            time.sleep(0.02)

    finally:
        cap.release()


@app.get("/video_feed")
def video_feed(cam: int = Query(0, ge=0, le=10)):
    """
    MJPEG stream endpoint for live video with face boxes + labels.
    cam: camera index (0/1/2...)
    """
    return StreamingResponse(
        generate_mjpeg_stream(camera_index=cam),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )