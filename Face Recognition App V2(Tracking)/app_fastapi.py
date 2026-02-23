# python -m uvicorn app_fastapi:app --reload
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, Response
import cv2
import time

from core.settings import SETTINGS
from core.camera import CameraManager
from core.storage import load_database, save_encoding, delete_encoding, list_people, sanitize_name
from core.vision import decode_image_bytes_to_bgr, bgr_to_rgb, detect_faces, encode_faces
from core.recognizer import recognize_one
from core.tracking import update_tracks, reconcile_tracks


app = FastAPI(title="Face Recognition API (V2 Tracking)", version="2.1.0")

CAM = CameraManager(camera_index=SETTINGS.camera_index, target_fps=SETTINGS.camera_target_fps)


@app.on_event("startup")
def on_startup():
    CAM.start()


@app.on_event("shutdown")
def on_shutdown():
    CAM.stop()


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


@app.get("/snapshot")
def snapshot():
    jpg = CAM.get_jpeg(quality=85)
    if jpg is None:
        raise HTTPException(status_code=503, detail="Camera frame not available yet.")
    return Response(content=jpg, media_type="image/jpeg")


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
    locations = detect_faces(rgb, model=SETTINGS.face_model)

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
    locations = detect_faces(rgb, model=SETTINGS.face_model)

    if len(locations) == 0:
        return {"faces": []}

    encs = encode_faces(rgb, locations)
    known_names, known_encs = load_database(SETTINGS.data_dir)

    faces = []
    for enc in encs:
        match = recognize_one(known_names, known_encs, enc, tolerance=SETTINGS.tolerance)
        if match:
            faces.append({"name": match.name, "distance": match.distance})
        else:
            faces.append({"name": "Unknown", "distance": None})

    return {"faces": faces}


def _draw_box_label(frame, bbox_xywh, label: str):
    x, y, w, h = bbox_xywh
    x2, y2 = x + w, y + h

    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    y1 = max(y - th - baseline - 6, 0)
    y2b = y
    x1 = x
    x2b = x + tw + 10

    cv2.rectangle(frame, (x1, y1), (x2b, y2b), (0, 255, 0), cv2.FILLED)
    cv2.putText(frame, label, (x + 5, y - 6), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def _locations_to_xywh(locations, scale_back: float):
    bboxes = []
    for (t, r, b, l) in locations:
        top = int(t * scale_back)
        right = int(r * scale_back)
        bottom = int(b * scale_back)
        left = int(l * scale_back)

        w = max(0, right - left)
        h = max(0, bottom - top)
        bboxes.append((left, top, w, h))
    return bboxes


def generate_mjpeg_stream():
    known_names, known_encs = load_database(SETTINGS.data_dir)

    frame_idx = 0
    tracks = []

    while True:
        frame = CAM.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        frame_idx += 1

        if frame_idx % 150 == 0:
            known_names, known_encs = load_database(SETTINGS.data_dir)

        # heavy detect+recognize every N frames
        if frame_idx % SETTINGS.detect_every_n_frames == 0:
            small = cv2.resize(frame, (0, 0), fx=SETTINGS.resize_scale, fy=SETTINGS.resize_scale)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            locations_small = detect_faces(rgb_small, model=SETTINGS.face_model)

            scale_back = 1.0 / SETTINGS.resize_scale
            bboxes = _locations_to_xywh(locations_small, scale_back)

            filtered_locations_small = []
            filtered_bboxes = []
            for loc, bb in zip(locations_small, bboxes):
                x, y, w, h = bb
                if w >= SETTINGS.min_face_size and h >= SETTINGS.min_face_size:
                    filtered_locations_small.append(loc)
                    filtered_bboxes.append(bb)

            encs = encode_faces(rgb_small, filtered_locations_small)
            labels = []

            for enc in encs:
                m = recognize_one(known_names, known_encs, enc, tolerance=SETTINGS.tolerance)
                labels.append(m.name if m else "Unknown")

            tracks = reconcile_tracks(frame, tracks, labels, filtered_bboxes, SETTINGS.tracker_type, iou_threshold=0.25)

        else:
            tracks = update_tracks(frame, tracks)

        for t in tracks:
            _draw_box_label(frame, t.bbox, t.label)

        ok2, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), SETTINGS.jpeg_quality])
        if ok2:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )

        if SETTINGS.stream_sleep_sec > 0:
            time.sleep(SETTINGS.stream_sleep_sec)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )