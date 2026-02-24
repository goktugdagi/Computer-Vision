import time
import requests
import streamlit as st
import cv2
import numpy as np

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

from core.settings import SETTINGS
from core.storage import load_database
from core.recognizer import recognize_one

import face_recognition  # face_locations + encodings


DEFAULT_API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Face App", layout="wide")
st.title("Face App — Streamlit UI + FastAPI Backend")


# ----------------------------
# API helpers (backend people/enroll/delete still used)
# ----------------------------
def api_get_people(api_base: str) -> list[str]:
    r = requests.get(f"{api_base}/people", timeout=10)
    r.raise_for_status()
    return r.json().get("people", [])


def api_delete_person(api_base: str, name: str) -> dict:
    r = requests.delete(f"{api_base}/people/{name}", timeout=30)
    r.raise_for_status()
    return r.json()


def api_enroll_file(api_base: str, name: str, image_bytes: bytes, filename: str) -> dict:
    files = {"image": (filename, image_bytes, "application/octet-stream")}
    r = requests.post(f"{api_base}/enroll_file", params={"name": name}, files=files, timeout=60)
    r.raise_for_status()
    return r.json()


def api_recognize_file(api_base: str, image_bytes: bytes, filename: str) -> dict:
    files = {"image": (filename, image_bytes, "application/octet-stream")}
    r = requests.post(f"{api_base}/recognize_file", files=files, timeout=60)
    r.raise_for_status()
    return r.json()


def extract_http_error_detail(err: requests.HTTPError) -> str:
    try:
        return err.response.json().get("detail", str(err))
    except Exception:
        return str(err)


# ----------------------------
# UI helper (camera snapshot for Enroll/Recognize)
# ----------------------------
def pick_image_bytes(label_upload: str):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Option A — Camera snapshot")
        st.caption("Use http://localhost:8501 and allow camera in browser.")
        cam = st.camera_input("Take Photo", key="enroll_camera")
        if cam is not None:
            return cam.getvalue(), "camera.jpg"

    with col2:
        st.subheader("Option B — Upload")
        up = st.file_uploader(label_upload, type=["jpg", "jpeg", "png", "bmp", "webp"])
        if up is not None:
            return up.read(), up.name

    return None, None


# ----------------------------
# LIVE: Browser camera (WebRTC) + local recognition (loads same DB as backend)
# ----------------------------
def _draw_box_label(bgr, top, right, bottom, left, label: str):
    cv2.rectangle(bgr, (left, top), (right, bottom), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    y1 = max(top - th - baseline - 6, 0)
    y2 = top
    x1 = left
    x2 = left + tw + 10

    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), cv2.FILLED)
    cv2.putText(bgr, label, (left + 5, top - 6), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


class LiveRecognizer(VideoProcessorBase):
    """
    Runs face detection + embedding on incoming frames from browser camera.
    Matches against stored encodings (same folder SETTINGS.data_dir).
    """

    def __init__(self):
        self.last_db_load = 0.0
        self.known_names = []
        self.known_encs = np.empty((0, 128), dtype=np.float64)

        self.frame_idx = 0
        self.detect_every_n = 3  # adjust for speed
        self.resize_scale = 0.5  # 0.5 or 0.25 for faster
        self.last_locations = []
        self.last_labels = []

    def _maybe_reload_db(self):
        now = time.time()
        # reload every 3 seconds to reflect new enrolls
        if now - self.last_db_load > 3.0:
            self.known_names, self.known_encs = load_database(SETTINGS.data_dir)
            self.last_db_load = now

    def recv(self, frame):
        self._maybe_reload_db()
        self.frame_idx += 1

        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # speed-up: downscale
        small = cv2.resize(img, (0, 0), fx=self.resize_scale, fy=self.resize_scale)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        if self.frame_idx % self.detect_every_n == 0:
            # detect + encode
            locations = face_recognition.face_locations(rgb_small, model=SETTINGS.model)
            encs = face_recognition.face_encodings(rgb_small, locations)

            inv = 1.0 / self.resize_scale
            self.last_locations = []
            self.last_labels = []

            for (top, right, bottom, left), enc in zip(locations, encs):
                # scale back to original frame
                top = int(top * inv)
                right = int(right * inv)
                bottom = int(bottom * inv)
                left = int(left * inv)

                match = recognize_one(
                    self.known_names,
                    self.known_encs,
                    np.asarray(enc),
                    tolerance=SETTINGS.tolerance,
                )

                label = match.name if match else "Unknown"
                self.last_locations.append((top, right, bottom, left))
                self.last_labels.append(label)

        # draw last results (so we don't run heavy detect every frame)
        for (top, right, bottom, left), label in zip(self.last_locations, self.last_labels):
            # clamp
            top = max(0, min(top, h - 1))
            bottom = max(0, min(bottom, h - 1))
            left = max(0, min(left, w - 1))
            right = max(0, min(right, w - 1))
            _draw_box_label(img, top, right, bottom, left, label)

        return frame.from_ndarray(img, format="bgr24")


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Backend")
    api_base = st.text_input("FastAPI Base URL", value=DEFAULT_API_BASE)
    st.caption("Run backend first: `uvicorn app_fastapi:app --reload`")

    st.divider()
    page = st.radio("Page", ["Live Video", "Enroll", "Recognize", "People / Delete"], index=0)

    st.divider()
    st.subheader("Enrolled People")

    if st.button("Refresh people list", use_container_width=True):
        st.session_state["refresh_people"] = st.session_state.get("refresh_people", 0) + 1

    try:
        people_sidebar = api_get_people(api_base)
    except Exception:
        people_sidebar = []

    if not people_sidebar:
        st.caption("No enrolled people yet.")
    else:
        st.write(f"Total: {len(people_sidebar)}")
        st.code("\n".join(people_sidebar))


# ----------------------------
# Pages
# ----------------------------
if page == "Live Video":
    st.header("Live Video Recognition (Direct Camera)")

    st.info(
        "Bu sayfa **direkt browser kameranı** açar (WebRTC).\n"
        "İlk girişte Chrome kamera izni isteyecek → **Allow** de.\n"
        "Not: Kamera başka uygulamada açıksa (Teams/Discord/başka sekme) burada açılmaz."
    )

    # This is the critical part: browser camera
    webrtc_streamer(
        key="live_cam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=LiveRecognizer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

elif page == "Enroll":
    st.header("Enroll (Register) a Person")

    st.info(
        "Camera snapshot uses your browser camera.\n"
        "- Open as **http://localhost:8501**\n"
        "- Chrome: Site settings → Camera → **Allow**\n"
    )

    name = st.text_input("Person name", value="")
    image_bytes, filename = pick_image_bytes("Upload an image for enrollment")

    if image_bytes is not None:
        st.image(image_bytes, caption="Selected image", width="stretch")

    if st.button("Enroll", type="primary", use_container_width=True):
        if not name.strip():
            st.error("Please enter a name.")
        elif image_bytes is None:
            st.error("Please take a picture or upload an image.")
        else:
            try:
                res = api_enroll_file(api_base, name.strip(), image_bytes, filename or "image.jpg")
                st.success(f"Enrolled: {res.get('saved')}")
            except requests.HTTPError as e:
                st.error(f"Enroll failed: {extract_http_error_detail(e)}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

elif page == "Recognize":
    st.header("Recognize Faces (Snapshot)")

    image_bytes, filename = pick_image_bytes("Upload an image for recognition")

    if image_bytes is not None:
        st.image(image_bytes, caption="Selected image", width="stretch")

    if st.button("Recognize", type="primary", use_container_width=True):
        if image_bytes is None:
            st.error("Please take a picture or upload an image.")
        else:
            try:
                res = api_recognize_file(api_base, image_bytes, filename or "image.jpg")
                faces = res.get("faces", [])
                if not faces:
                    st.warning("No face found.")
                else:
                    st.subheader("Results")
                    for i, f in enumerate(faces, start=1):
                        nm = f.get("name", "Unknown")
                        dist = f.get("distance", None)
                        st.write(f"{i}. **{nm}**" + (f" — distance: `{float(dist):.4f}`" if dist is not None else ""))
            except requests.HTTPError as e:
                st.error(f"Recognition failed: {extract_http_error_detail(e)}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

else:
    st.header("People / Delete")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Enrolled People (from backend)")
        if st.button("Reload list", use_container_width=True):
            st.rerun()

        try:
            people = api_get_people(api_base)
        except Exception as e:
            st.error(f"Could not fetch people list: {e}")
            people = []

        if not people:
            st.info("No enrolled people yet.")
        else:
            st.write(f"Total enrolled: **{len(people)}**")
            st.dataframe({"people": people}, use_container_width=True)

    with col2:
        st.subheader("Delete")
        name_del = st.text_input("Name to delete", value="")

        if st.button("Delete", use_container_width=True):
            if not name_del.strip():
                st.error("Please enter a name.")
            else:
                try:
                    res = api_delete_person(api_base, name_del.strip())
                    st.success(f"Deleted: {res.get('deleted')}")
                except requests.HTTPError as e:
                    st.error(f"Delete failed: {extract_http_error_detail(e)}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")