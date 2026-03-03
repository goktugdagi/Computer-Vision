import time
import requests
import streamlit as st

st.set_page_config(page_title="Face App V4", layout="wide")

# --- session state init ---
if "captured_jpg" not in st.session_state:
    st.session_state["captured_jpg"] = None
if "captured_ts" not in st.session_state:
    st.session_state["captured_ts"] = None

st.title("Face App V4 — SCRFD + ArcFace (ONNX) + FastAPI + Streamlit")

with st.sidebar:
    st.header("FastAPI Base URL")
    base_url = st.text_input("Base URL", "http://127.0.0.1:8000")

    st.caption("Run backend:")
    st.code("python -m uvicorn app_fastapi:app --reload")

    st.markdown("---")
    page = st.radio("Page", ["Live Video", "Enroll", "Recognize", "People / Delete"])

    st.markdown("---")
    st.subheader("Enrolled People")
    if st.button("Refresh people list"):
        try:
            st.rerun()
        except Exception:
            st.stop()

    try:
        r = requests.get(f"{base_url}/api/people", timeout=3)
        data = r.json()
        people = data.get("people", [])
    except Exception:
        people = []

    st.write(f"Total: {len(people)}")
    if people:
        for p in people:
            st.write(f"- {p}")
    else:
        st.caption("No enrolled people yet.")


def api_snapshot():
    r = requests.get(f"{base_url}/api/snapshot", timeout=10)
    return r.content


def api_enroll_bytes(name: str, jpg_bytes: bytes):
    files = {"file": ("capture.jpg", jpg_bytes, "image/jpeg")}
    data = {"name": name}
    r = requests.post(f"{base_url}/api/enroll_file", files=files, data=data, timeout=60)
    return r


def api_recognize_bytes(jpg_bytes: bytes):
    files = {"file": ("img.jpg", jpg_bytes, "image/jpeg")}
    r = requests.post(f"{base_url}/api/recognize_file", files=files, timeout=120)
    return r


if page == "Live Video":
    st.subheader("Live Video")
    st.caption("MJPEG stream from backend (/live)")
    st.image(f"{base_url}/live")

elif page == "Enroll":
    st.subheader("Enroll (Register) a Person")
    name = st.text_input("Person name", "")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### Option A — Live Preview + Take Photo (recommended)")
        st.image(f"{base_url}/preview")

        if st.button("Take Photo"):
            try:
                jpg = api_snapshot()
                st.session_state["captured_jpg"] = jpg
                st.session_state["captured_ts"] = time.time()
                st.success("Photo captured.")
            except Exception as e:
                st.error(f"Snapshot failed: {e}")

        if st.session_state["captured_jpg"] is not None:
            st.markdown("**Captured photo (from camera):**")
            st.image(st.session_state["captured_jpg"])

            if st.button("Enroll captured photo"):
                if not name.strip():
                    st.error("Please enter a person name.")
                else:
                    resp = api_enroll_bytes(name.strip(), st.session_state["captured_jpg"])
                    if resp.status_code == 200:
                        st.success(f"Enrolled: {name}")
                        st.write(resp.json())

                        # reset UI state
                        st.session_state["captured_jpg"] = None
                        st.session_state["captured_ts"] = None
                        try:
                            st.rerun()
                        except Exception:
                            st.stop()
                    else:
                        st.error(f"Enroll failed ({resp.status_code}): {resp.text}")

    with colB:
        st.markdown("### Option B — Upload")
        up = st.file_uploader("Upload an image for enrollment", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if up is not None:
            img_bytes = up.read()
            st.image(img_bytes)

            if st.button("Enroll uploaded image"):
                if not name.strip():
                    st.error("Please enter a person name.")
                else:
                    resp = api_enroll_bytes(name.strip(), img_bytes)
                    if resp.status_code == 200:
                        st.success(f"Enrolled: {name}")
                        st.write(resp.json())

                        st.session_state["captured_jpg"] = None
                        st.session_state["captured_ts"] = None
                        try:
                            st.rerun()
                        except Exception:
                            st.stop()
                    else:
                        st.error(f"Enroll failed ({resp.status_code}): {resp.text}")

elif page == "Recognize":
    st.subheader("Recognize (Single Image)")
    up = st.file_uploader("Upload an image to recognize", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if up is not None:
        img_bytes = up.read()
        st.image(img_bytes)

        if st.button("Run Recognize"):
            try:
                resp = api_recognize_bytes(img_bytes)
                if resp.status_code == 200:
                    st.markdown("**Result:**")
                    st.image(resp.content)
                else:
                    st.error(f"Recognize failed ({resp.status_code}): {resp.text}")
            except Exception as e:
                st.error(f"Recognize failed: {e}")

elif page == "People / Delete":
    st.subheader("People / Delete")
    if not people:
        st.info("No enrolled people.")
    else:
        selected = st.selectbox("Select person", people)
        if st.button("Delete selected person"):
            try:
                r = requests.post(f"{base_url}/api/delete_person", data={"name": selected}, timeout=10)
                if r.status_code == 200 and r.json().get("ok"):
                    st.success(f"Deleted: {selected}")
                    try:
                        st.rerun()
                    except Exception:
                        st.stop()
                else:
                    st.error(f"Delete failed: {r.text}")
            except Exception as e:
                st.error(f"Delete failed: {e}")