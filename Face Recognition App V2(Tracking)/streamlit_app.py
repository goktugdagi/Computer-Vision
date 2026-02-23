import requests
import streamlit as st

DEFAULT_API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Face App V2 (Tracking)", layout="wide")
st.title("Face App V2 — Tracking (Backend Camera Ownership)")


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


def api_snapshot(api_base: str) -> bytes:
    r = requests.get(f"{api_base}/snapshot", timeout=10)
    r.raise_for_status()
    return r.content


def extract_http_error_detail(err: requests.HTTPError) -> str:
    try:
        return err.response.json().get("detail", str(err))
    except Exception:
        return str(err)


with st.sidebar:
    st.header("Backend")
    api_base = st.text_input("FastAPI Base URL", value=DEFAULT_API_BASE)
    st.caption("Run backend first: `uvicorn app_fastapi:app --reload`")

    st.divider()
    page = st.radio("Page", ["Live Video", "Enroll", "Recognize", "People / Delete"], index=1)

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


if page == "Live Video":
    st.header("Live Video")

    stream_url = f"{api_base}/video_feed"
    st.markdown(f"**Stream URL:** `{stream_url}`")

    st.markdown(
        f"""
        <img src="{stream_url}" style="width: 100%; border: 1px solid #ddd; border-radius: 8px;" />
        """,
        unsafe_allow_html=True,
    )

    st.info("Camera is owned by backend. Streamlit does NOT request camera permission.")


elif page == "Enroll":
    st.header("Enroll (Register) a Person")
    name = st.text_input("Person name", value="")

    colA, colB = st.columns([1.25, 1])

    # -------------------------
    # Option A: Live preview + Take Photo (backend snapshot)
    # -------------------------
    with colA:
        st.subheader("Option A — Live Preview + Take Photo (recommended)")

        stream_url = f"{api_base}/video_feed"
        st.caption("Live preview (backend MJPEG stream):")

        # Live preview always visible BEFORE taking photo
        st.markdown(
            f"""
            <img src="{stream_url}" style="width: 100%; border: 1px solid #ddd; border-radius: 10px;" />
            """,
            unsafe_allow_html=True,
        )

        st.write("")  # spacing

        # Message and captured image placeholders (stable UI)
        msg_box = st.empty()
        captured_box = st.empty()

        if st.button("Take Photo", type="primary", use_container_width=True):
            try:
                st.session_state["snapshot_bytes"] = api_snapshot(api_base)
                msg_box.success("Photo captured.")
            except requests.HTTPError as e:
                msg_box.error(f"Snapshot failed: {extract_http_error_detail(e)}")
            except Exception as e:
                msg_box.error(f"Unexpected error: {e}")

        snap = st.session_state.get("snapshot_bytes", None)
        if snap is not None:
            captured_box.image(snap, caption="Captured photo", use_container_width=True)

    # -------------------------
    # Option B: Upload
    # -------------------------
    with colB:
        st.subheader("Option B — Upload")
        up = st.file_uploader("Upload an image for enrollment", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if up is not None:
            st.session_state["upload_bytes"] = up.read()
            st.session_state["upload_name"] = up.name
            st.image(st.session_state["upload_bytes"], caption="Uploaded image", use_container_width=True)

    st.divider()

    if st.button("Enroll", use_container_width=True):
        if not name.strip():
            st.error("Please enter a name.")
        else:
            # choose snapshot first, else upload
            img = st.session_state.get("snapshot_bytes") or st.session_state.get("upload_bytes")
            fname = "snapshot.jpg" if st.session_state.get("snapshot_bytes") else st.session_state.get("upload_name", "upload.jpg")

            if img is None:
                st.error("Please take a photo (Option A) or upload an image (Option B).")
            else:
                try:
                    res = api_enroll_file(api_base, name.strip(), img, fname)
                    st.success(f"Enrolled: {res.get('saved')}")
                except requests.HTTPError as e:
                    st.error(f"Enroll failed: {extract_http_error_detail(e)}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")


elif page == "Recognize":
    st.header("Recognize Faces")

    colA, colB = st.columns([1.25, 1])

    with colA:
        st.subheader("Option A — Live Preview + Snapshot (recommended)")

        stream_url = f"{api_base}/video_feed"
        st.caption("Live preview (backend MJPEG stream):")
        st.markdown(
            f"""
            <img src="{stream_url}" style="width: 100%; border: 1px solid #ddd; border-radius: 10px;" />
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        msg_box = st.empty()
        captured_box = st.empty()

        if st.button("Take Photo for Recognition", type="primary", use_container_width=True):
            try:
                st.session_state["snapshot_bytes_rec"] = api_snapshot(api_base)
                msg_box.success("Photo captured.")
            except requests.HTTPError as e:
                msg_box.error(f"Snapshot failed: {extract_http_error_detail(e)}")
            except Exception as e:
                msg_box.error(f"Unexpected error: {e}")

        snap = st.session_state.get("snapshot_bytes_rec", None)
        if snap is not None:
            captured_box.image(snap, caption="Captured photo (for recognition)", use_container_width=True)

    with colB:
        st.subheader("Option B — Upload")
        up = st.file_uploader("Upload an image for recognition", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if up is not None:
            st.session_state["upload_bytes_rec"] = up.read()
            st.session_state["upload_name_rec"] = up.name
            st.image(st.session_state["upload_bytes_rec"], caption="Uploaded image", use_container_width=True)

    st.divider()

    if st.button("Recognize", use_container_width=True):
        img = st.session_state.get("snapshot_bytes_rec") or st.session_state.get("upload_bytes_rec")
        fname = "snapshot.jpg" if st.session_state.get("snapshot_bytes_rec") else st.session_state.get("upload_name_rec", "upload.jpg")

        if img is None:
            st.error("Please take a photo (Option A) or upload an image (Option B).")
        else:
            try:
                res = api_recognize_file(api_base, img, fname)
                faces = res.get("faces", [])

                if not faces:
                    st.warning("No face found.")
                else:
                    st.subheader("Results")
                    for i, f in enumerate(faces, start=1):
                        nm = f.get("name", "Unknown")
                        dist = f.get("distance", None)
                        if dist is None:
                            st.write(f"{i}. **{nm}**")
                        else:
                            st.write(f"{i}. **{nm}** — distance: `{float(dist):.4f}`")

            except requests.HTTPError as e:
                st.error(f"Recognition failed: {extract_http_error_detail(e)}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")


else:
    st.header("People / Delete")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Enrolled People")
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