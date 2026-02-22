import requests
import streamlit as st

DEFAULT_API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Face App", layout="wide")
st.title("Face App — Streamlit UI + FastAPI Backend")


# ----------------------------
# API helpers
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


def pick_image_bytes(label_upload: str):
    """
    Returns (image_bytes, filename) from camera snapshot OR file upload.
    """
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Option A — Camera snapshot")
        cam = st.camera_input("Take a picture")
        if cam is not None:
            return cam.getvalue(), "camera.jpg"

    with col2:
        st.subheader("Option B — Upload")
        up = st.file_uploader(label_upload, type=["jpg", "jpeg", "png", "bmp", "webp"])
        if up is not None:
            return up.read(), up.name

    return None, None


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
    st.header("Live Video Recognition (Boxes + Labels)")

    st.write(
        "This page displays a **live MJPEG stream** from FastAPI, "
        "with face bounding boxes and name labels drawn on each frame."
    )

    # Stream URL
    stream_url = f"{api_base}/video_feed"

    st.info("If you just enrolled a new person, wait ~3 seconds for the stream to refresh its database.")
    st.markdown(f"**Stream URL:** `{stream_url}`")

    # Render MJPEG stream
    # Streamlit can display MJPEG if served as multipart/x-mixed-replace and embedded as <img>.
    st.markdown(
        f"""
        <img src="{stream_url}" style="width: 100%; border: 1px solid #ddd; border-radius: 8px;" />
        """,
        unsafe_allow_html=True,
    )


elif page == "Enroll":
    st.header("Enroll (Register) a Person")

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
                st.info("Tip: refresh the sidebar people list.")
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
                        if dist is None:
                            st.write(f"{i}. **{nm}**")
                        else:
                            st.write(f"{i}. **{nm}** — distance: `{float(dist):.4f}`")

            except requests.HTTPError as e:
                st.error(f"Recognition failed: {extract_http_error_detail(e)}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")


else:  # People / Delete
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
                    st.info("Tip: reload list to see changes.")
                except requests.HTTPError as e:
                    st.error(f"Delete failed: {extract_http_error_detail(e)}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")