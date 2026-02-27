import requests
import streamlit as st
import streamlit.components.v1 as components

DEFAULT_API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Face App V3 (InsightFace)", layout="wide")
st.title("Face App V3 — InsightFace (ArcFace) + FastAPI + Streamlit")


# ----------------------------
# Helpers
# ----------------------------
def api_get_people(api_base: str) -> list[str]:
    r = requests.get(f"{api_base}/people", timeout=10)
    r.raise_for_status()
    return r.json().get("people", [])


def api_delete_person(api_base: str, name: str) -> dict:
    r = requests.delete(f"{api_base}/people/{name}", timeout=30)
    r.raise_for_status()
    return r.json()


def api_snapshot(api_base: str) -> bytes:
    r = requests.get(f"{api_base}/snapshot", timeout=30)
    r.raise_for_status()
    return r.content


def api_enroll_file(api_base: str, name: str, image_bytes: bytes, filename: str) -> dict:
    files = {"image": (filename, image_bytes, "application/octet-stream")}
    r = requests.post(f"{api_base}/enroll_file", params={"name": name}, files=files, timeout=60)
    r.raise_for_status()
    return r.json()


def api_recognize_json(api_base: str, image_bytes: bytes, filename: str) -> dict:
    files = {"image": (filename, image_bytes, "application/octet-stream")}
    r = requests.post(f"{api_base}/recognize_json", files=files, timeout=60)
    r.raise_for_status()
    return r.json()


def api_recognize_image(api_base: str, image_bytes: bytes, filename: str) -> bytes:
    files = {"image": (filename, image_bytes, "application/octet-stream")}
    r = requests.post(f"{api_base}/recognize_file", files=files, timeout=60)
    r.raise_for_status()
    return r.content


def extract_http_error_detail(err: requests.HTTPError) -> str:
    try:
        return err.response.json().get("detail", str(err))
    except Exception:
        return str(err)


def mjpeg_html(url: str, height: int = 520) -> str:
    return f"""
    <div style="display:flex; justify-content:center;">
      <img src="{url}" style="max-width: 100%; height: {height}px; border-radius: 12px; border: 1px solid #ddd;" />
    </div>
    """


# ----------------------------
# Sidebar
# ----------------------------
api_base = st.sidebar.text_input("FastAPI Base URL", DEFAULT_API_BASE)
st.sidebar.caption("Run backend: `python -m uvicorn app_fastapi:app --reload`")

st.sidebar.markdown("---")
page = st.sidebar.radio("Page", ["Live Video", "Enroll", "Recognize", "People / Delete"])

st.sidebar.markdown("---")
st.sidebar.subheader("Enrolled People")

if st.sidebar.button("Refresh people list"):
    st.session_state["people_cache"] = None

people = st.session_state.get("people_cache")
if people is None:
    try:
        people = api_get_people(api_base)
    except Exception:
        people = []
    st.session_state["people_cache"] = people

st.sidebar.write(f"Total: {len(people)}")
if len(people) == 0:
    st.sidebar.caption("No enrolled people yet.")
else:
    st.sidebar.code("\n".join(people))


# ----------------------------
# Session state
# ----------------------------
if "captured_jpeg" not in st.session_state:
    st.session_state["captured_jpeg"] = None


# ----------------------------
# Pages
# ----------------------------
if page == "Live Video":
    st.subheader("Live Video (Recognition Overlay)")
    st.caption("Camera is owned by backend. This is an MJPEG stream.")
    components.html(mjpeg_html(f"{api_base}/live_feed", height=560), height=600)

elif page == "Enroll":
    st.subheader("Enroll (Register) a Person")

    name = st.text_input("Person name")

    colA, colB = st.columns(2)

    # ---- Option A: Backend Live Preview + Take Photo ----
    with colA:
        st.markdown("### Option A — Live Preview + Take Photo (recommended)")

        if st.session_state["captured_jpeg"] is None:
            # Live preview before taking photo (your request)
            components.html(mjpeg_html(f"{api_base}/preview_feed", height=420), height=460)

            if st.button("Take Photo", type="primary", use_container_width=True):
                try:
                    st.session_state["captured_jpeg"] = api_snapshot(api_base)
                    st.success("Photo captured. Now you can Enroll.")
                    st.rerun()
                except requests.HTTPError as e:
                    st.error(f"Snapshot failed: {extract_http_error_detail(e)}")
                except Exception as e:
                    st.error(f"Snapshot failed: {e}")
        else:
            # Show which photo was taken (your request)
            st.image(st.session_state["captured_jpeg"], caption="Captured photo (from camera)", use_container_width=True)
            if st.button("Retake", use_container_width=True):
                st.session_state["captured_jpeg"] = None
                st.rerun()

    # ---- Option B: Upload ----
    with colB:
        st.markdown("### Option B — Upload")
        uploaded = st.file_uploader("Upload an image for enrollment", type=["jpg", "jpeg", "png", "bmp", "webp"])

        # ✅ Bring back preview of uploaded image (your request)
        if uploaded is not None:
            st.image(uploaded.getvalue(), caption=f"Uploaded image: {uploaded.name}", use_container_width=True)

    st.markdown("---")

    if st.button("Enroll", use_container_width=True):
        if not name.strip():
            st.warning("Please enter a person name.")
        else:
            try:
                # Prefer captured photo if exists; otherwise use uploaded.
                if st.session_state["captured_jpeg"] is not None:
                    res = api_enroll_file(api_base, name, st.session_state["captured_jpeg"], "snapshot.jpg")
                elif uploaded is not None:
                    res = api_enroll_file(api_base, name, uploaded.getvalue(), uploaded.name)
                else:
                    st.warning("Take a photo or upload an image first.")
                    st.stop()

                st.success(f"Enrolled: {res.get('saved')} (engine: {res.get('engine_backend')})")

                # ✅ After success: clear captured photo so Enroll returns to live preview.
                st.session_state["captured_jpeg"] = None
                st.session_state["people_cache"] = None

                # Re-render page (live preview appears again)
                st.rerun()

            except requests.HTTPError as e:
                st.error(extract_http_error_detail(e))
            except Exception as e:
                st.error(f"Enroll failed: {e}")

elif page == "Recognize":
    st.subheader("Recognize (Upload an Image)")
    st.caption("Upload a photo. The backend will detect faces, match them, and return an annotated image + results.")

    up = st.file_uploader("Upload an image to recognize", type=["jpg", "jpeg", "png", "bmp", "webp"], key="rec_up")
    if up is None:
        st.info("Upload an image to start.")
        st.stop()

    img_bytes = up.getvalue()

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_bytes, caption=f"Input image: {up.name}", use_container_width=True)

    if st.button("Run Recognition", type="primary", use_container_width=True):
        try:
            # 1) Get JSON results
            js = api_recognize_json(api_base, img_bytes, up.name)

            # 2) Get annotated image (with boxes+labels)
            annotated = api_recognize_image(api_base, img_bytes, up.name)

            with col2:
                st.image(annotated, caption="Annotated output (boxes + labels)", use_container_width=True)

            st.markdown("### Results")
            st.json(js)

        except requests.HTTPError as e:
            st.error(extract_http_error_detail(e))
        except Exception as e:
            st.error(f"Recognition failed: {e}")

elif page == "People / Delete":
    st.subheader("People / Delete")

    if len(people) == 0:
        st.info("No enrolled people.")
    else:
        who = st.selectbox("Select a person to delete", people)
        if st.button("Delete", type="secondary"):
            try:
                res = api_delete_person(api_base, who)
                st.success(f"Deleted: {res.get('deleted')}")
                st.session_state["people_cache"] = None
                st.rerun()
            except requests.HTTPError as e:
                st.error(extract_http_error_detail(e))
            except Exception as e:
                st.error(f"Delete failed: {e}")