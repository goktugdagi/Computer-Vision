# streamlit run app_streamlit.py

import time
import requests
import streamlit as st
import av

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from facial_landmarks import FaceLandmarks
from core_blur import blur_faces


st.set_page_config(
    page_title="BlurGuard Studio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 0.6rem; padding-bottom: 2.5rem; }
    header[data-testid="stHeader"] { background: transparent; }
    div[data-testid="stToolbar"] { visibility: hidden; height: 0%; position: fixed; }

    .hero {
        background: linear-gradient(120deg, #6a11cb 0%, #2575fc 100%);
        padding: 28px 32px;
        border-radius: 22px;
        color: white;
        box-shadow: 0 10px 30px rgba(37,117,252,0.35);
        margin-bottom: 22px;
    }
    .hero-title { font-size: 2.55rem; font-weight: 900; letter-spacing: -0.8px; margin-bottom: 6px; line-height: 1.1; }
    .hero-sub { font-size: 1.03rem; opacity: 0.92; max-width: 920px; line-height: 1.55; }

    .status {
        display: inline-flex; align-items: center; gap: 10px;
        padding: 8px 14px; border-radius: 999px;
        font-size: 0.92rem; font-weight: 700;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.18);
        backdrop-filter: blur(6px); margin-top: 14px;
    }
    .dot { width: 10px; height: 10px; border-radius: 50%; background: #22c55e; box-shadow: 0 0 0 4px rgba(34,197,94,0.25); display: inline-block; }
    .dot-red { background: #ef4444; box-shadow: 0 0 0 4px rgba(239,68,68,0.25); }

    .card {
        background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%);
        border-radius: 18px; padding: 18px 20px;
        box-shadow: 0 8px 22px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.06);
    }
    .card-title { font-size: 1.07rem; font-weight: 800; margin-bottom: 6px; color: #16233a; }
    .card-sub { font-size: 0.93rem; color: #5a6b86; margin-bottom: 12px; line-height: 1.45; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8faff 0%, #eef2ff 100%);
        border-right: 1px solid rgba(0,0,0,0.06);
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.1rem; }

    .stTextInput input { border-radius: 14px !important; }

    div.stButton > button {
        background: linear-gradient(120deg, #6a11cb, #2575fc);
        color: white; font-weight: 800;
        border-radius: 14px; border: none;
        padding: 0.70rem 1.05rem;
        box-shadow: 0 8px 18px rgba(37,117,252,0.30);
        transition: all 0.2s ease;
    }
    div.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 12px 26px rgba(37,117,252,0.40); }

    div.stDownloadButton > button { border-radius: 14px; font-weight: 800; padding: 0.70rem 1.05rem; }
    button[data-baseweb="tab"] { font-weight: 900; font-size: 0.95rem; }

    .muted { color: rgba(49,51,63,0.70); font-size: 0.95rem; line-height: 1.45; }
    hr { margin: 1.1rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_health(api_base: str) -> bool:
    try:
        r = requests.get(f"{api_base}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def post_blur_video(api_base: str, video_bytes: bytes, filename: str, blur_ksize: int, max_frames: int | None, detect_every_n: int) -> bytes:
    files = {"file": (filename, video_bytes)}
    params = {"blur_ksize": blur_ksize, "detect_every_n": detect_every_n}
    if max_frames is not None:
        params["max_frames"] = max_frames

    r = requests.post(f"{api_base}/blur-video", files=files, params=params, timeout=600)
    if r.status_code != 200:
        raise RuntimeError(f"API Error ({r.status_code}): {r.text}")
    return r.content


@st.cache_resource
def get_live_processor(max_faces_cached: int):
    return FaceLandmarks(max_num_faces=max_faces_cached, static_image_mode=False)


RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


def live_video_callback(frame: av.VideoFrame, blur_ksize: int, max_faces: int) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    fl = get_live_processor(max_faces)
    all_faces = fl.get_facial_landmarks(img)
    out = blur_faces(img, all_faces, blur_ksize=blur_ksize)
    return av.VideoFrame.from_ndarray(out, format="bgr24")


with st.sidebar:
    st.markdown("## Control Panel")
    st.caption("Tune blur strength and processing parameters.")

    st.markdown("### Backend")
    api_base = st.text_input("FastAPI Base URL", value="http://localhost:8000").strip().rstrip("/")
    api_ok = api_health(api_base)

    st.markdown("---")
    st.markdown("### Blur Settings")
    blur_ksize = st.slider("Blur strength (base kernel size)", 5, 99, 35, 2)

    st.markdown("### Webcam Settings")
    max_faces = st.slider("Max faces (webcam)", 1, 10, 5, 1)

    st.markdown("### Video Upload Performance")
    detect_every_n = st.slider("Detect every N frames", 1, 30, 5, 1)

    test_limit = st.checkbox("Limit frames (testing)", value=False)
    max_frames = None
    if test_limit:
        max_frames = st.number_input("Max frames", 1, 5000, 300, 50)

    st.markdown("---")
    st.caption("Note: FFmpeg must be installed and available in PATH for browser-playable MP4 output.")


dot_class = "dot" if api_ok else "dot dot-red"
status_text = "Backend Connected" if api_ok else "Backend Not Reachable"
status_hint = "Ready for video uploads." if api_ok else "Start it with: uvicorn api:app --reload"

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">BlurGuard Studio</div>
        <div class="hero-sub">
            Privacy-first face anonymization for <b>live webcam streams</b> and <b>uploaded videos</b>.
            Powered by MediaPipe FaceMesh and OpenCV, with a FastAPI backend for batch processing.
        </div>
        <div class="status">
            <span class="{dot_class}"></span>
            {status_text} &nbsp;•&nbsp; {status_hint}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_webcam, tab_upload = st.tabs(["Webcam Live Blur", "Video Upload"])


with tab_webcam:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Webcam Live Blur</div>
            <div class="card-sub">
                Real-time anonymization in the browser via WebRTC. Grant camera permissions when prompted.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    def _callback(frame: av.VideoFrame) -> av.VideoFrame:
        return live_video_callback(frame, blur_ksize=blur_ksize, max_faces=max_faces)

    # IMPORTANT: request higher webcam resolution & fps (browser decides the closest possible)
    media_constraints = {
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
            "frameRate": {"ideal": 30},
            "facingMode": "user",
        },
        "audio": False,
    }

    st.markdown('<div class="card">', unsafe_allow_html=True)
    webrtc_streamer(
        key="live_blur",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=_callback,
        media_stream_constraints=media_constraints,
        async_processing=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="muted">'
        "<b>Tip:</b> If FPS drops, reduce <code>Max faces</code> or lower the requested webcam resolution in code."
        "</div>",
        unsafe_allow_html=True,
    )


with tab_upload:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Video Upload</div>
            <div class="card-sub">
                Upload a video, process it via FastAPI, preview the result, and download a blurred MP4.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2 = st.columns([0.5, 0.5], gap="large")

    with c1:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">Input</div>
                <div class="card-sub">Supported formats: MP4, MOV, AVI, MKV</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        video_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])
        if video_file is not None:
            st.video(video_file)

    with c2:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">Output</div>
                <div class="card-sub">Processed MP4 output (H.264 recommended for browser playback)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")

        if video_file is None:
            st.info("Upload a video to enable processing.")
        else:
            if not api_ok:
                st.error("FastAPI backend is not reachable. Start it and try again.")
            else:
                if st.button("Process Video", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Processing video..."):
                            start = time.time()
                            out_bytes = post_blur_video(
                                api_base=api_base,
                                video_bytes=video_file.getvalue(),
                                filename=video_file.name,
                                blur_ksize=blur_ksize,
                                max_frames=max_frames,
                                detect_every_n=detect_every_n,
                            )
                            elapsed = time.time() - start

                        st.success(f"Done in {elapsed:.1f}s")
                        st.video(out_bytes, format="video/mp4")
                        st.download_button(
                            "Download processed video",
                            data=out_bytes,
                            file_name="blurred.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(str(e))


st.markdown("---")
st.markdown(
    """
    <div class="muted">
        <b>Ethics note:</b> Use this tool for privacy-preserving anonymization only. Avoid surveillance, tracking, or identity inference use cases.
    </div>
    """,
    unsafe_allow_html=True,
)
