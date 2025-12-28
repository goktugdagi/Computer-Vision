import os  # Built-in library for file and directory operations
import time  # Built-in library to control frame loop delays
import cv2  # OpenCV for camera capture and image processing
import numpy as np  # NumPy for array operations and canvas creation
import streamlit as st  # Streamlit for building a web-based UI
from cvzone.SelfiSegmentationModule import SelfiSegmentation  # Cvzone segmentation module for background removal
from urllib.request import urlopen, Request  # Libraries for opening URLs and sending HTTP requests

def letterbox(img, target_w, target_h, pad_value=0):  # Function to fit image into target resolution without cropping
    if img is None:  # Check if image could not be read
        return None  # Return None if invalid
    h, w = img.shape[:2]  # Get original image height and width
    if h == 0 or w == 0:  # Check if image has corrupt dimensions
        return None  # Skip if invalid

    scale = min(target_w / w, target_h / h)  # Compute scaling factor to contain the image
    new_w = max(1, int(w * scale))  # Calculate new width after scaling, avoid zero
    new_h = max(1, int(h * scale))  # Calculate new height after scaling, avoid zero

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)  # Resize image with linear interpolation
    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)  # Create target-size canvas filled with pad color

    x = (target_w - new_w) // 2  # Compute horizontal offset to center the image
    y = (target_h - new_h) // 2  # Compute vertical offset to center the image
    canvas[y:y + new_h, x:x + new_w] = resized  # Paste resized image into centered region
    return canvas  # Return letterboxed image

def load_backgrounds_from_folder(images_dir, target_w, target_h):  # Function to load backgrounds from directory
    bg_list, bg_names = [], []  # Lists to store background images and names
    if not os.path.isdir(images_dir):  # Check if directory exists
        return bg_list, bg_names  # Return empty lists if folder missing

    for f in sorted(os.listdir(images_dir)):  # Iterate over sorted files in folder
        path = os.path.join(images_dir, f)  # Build full file path
        if not os.path.isfile(path):  # Ensure entry is a file
            continue  # Skip non-files
        bg = cv2.imread(path)  # Read background image (BGR)
        if bg is None:  # Check if image decode failed
            continue  # Skip invalid images
        bg = cv2.resize(bg, (target_w, target_h), interpolation=cv2.INTER_LINEAR)  # Resize background to match target resolution
        bg_list.append(bg)  # Append background image
        bg_names.append(f)  # Append file name
    return bg_list, bg_names  # Return lists

def decode_uploaded_image_to_bgr(uploaded_file):  # Decode uploaded file into OpenCV BGR format
    if uploaded_file is None:  # Check if file exists
        return None  # Return None if no file
    bytes_data = uploaded_file.read()  # Read raw bytes (note: consumes the file buffer)
    if not bytes_data:  # Check if file empty
        return None  # Skip if empty
    buf = np.frombuffer(bytes_data, dtype=np.uint8)  # Convert bytes into NumPy buffer
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # Decode into BGR
    return img  # Return decoded image

def decode_url_image_to_bgr(url: str, timeout_sec: int = 10):  # Download and decode direct image URL into BGR
    url = (url or "").strip()  # Strip whitespace safely
    if not url:  # Check if empty
        return None  # Return None if invalid

    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})  # Send HTTP request with browser-like header
    with urlopen(req, timeout=timeout_sec) as resp:  # Open URL with timeout
        data = resp.read()  # Read downloaded image bytes

    if not data:  # Check if empty
        return None  # Skip if failed

    buf = np.frombuffer(data, dtype=np.uint8)  # Convert downloaded bytes into buffer
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # Decode into BGR
    return img  # Return decoded image

st.set_page_config(page_title="Background Removal (cvzone)", layout="wide")  # Configure Streamlit page
st.title("Camera Background Removal — Streamlit UI")  # Set UI title

if "cap" not in st.session_state:  # Check if camera object stored
    st.session_state.cap = None  # Initialize as None
if "segmentor" not in st.session_state:  # Check if segmentor exists
    st.session_state.segmentor = SelfiSegmentation()  # Initialize model once per session
if "target_wh" not in st.session_state:  # Check if resolution stored
    st.session_state.target_wh = None  # Init None
if "bg_list" not in st.session_state:  # Check background list
    st.session_state.bg_list = []  # Init empty
if "bg_names" not in st.session_state:  # Check background name list
    st.session_state.bg_names = []  # Init empty
if "uploaded_bg_list" not in st.session_state:  # Check upload/url backgrounds
    st.session_state.uploaded_bg_list = []  # Init empty
if "uploaded_bg_names" not in st.session_state:  # Check upload/url names
    st.session_state.uploaded_bg_names = []  # Init empty
if "bg_index" not in st.session_state:  # Check selected background index
    st.session_state.bg_index = 0  # Default 0

with st.sidebar:  # Sidebar section begin
    st.subheader("Controls")  # Sidebar title

    camera_index = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)  # Camera selector
    cut_thr = st.slider("cutThreshold", 0.0, 1.0, 0.95, 0.01)  # Segmentation threshold slider
    fps_limit = st.slider("UI FPS limit", 5, 60, 24, 1)  # UI FPS limit slider
    pad_value = st.slider("Letterbox padding (0=black)", 0, 255, 0, 1)  # Padding slider

    st.divider()  # UI divider
    run = st.toggle("Run", value=False)  # Camera start/stop toggle

    st.divider()  # UI divider
    st.subheader("Background Blur")  # Section header for blur controls
    blur_on = st.toggle("Blur background", value=False)  # Toggle to enable/disable blur background mode
    blur_k = st.slider("Blur strength (kernel)", 3, 51, 21, 2)  # Blur kernel size (odd preferred), step=2 keeps it mostly odd

    st.divider()  # UI divider
    st.subheader("Add background")  # Upload section title

    uploaded = st.file_uploader(  # File uploader widget
        "Upload background image (jpg/png)",  # Widget label
        type=["jpg", "jpeg", "png", "webp"],  # Allowed formats
        accept_multiple_files=False  # Single file only
    )

    col_u1, col_u2 = st.columns(2)  # Two buttons in one row
    add_upload_clicked = col_u1.button("Add upload", use_container_width=True)  # Add uploaded image to list
    clear_clicked = col_u2.button("Clear added", use_container_width=True)  # Clear uploaded/url-added backgrounds

    st.markdown("---")  # Horizontal separator
    bg_url = st.text_input("Or paste image URL", placeholder="https://.../image.jpg")  # URL input
    add_url_clicked = st.button("Add URL", use_container_width=True)  # Button to add URL image

col1, col2 = st.columns(2)  # Create 2 main columns for images
ph_in = col1.empty()  # Placeholder for input frame
ph_out = col2.empty()  # Placeholder for output frame

def open_camera(idx: int):  # Function to open camera
    return cv2.VideoCapture(idx)  # Open webcam by index

if not run:  # If Run toggle is off
    if st.session_state.cap is not None:  # If camera was previously opened
        st.session_state.cap.release()  # Release camera resource (recommended Streamlit-safe cleanup)
        st.session_state.cap = None  # Reset camera object
    st.info("Enable 'Run' from sidebar to start the camera.")  # Inform the user
    st.stop()  # Stop Streamlit script execution (wait for next rerun)

if st.session_state.cap is None:  # If camera is not opened yet
    st.session_state.cap = open_camera(int(camera_index))  # Open selected camera

    ok, first = st.session_state.cap.read()  # Read first frame (used to set target size)
    if not ok or first is None:  # If camera read fails
        st.error("Camera frame could not be read. Check camera index / permissions.")  # Show error
        st.session_state.cap.release()  # Release camera
        st.session_state.cap = None  # Reset
        st.stop()  # Stop

    h, w = first.shape[:2]  # Get first frame resolution
    st.session_state.target_wh = (w, h)  # Store target width/height from first frame

    folder_bgs, folder_names = load_backgrounds_from_folder("images", w, h)  # Load backgrounds from ./images
    st.session_state.bg_list = folder_bgs  # Store folder backgrounds
    st.session_state.bg_names = folder_names  # Store folder names
    st.session_state.bg_index = 0  # Reset background index

w, h = st.session_state.target_wh  # Unpack target resolution (width, height)

if clear_clicked:  # If clear button pressed
    st.session_state.uploaded_bg_list = []  # Clear uploaded/url background images
    st.session_state.uploaded_bg_names = []  # Clear uploaded/url background names
    st.session_state.bg_index = 0  # Reset selection index
    st.success("Added backgrounds cleared.")  # Show success message

if add_upload_clicked:  # If add-upload button pressed
    img_bgr = decode_uploaded_image_to_bgr(uploaded)  # Decode uploaded file into BGR image
    if img_bgr is None:  # If no file or decode failure
        st.warning("No image uploaded (or it could not be decoded).")  # Warn user
    else:
        img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)  # Resize to target resolution
        name = getattr(uploaded, "name", "uploaded")  # Get uploaded file name if available
        name_disp = f"[UPLOAD] {name}"  # Display name with upload tag
        st.session_state.uploaded_bg_list.append(img_bgr)  # Add image to list
        st.session_state.uploaded_bg_names.append(name_disp)  # Add name to list
        st.success(f"Added upload: {name}")  # Show success

if add_url_clicked:  # If add-URL button pressed
    if not bg_url.strip():  # If URL empty
        st.warning("Please paste an image URL.")  # Warn user
    else:
        try:
            img_bgr = decode_url_image_to_bgr(bg_url.strip(), timeout_sec=10)  # Download and decode image from URL
            if img_bgr is None:  # If download/decode fails
                st.error("Could not download/decode image from URL. Make sure it is a direct image link.")  # Show error
            else:
                img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)  # Resize to target resolution
                short = bg_url.strip()  # Prepare short display label
                if len(short) > 40:  # If URL too long
                    short = short[:37] + "..."  # Truncate for UI
                st.session_state.uploaded_bg_list.append(img_bgr)  # Add URL image to list
                st.session_state.uploaded_bg_names.append(f"[URL] {short}")  # Add URL label to list
                st.success("Added URL image.")  # Show success
        except Exception as e:  # Catch network / decode exceptions
            st.error(f"URL download failed: {e}")  # Show exception message

all_bgs = st.session_state.bg_list + st.session_state.uploaded_bg_list  # Merge folder and added background images
all_names = st.session_state.bg_names + st.session_state.uploaded_bg_names  # Merge folder and added background names

if not all_bgs:  # If no backgrounds exist
    st.warning("No backgrounds found. Add images to ./images, upload, or add URL.")  # Warn user
    all_bgs = []  # Keep as empty list
    all_names = []  # Keep as empty list

if all_names:  # If there are any background labels
    st.session_state.bg_index = st.sidebar.selectbox(  # Background selection dropdown
        "Background image",  # Dropdown label
        options=list(range(len(all_names))),  # Indices of available backgrounds
        format_func=lambda i: all_names[i],  # Show human-readable name
        index=min(st.session_state.bg_index, len(all_names) - 1),  # Clamp index safely
    )

delay = 1.0 / float(fps_limit)  # Convert FPS limit to per-frame sleep delay

while True:  # Main processing loop (runs while Streamlit script is in 'Run' mode)
    ok, frame = st.session_state.cap.read()  # Read frame from camera
    if not ok or frame is None:  # If reading fails
        st.warning("Camera stream ended.")  # Warn user
        break  # Exit loop

    frame = cv2.flip(frame, 1)  # Mirror the camera for a more natural selfie view
    frame_lb = letterbox(frame, w, h, pad_value=pad_value)  # Fit frame into target size without cropping

    if blur_on:  # If blur background mode is enabled
        k = int(blur_k)  # Read blur kernel size from slider
        if k < 3:  # Ensure minimum odd kernel size
            k = 3  # Clamp to 3
        if k % 2 == 0:  # GaussianBlur requires odd kernel size for best behavior
            k += 1  # Make it odd if it isn't
        blurred_bg = cv2.GaussianBlur(frame_lb, (k, k), 0)  # Create blurred version of current frame as background
        out = st.session_state.segmentor.removeBG(frame_lb, blurred_bg, cutThreshold=float(cut_thr))  # Composite person over blurred background
    else:  # If blur mode is off
        if all_bgs:  # If we have backgrounds
            bg = all_bgs[st.session_state.bg_index]  # Get selected background image
            out = st.session_state.segmentor.removeBG(frame_lb, bg, cutThreshold=float(cut_thr))  # Replace background with chosen image
        else:
            out = frame_lb  # Fallback: show original frame if no backgrounds exist

    ph_in.image(cv2.cvtColor(frame_lb, cv2.COLOR_BGR2RGB), caption="Input", use_container_width=True)  # Show input frame in UI
    ph_out.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="Output", use_container_width=True)  # Show output frame in UI

    time.sleep(delay)  # Sleep to respect the UI FPS limit

    if not st.session_state.get("Run", True):  # If user turns off Run toggle
        break  # Exit loop cleanly

# NOTE: We do NOT call cap.release() here, because Streamlit reruns scripts and we manage cleanup when Run turns off.  # Explanation comment
# NOTE: We do NOT call cv2.destroyAllWindows(), because we are not creating any OpenCV GUI windows in Streamlit.  # Explanation comment



# import os  # Built-in library for file and directory operations
# import time  # Built-in library to control frame loop delays
# import cv2  # OpenCV for camera capture and image processing
# import numpy as np  # NumPy for array operations and canvas creation
# import streamlit as st  # Streamlit for building a web-based UI
# from cvzone.SelfiSegmentationModule import SelfiSegmentation  # Cvzone segmentation module for background removal
# from urllib.request import urlopen, Request  # Libraries for opening URLs and sending HTTP requests

# def letterbox(img, target_w, target_h, pad_value=0):  # Function to fit image into target resolution without cropping
#     if img is None:  # Check if image could not be read
#         return None  # Return None if invalid
#     h, w = img.shape[:2]  # Get original image height and width
#     if h == 0 or w == 0:  # Check if image has corrupt dimensions
#         return None  # Skip if invalid

#     scale = min(target_w / w, target_h / h)  # Compute scaling factor to contain the image
#     new_w = max(1, int(w * scale))  # Calculate new width after scaling, avoid zero
#     new_h = max(1, int(h * scale))  # Calculate new height after scaling, avoid zero

#     resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)  # Resize image with linear interpolation
#     canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)  # Create target-size canvas filled with pad color

#     x = (target_w - new_w) // 2  # Compute horizontal offset to center the image
#     y = (target_h - new_h) // 2  # Compute vertical offset to center the image
#     canvas[y:y + new_h, x:x + new_w] = resized  # Paste resized image into centered region
#     return canvas  # Return letterboxed image

# def load_backgrounds_from_folder(images_dir, target_w, target_h):  # Function to load backgrounds from directory
#     bg_list, bg_names = [], []  # Lists to store background images and names
#     if not os.path.isdir(images_dir):  # Check if directory exists
#         return bg_list, bg_names  # Return empty lists if folder missing

#     for f in sorted(os.listdir(images_dir)):  # Iterate over sorted files in folder
#         path = os.path.join(images_dir, f)  # Build full file path
#         if not os.path.isfile(path):  # Ensure entry is a file
#             continue  # Skip non-files
#         bg = cv2.imread(path)  # Read background image
#         if bg is None:  # Check if image decode failed
#             continue  # Skip invalid images
#         bg = cv2.resize(bg, (target_w, target_h), interpolation=cv2.INTER_LINEAR)  # Resize background to match target resolution
#         bg_list.append(bg)  # Append background image
#         bg_names.append(f)  # Append file name
#     return bg_list, bg_names  # Return lists

# def decode_uploaded_image_to_bgr(uploaded_file):  # Decode uploaded file into OpenCV BGR format
#     if uploaded_file is None:  # Check if file exists
#         return None  # Return None if no file
#     bytes_data = uploaded_file.read()  # Read raw bytes
#     if not bytes_data:  # Check if file empty
#         return None  # Skip if empty
#     buf = np.frombuffer(bytes_data, dtype=np.uint8)  # Convert bytes into NumPy buffer
#     img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # Decode into BGR
#     return img  # Return decoded image

# def decode_url_image_to_bgr(url: str, timeout_sec: int = 10):  # Download and decode direct image URL into BGR
#     url = (url or "").strip()  # Strip whitespace safely
#     if not url:  # Check if empty
#         return None  # Return None if invalid
#     req = Request(url, headers={"User-Agent": "Mozilla/5.0"})  # Send HTTP request with browser-like header
#     with urlopen(req, timeout=timeout_sec) as resp:  # Open URL with timeout
#         data = resp.read()  # Read downloaded image bytes
#     if not data:  # Check if empty
#         return None  # Skip if failed
#     buf = np.frombuffer(data, dtype=np.uint8)  # Convert downloaded bytes into buffer
#     img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # Decode into BGR
#     return img  # Return decoded image

# st.set_page_config(page_title="Background Removal (cvzone)", layout="wide")  # Configure Streamlit page
# st.title("Camera Background Removal — Streamlit UI")  # Set UI title

# if "cap" not in st.session_state:  # Check if camera object stored
#     st.session_state.cap = None  # Initialize as None
# if "segmentor" not in st.session_state:  # Check if segmentor exists
#     st.session_state.segmentor = SelfiSegmentation()  # Initialize model
# if "target_wh" not in st.session_state:  # Check if resolution stored
#     st.session_state.target_wh = None  # Init None
# if "bg_list" not in st.session_state:  # Check background list
#     st.session_state.bg_list = []  # Init empty
# if "bg_names" not in st.session_state:  # Check background name list
#     st.session_state.bg_names = []  # Init empty
# if "uploaded_bg_list" not in st.session_state:  # Check upload backgrounds
#     st.session_state.uploaded_bg_list = []  # Init empty
# if "uploaded_bg_names" not in st.session_state:  # Check upload names
#     st.session_state.uploaded_bg_names = []  # Init empty
# if "bg_index" not in st.session_state:  # Check selected background index
#     st.session_state.bg_index = 0  # Default 0

# with st.sidebar:  # Sidebar section begin
#     st.subheader("Controls")  # Sidebar title

#     camera_index = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)  # Camera selector
#     cut_thr = st.slider("cutThreshold", 0.0, 1.0, 0.95, 0.01)  # Background removal threshold slider
#     fps_limit = st.slider("UI FPS limit", 5, 60, 24, 1)  # UI FPS limit
#     pad_value = st.slider("Letterbox padding (0=black)", 0, 255, 0, 1)  # Padding slider
#     st.divider()  # UI divider
#     run = st.toggle("Run", value=False)  # Camera start toggle
#     st.divider()  # UI divider
#     st.subheader("Add background")  # Upload section title

#     uploaded = st.file_uploader("Upload background image (jpg/png)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=False)  # Upload input
#     col_u1, col_u2 = st.columns(2)  # Two buttons row
#     add_upload_clicked = col_u1.button("Add upload", use_container_width=True)  # Add upload button
#     clear_clicked = col_u2.button("Clear added", use_container_width=True)  # Clear added backgrounds
#     st.markdown("---")  # Separator
#     bg_url = st.text_input("Or paste image URL", placeholder="https://.../image.jpg")  # URL input
#     add_url_clicked = st.button("Add URL", use_container_width=True)  # Add URL button

# col1, col2 = st.columns(2)  # Main UI columns
# ph_in = col1.empty()  # Input image placeholder
# ph_out = col2.empty()  # Output image placeholder

# def open_camera(idx: int):  # Function to open camera
#     return cv2.VideoCapture(idx)  # Open webcam

# if not run:  # If toggle off
#     if st.session_state.cap is not None:  # If camera open
#         st.session_state.cap.release()  # Release camera
#         st.session_state.cap = None  # Reset
#     st.info("Enable 'Run' from sidebar to start the camera.")  # Info
#     st.stop()  # Stop

# if st.session_state.cap is None:  # If camera not open
#     st.session_state.cap = open_camera(int(camera_index))  # Open camera

#     ok, first = st.session_state.cap.read()  # Read first frame
#     if not ok or first is None:  # If fail
#         st.error("Camera frame could not be read. Check camera index / permissions.")  # Error
#         st.session_state.cap.release()  # Release
#         st.session_state.cap = None  # Reset
#         st.stop()  # Stop

#     h, w = first.shape[:2]  # Get resolution
#     st.session_state.target_wh = (w, h)  # Store resolution
#     folder_bgs, folder_names = load_backgrounds_from_folder("images", w, h)  # Load folder BGs
#     st.session_state.bg_list = folder_bgs  # Store BGs
#     st.session_state.bg_names = folder_names  # Store names
#     st.session_state.bg_index = 0  # Reset BG index

# w, h = st.session_state.target_wh  # Unpack target resolution

# if clear_clicked:  # If clear pressed
#     st.session_state.uploaded_bg_list = []  # Clear uploaded BGs
#     st.session_state.uploaded_bg_names = []  # Clear names
#     st.session_state.bg_index = 0  # Reset index
#     st.success("Added backgrounds cleared.")  # Success message

# if add_upload_clicked:  # If add upload pressed
#     img_bgr = decode_uploaded_image_to_bgr(uploaded)  # Decode file to BGR
#     if img_bgr is None:  # If fail
#         st.warning("No image uploaded (or it could not be decoded).")  # Warn
#     else:
#         img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)  # Resize
#         name = getattr(uploaded, "name", "uploaded")  # Get name
#         name_disp = f"[UPLOAD] {name}"  # Tag name
#         st.session_state.uploaded_bg_list.append(img_bgr)  # Append BG
#         st.session_state.uploaded_bg_names.append(name_disp)  # Append name
#         st.success(f"Added upload: {name}")  # Success

# if add_url_clicked:  # If add URL pressed
#     if not bg_url.strip():  # If empty
#         st.warning("Please paste an image URL.")  # Warn
#     else:
#         try:
#             img_bgr = decode_url_image_to_bgr(bg_url.strip(), timeout_sec=10)  # Download + decode
#             if img_bgr is None:  # If fail
#                 st.error("Could not download/decode image from URL. Make sure it is a direct image link.")  # Error
#             else:
#                 img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)  # Resize
#                 short = bg_url.strip()  # Shorten for display
#                 if len(short) > 40:  # If long
#                     short = short[:37] + "..."  # Truncate
#                 st.session_state.uploaded_bg_list.append(img_bgr)  # Append URL BG
#                 st.session_state.uploaded_bg_names.append(f"[URL] {short}")  # Append name
#                 st.success("Added URL image.")  # Success
#         except Exception as e:  # Catch errors
#             st.error(f"URL download failed: {e}")  # Show error

# all_bgs = st.session_state.bg_list + st.session_state.uploaded_bg_list  # Merge BG lists
# all_names = st.session_state.bg_names + st.session_state.uploaded_bg_names  # Merge BG names

# if not all_bgs:  # If still empty
#     st.warning("No backgrounds found. Add images to ./images, upload, or add URL.")  # Warn
#     all_bgs = []  # Reset local
#     all_names = []  # Reset names

# if all_names:  # If BG names exist
#     st.session_state.bg_index = st.sidebar.selectbox("Background image", options=list(range(len(all_names))), format_func=lambda i: all_names[i], index=min(st.session_state.bg_index, len(all_names) - 1))  # Select BG

# delay = 1.0 / float(fps_limit)  # Convert FPS limit to delay

# while True:  # Main camera loop
#     ok, frame = st.session_state.cap.read()  # Capture camera frame
#     if not ok or frame is None:  # If fail
#         st.warning("Camera stream ended.")  # Warn
#         break  # Exit loop

#     frame = cv2.flip(frame, 1)  # Mirror flip
#     frame_lb = letterbox(frame, w, h, pad_value=pad_value)  # Fit frame into target resolution

#     if all_bgs:  # If backgrounds exist
#         bg = all_bgs[st.session_state.bg_index]  # Get selected BG
#         out = st.session_state.segmentor.removeBG(frame_lb, bg, cutThreshold=float(cut_thr))  # Remove BG
#     else:
#         out = frame_lb  # Fallback

#     ph_in.image(cv2.cvtColor(frame_lb, cv2.COLOR_BGR2RGB), caption="Input", use_container_width=True)  # Show input
#     ph_out.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="Output", use_container_width=True)  # Show output

#     time.sleep(1.0 / float(fps_limit))  # FPS delay

#     if not st.session_state.get("Run", True):  # If Run toggled off
#         break  # Exit

# # cap.release()  # Release camera resource
# cv2.destroyAllWindows()  # Destroy OpenCV windows




# URL ile arka kapak ekleme
# import os
# import time
# import cv2
# import numpy as np
# import streamlit as st
# from cvzone.SelfiSegmentationModule import SelfiSegmentation
# from urllib.request import urlopen, Request


# def letterbox(img, target_w, target_h, pad_value=0):
#     if img is None:
#         return None
#     h, w = img.shape[:2]
#     if h == 0 or w == 0:
#         return None

#     scale = min(target_w / w, target_h / h)
#     new_w = max(1, int(w * scale))
#     new_h = max(1, int(h * scale))

#     resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#     canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)

#     x = (target_w - new_w) // 2
#     y = (target_h - new_h) // 2
#     canvas[y:y + new_h, x:x + new_w] = resized
#     return canvas


# def load_backgrounds_from_folder(images_dir, target_w, target_h):
#     bg_list, bg_names = [], []
#     if not os.path.isdir(images_dir):
#         return bg_list, bg_names

#     for f in sorted(os.listdir(images_dir)):
#         path = os.path.join(images_dir, f)
#         if not os.path.isfile(path):
#             continue
#         bg = cv2.imread(path)
#         if bg is None:
#             continue
#         bg = cv2.resize(bg, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
#         bg_list.append(bg)
#         bg_names.append(f)
#     return bg_list, bg_names


# def decode_uploaded_image_to_bgr(uploaded_file):
#     if uploaded_file is None:
#         return None
#     bytes_data = uploaded_file.read()
#     if not bytes_data:
#         return None
#     buf = np.frombuffer(bytes_data, dtype=np.uint8)
#     img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR
#     return img


# def decode_url_image_to_bgr(url: str, timeout_sec: int = 10):
#     """
#     Downloads an image from URL and decodes into OpenCV BGR.
#     Supports common direct image URLs (jpg/png/webp).
#     """
#     url = (url or "").strip()
#     if not url:
#         return None

#     # Some servers block "Python-urllib"; set User-Agent
#     req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
#     with urlopen(req, timeout=timeout_sec) as resp:
#         data = resp.read()

#     if not data:
#         return None

#     buf = np.frombuffer(data, dtype=np.uint8)
#     img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
#     return img


# # -------------------------
# # Streamlit UI
# # -------------------------
# st.set_page_config(page_title="Background Removal (cvzone)", layout="wide")
# st.title("Camera Background Removal — Streamlit UI")

# # Session state init
# if "cap" not in st.session_state:
#     st.session_state.cap = None
# if "segmentor" not in st.session_state:
#     st.session_state.segmentor = SelfiSegmentation()
# if "target_wh" not in st.session_state:
#     st.session_state.target_wh = None
# if "bg_list" not in st.session_state:
#     st.session_state.bg_list = []
# if "bg_names" not in st.session_state:
#     st.session_state.bg_names = []
# if "uploaded_bg_list" not in st.session_state:
#     st.session_state.uploaded_bg_list = []
# if "uploaded_bg_names" not in st.session_state:
#     st.session_state.uploaded_bg_names = []
# if "bg_index" not in st.session_state:
#     st.session_state.bg_index = 0


# with st.sidebar:
#     st.subheader("Controls")

#     camera_index = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
#     cut_thr = st.slider("cutThreshold", 0.0, 1.0, 0.95, 0.01)
#     fps_limit = st.slider("UI FPS limit", 5, 60, 24, 1)
#     pad_value = st.slider("Letterbox padding (0=black)", 0, 255, 0, 1)

#     st.divider()
#     run = st.toggle("Run", value=False)

#     st.divider()
#     st.subheader("Add background")

#     # Upload
#     uploaded = st.file_uploader(
#         "Upload background image (jpg/png)",
#         type=["jpg", "jpeg", "png", "webp"],
#         accept_multiple_files=False
#     )

#     col_u1, col_u2 = st.columns(2)
#     add_upload_clicked = col_u1.button("Add upload", use_container_width=True)
#     clear_clicked = col_u2.button("Clear added", use_container_width=True)

#     st.markdown("---")
#     # URL input
#     bg_url = st.text_input("Or paste image URL", placeholder="https://.../image.jpg")
#     add_url_clicked = st.button("Add URL", use_container_width=True)


# # Placeholders
# col1, col2 = st.columns(2)
# ph_in = col1.empty()
# ph_out = col2.empty()


# def open_camera(idx: int):
#     return cv2.VideoCapture(idx)


# # Stop state: release camera if turned off
# if not run:
#     if st.session_state.cap is not None:
#         st.session_state.cap.release()
#         st.session_state.cap = None
#     st.info("Enable 'Run' from sidebar to start the camera.")
#     st.stop()

# # Run state: open camera if needed
# if st.session_state.cap is None:
#     st.session_state.cap = open_camera(int(camera_index))

#     ok, first = st.session_state.cap.read()
#     if not ok or first is None:
#         st.error("Camera frame could not be read. Check camera index / permissions.")
#         st.session_state.cap.release()
#         st.session_state.cap = None
#         st.stop()

#     h, w = first.shape[:2]
#     st.session_state.target_wh = (w, h)

#     # Load folder backgrounds once
#     folder_bgs, folder_names = load_backgrounds_from_folder("images", w, h)
#     st.session_state.bg_list = folder_bgs
#     st.session_state.bg_names = folder_names
#     st.session_state.bg_index = 0

# # Target resolution
# w, h = st.session_state.target_wh

# # Clear added (uploads + urls)
# if clear_clicked:
#     st.session_state.uploaded_bg_list = []
#     st.session_state.uploaded_bg_names = []
#     st.session_state.bg_index = 0
#     st.success("Added backgrounds cleared.")

# # Add upload
# if add_upload_clicked:
#     img_bgr = decode_uploaded_image_to_bgr(uploaded)
#     if img_bgr is None:
#         st.warning("No image uploaded (or it could not be decoded).")
#     else:
#         img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
#         name = getattr(uploaded, "name", "uploaded")
#         name_disp = f"[UPLOAD] {name}"
#         st.session_state.uploaded_bg_list.append(img_bgr)
#         st.session_state.uploaded_bg_names.append(name_disp)
#         st.success(f"Added upload: {name}")

# # Add URL
# if add_url_clicked:
#     if not bg_url.strip():
#         st.warning("Please paste an image URL.")
#     else:
#         try:
#             img_bgr = decode_url_image_to_bgr(bg_url.strip(), timeout_sec=10)
#             if img_bgr is None:
#                 st.error("Could not download/decode image from URL. Make sure it is a direct image link.")
#             else:
#                 img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
#                 short = bg_url.strip()
#                 if len(short) > 40:
#                     short = short[:37] + "..."
#                 st.session_state.uploaded_bg_list.append(img_bgr)
#                 st.session_state.uploaded_bg_names.append(f"[URL] {short}")
#                 st.success("Added URL image.")
#         except Exception as e:
#             st.error(f"URL download failed: {e}")

# # Merge backgrounds: folder + added
# all_bgs = st.session_state.bg_list + st.session_state.uploaded_bg_list
# all_names = st.session_state.bg_names + st.session_state.uploaded_bg_names

# if not all_bgs:
#     st.warning("No backgrounds found. Add images to ./images, upload, or add URL.")
#     all_bgs = []
#     all_names = []

# # Background selector
# if all_names:
#     st.session_state.bg_index = st.sidebar.selectbox(
#         "Background image",
#         options=list(range(len(all_names))),
#         format_func=lambda i: all_names[i],
#         index=min(st.session_state.bg_index, len(all_names) - 1),
#     )

# delay = 1.0 / float(fps_limit)

# # Main loop
# while True:
#     ok, frame = st.session_state.cap.read()
#     if not ok or frame is None:
#         st.warning("Camera stream ended.")
#         break

#     frame = cv2.flip(frame, 1)
#     frame_lb = letterbox(frame, w, h, pad_value=pad_value)

#     if all_bgs:
#         bg = all_bgs[st.session_state.bg_index]
#         out = st.session_state.segmentor.removeBG(frame_lb, bg, cutThreshold=float(cut_thr))
#     else:
#         out = frame_lb

#     ph_in.image(cv2.cvtColor(frame_lb, cv2.COLOR_BGR2RGB), caption="Input", use_container_width=True)
#     ph_out.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="Output", use_container_width=True)

#     time.sleep(delay)

#     if not st.session_state.get("Run", True):
#         break


# import os
# import time
# import cv2
# import numpy as np
# import streamlit as st
# from cvzone.SelfiSegmentationModule import SelfiSegmentation


# def letterbox(img, target_w, target_h, pad_value=0):
#     if img is None:
#         return None
#     h, w = img.shape[:2]
#     if h == 0 or w == 0:
#         return None

#     scale = min(target_w / w, target_h / h)
#     new_w = max(1, int(w * scale))
#     new_h = max(1, int(h * scale))

#     resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#     canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)

#     x = (target_w - new_w) // 2
#     y = (target_h - new_h) // 2
#     canvas[y:y + new_h, x:x + new_w] = resized
#     return canvas


# def load_backgrounds_from_folder(images_dir, target_w, target_h):
#     bg_list, bg_names = [], []
#     if not os.path.isdir(images_dir):
#         return bg_list, bg_names

#     for f in sorted(os.listdir(images_dir)):
#         path = os.path.join(images_dir, f)
#         if not os.path.isfile(path):
#             continue
#         bg = cv2.imread(path)
#         if bg is None:
#             continue
#         # hız için direct resize
#         bg = cv2.resize(bg, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
#         bg_list.append(bg)
#         bg_names.append(f)
#     return bg_list, bg_names


# def decode_uploaded_image_to_bgr(uploaded_file):
#     """
#     Streamlit uploaded_file -> OpenCV BGR image
#     """
#     if uploaded_file is None:
#         return None
#     bytes_data = uploaded_file.read()
#     if not bytes_data:
#         return None
#     np_buf = np.frombuffer(bytes_data, dtype=np.uint8)
#     img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)  # BGR
#     return img


# # -------------------------
# # Streamlit UI
# # -------------------------
# st.set_page_config(page_title="Background Removal (cvzone)", layout="wide")
# st.title("Camera Background Removal — Streamlit UI")

# # Session state init
# if "cap" not in st.session_state:
#     st.session_state.cap = None
# if "segmentor" not in st.session_state:
#     st.session_state.segmentor = SelfiSegmentation()
# if "target_wh" not in st.session_state:
#     st.session_state.target_wh = None
# if "bg_list" not in st.session_state:
#     st.session_state.bg_list = []
# if "bg_names" not in st.session_state:
#     st.session_state.bg_names = []
# if "uploaded_bg_list" not in st.session_state:
#     st.session_state.uploaded_bg_list = []
# if "uploaded_bg_names" not in st.session_state:
#     st.session_state.uploaded_bg_names = []
# if "bg_index" not in st.session_state:
#     st.session_state.bg_index = 0


# with st.sidebar:
#     st.subheader("Controls")

#     camera_index = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
#     cut_thr = st.slider("cutThreshold", 0.0, 1.0, 0.95, 0.01)
#     fps_limit = st.slider("UI FPS limit", 5, 60, 24, 1)
#     pad_value = st.slider("Letterbox padding (0=black)", 0, 255, 0, 1)

#     st.divider()
#     run = st.toggle("Run", value=False)

#     st.divider()
#     st.subheader("Add background (drag & drop)")

#     uploaded = st.file_uploader(
#         "Upload background image (jpg/png)",
#         type=["jpg", "jpeg", "png"],
#         accept_multiple_files=False
#     )

#     col_u1, col_u2 = st.columns(2)
#     add_clicked = col_u1.button("Add to list", use_container_width=True)
#     clear_clicked = col_u2.button("Clear uploaded", use_container_width=True)

# # Placeholders
# col1, col2 = st.columns(2)
# ph_in = col1.empty()
# ph_out = col2.empty()


# def open_camera(idx: int):
#     return cv2.VideoCapture(idx)


# # Stop state: release camera if turned off
# if not run:
#     if st.session_state.cap is not None:
#         st.session_state.cap.release()
#         st.session_state.cap = None
#     st.info("Enable 'Run' from sidebar to start the camera.")
#     st.stop()

# # Run state: open camera if needed
# if st.session_state.cap is None:
#     st.session_state.cap = open_camera(int(camera_index))

#     ok, first = st.session_state.cap.read()
#     if not ok or first is None:
#         st.error("Camera frame could not be read. Check camera index / permissions.")
#         st.session_state.cap.release()
#         st.session_state.cap = None
#         st.stop()

#     h, w = first.shape[:2]
#     st.session_state.target_wh = (w, h)

#     # Load folder backgrounds
#     folder_bgs, folder_names = load_backgrounds_from_folder("images", w, h)
#     st.session_state.bg_list = folder_bgs
#     st.session_state.bg_names = folder_names

#     st.session_state.bg_index = 0

# # Handle uploaded backgrounds
# w, h = st.session_state.target_wh

# if clear_clicked:
#     st.session_state.uploaded_bg_list = []
#     st.session_state.uploaded_bg_names = []
#     st.session_state.bg_index = 0
#     st.success("Uploaded backgrounds cleared.")

# if add_clicked:
#     img_bgr = decode_uploaded_image_to_bgr(uploaded)
#     if img_bgr is None:
#         st.warning("No image uploaded (or it could not be decoded).")
#     else:
#         # Resize to target
#         img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
#         # Name
#         name = getattr(uploaded, "name", "uploaded")
#         # If same name, make unique
#         base = name
#         i = 1
#         while name in st.session_state.uploaded_bg_names:
#             name = f"{os.path.splitext(base)[0]}_{i}{os.path.splitext(base)[1]}"
#             i += 1

#         st.session_state.uploaded_bg_list.append(img_bgr)
#         st.session_state.uploaded_bg_names.append(f"[UPLOAD] {name}")
#         st.success(f"Added: {name}")

# # Merge backgrounds: folder + uploaded
# all_bgs = st.session_state.bg_list + st.session_state.uploaded_bg_list
# all_names = st.session_state.bg_names + st.session_state.uploaded_bg_names

# if not all_bgs:
#     st.warning("No backgrounds found. Add images to ./images or upload one from sidebar.")
#     # Fallback: show camera only

# # Background selector
# if all_names:
#     st.session_state.bg_index = st.sidebar.selectbox(
#         "Background image",
#         options=list(range(len(all_names))),
#         format_func=lambda i: all_names[i],
#         index=min(st.session_state.bg_index, len(all_names) - 1),
#     )

# delay = 1.0 / float(fps_limit)

# # Main loop
# while True:
#     # Streamlit reruns on interaction; keep loop simple
#     ok, frame = st.session_state.cap.read()
#     if not ok or frame is None:
#         st.warning("Camera stream ended.")
#         break

#     frame = cv2.flip(frame, 1)
#     frame_lb = letterbox(frame, w, h, pad_value=pad_value)

#     if all_bgs:
#         bg = all_bgs[st.session_state.bg_index]
#         out = st.session_state.segmentor.removeBG(frame_lb, bg, cutThreshold=float(cut_thr))
#     else:
#         out = frame_lb

#     ph_in.image(cv2.cvtColor(frame_lb, cv2.COLOR_BGR2RGB), caption="Input", use_container_width=True)
#     ph_out.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="Output", use_container_width=True)

#     time.sleep(delay)

#     # If user toggles Run off, Streamlit will rerun; break defensively
#     if not st.session_state.get("Run", True):
#         break



# # import os
# # import time
# # import cv2
# # import numpy as np
# # import streamlit as st
# # from cvzone.SelfiSegmentationModule import SelfiSegmentation

# # # -------------------------
# # # Helpers
# # # -------------------------
# # def letterbox(img, target_w, target_h, pad_value=0):
# #     if img is None:
# #         return None
# #     h, w = img.shape[:2]
# #     if h == 0 or w == 0:
# #         return None

# #     scale = min(target_w / w, target_h / h)
# #     new_w = max(1, int(w * scale))
# #     new_h = max(1, int(h * scale))

# #     resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
# #     canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)

# #     x = (target_w - new_w) // 2
# #     y = (target_h - new_h) // 2
# #     canvas[y:y + new_h, x:x + new_w] = resized
# #     return canvas

# # def load_backgrounds(images_dir, target_w, target_h):
# #     bg_list = []
# #     bg_names = []
# #     if not os.path.isdir(images_dir):
# #         return bg_list, bg_names

# #     for f in sorted(os.listdir(images_dir)):
# #         path = os.path.join(images_dir, f)
# #         if not os.path.isfile(path):
# #             continue
# #         bg = cv2.imread(path)
# #         if bg is None:
# #             continue
# #         # Hız için doğrudan resize (letterbox istersen değiştiririz)
# #         bg = cv2.resize(bg, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
# #         bg_list.append(bg)
# #         bg_names.append(f)

# #     return bg_list, bg_names

# # # -------------------------
# # # Streamlit UI
# # # -------------------------
# # st.set_page_config(page_title="Background Removal (cvzone)", layout="wide")
# # st.title("Camera Background Removal — Streamlit UI")

# # with st.sidebar:
# #     st.subheader("Controls")
# #     camera_index = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)
# #     cut_thr = st.slider("cutThreshold", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
# #     fps_limit = st.slider("UI FPS limit", min_value=5, max_value=60, value=24, step=1)
# #     pad_value = st.slider("Letterbox padding (0=black)", min_value=0, max_value=255, value=0, step=1)

# #     st.divider()
# #     run = st.toggle("Run", value=False)

# # # Placeholders
# # col1, col2 = st.columns(2)
# # ph_in = col1.empty()
# # ph_out = col2.empty()

# # # -------------------------
# # # Camera / model init (cached in session_state)
# # # -------------------------
# # if "cap" not in st.session_state:
# #     st.session_state.cap = None
# # if "segmentor" not in st.session_state:
# #     st.session_state.segmentor = SelfiSegmentation()
# # if "target_wh" not in st.session_state:
# #     st.session_state.target_wh = None
# # if "bg_list" not in st.session_state:
# #     st.session_state.bg_list = []
# # if "bg_names" not in st.session_state:
# #     st.session_state.bg_names = []
# # if "bg_index" not in st.session_state:
# #     st.session_state.bg_index = 0

# # def open_camera(idx):
# #     cap = cv2.VideoCapture(idx)
# #     return cap

# # # Open camera and determine target resolution once Run is enabled
# # if run:
# #     if st.session_state.cap is None:
# #         st.session_state.cap = open_camera(int(camera_index))

# #         ok, first = st.session_state.cap.read()
# #         if not ok or first is None:
# #             st.error("Camera frame could not be read. Check camera index / permissions.")
# #             st.session_state.cap.release()
# #             st.session_state.cap = None
# #             st.stop()

# #         # Target resolution = first frame resolution (no cap.set)
# #         h, w = first.shape[:2]
# #         st.session_state.target_wh = (w, h)

# #         # Load backgrounds resized to target
# #         bg_list, bg_names = load_backgrounds("images", w, h)
# #         if not bg_list:
# #             st.warning("No valid images found in ./images. Output will show original frame only.")
# #         st.session_state.bg_list = bg_list
# #         st.session_state.bg_names = bg_names
# #         st.session_state.bg_index = 0

# #     # Background selector
# #     if st.session_state.bg_names:
# #         st.session_state.bg_index = st.sidebar.selectbox(
# #             "Background image",
# #             options=list(range(len(st.session_state.bg_names))),
# #             format_func=lambda i: st.session_state.bg_names[i],
# #             index=st.session_state.bg_index
# #         )
# #     else:
# #         st.sidebar.info("Put background images into ./images folder.")

# #     # Main loop (runs until toggle is turned off; Streamlit reruns script on interactions)
# #     w, h = st.session_state.target_wh
# #     delay = 1.0 / float(fps_limit)

# #     while run:
# #         ok, frame = st.session_state.cap.read()
# #         if not ok or frame is None:
# #             st.warning("Camera stream ended.")
# #             break

# #         frame = cv2.flip(frame, 1)
# #         frame_lb = letterbox(frame, w, h, pad_value=pad_value)

# #         if st.session_state.bg_list:
# #             bg = st.session_state.bg_list[st.session_state.bg_index]
# #             out = st.session_state.segmentor.removeBG(frame_lb, bg, cutThreshold=float(cut_thr))
# #         else:
# #             out = frame_lb

# #         # Streamlit expects RGB
# #         ph_in.image(cv2.cvtColor(frame_lb, cv2.COLOR_BGR2RGB), caption="Input", use_container_width=True)
# #         ph_out.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="Output", use_container_width=True)

# #         # Allow UI updates
# #         time.sleep(delay)

# #         # IMPORTANT: Re-read toggle value (Streamlit reruns script on interaction)
# #         run = st.session_state.get("Run", True)

# # else:
# #     # Stop and release camera if it was open
# #     if st.session_state.cap is not None:
# #         st.session_state.cap.release()
# #         st.session_state.cap = None
# #     st.info("Enable 'Run' from sidebar to start the camera.")
