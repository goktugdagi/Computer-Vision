import cv2 
import numpy as np
import face_recognition

def decode_image_bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    """
        Decode raw image bytes into OpenCV BGR image
    """

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data/file.")
    return img

def bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """
        Convert OpenCV BGR to RGB (face_recognition expects RGB).
    """
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

def detect_faces(rgb_image: np.ndarray, model: str = "hog"):
    """
        Return face locations in (top, right, bottom, left) format.
    """

    return face_recognition.face_locations(rgb_image, model=model)

def encode_faces(rgb_image: np.ndarray, locations):
    """
        Return 128-d embeddings for the given face locations.
    """
    return face_recognition.face_encodings(rgb_image, locations)
