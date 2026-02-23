import cv2
import numpy as np
import face_recognition


def decode_image_bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data/file.")
    return img


def bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def detect_faces(rgb_image: np.ndarray, model: str = "hog"):
    return face_recognition.face_locations(rgb_image, model=model)


def encode_faces(rgb_image: np.ndarray, locations):
    return face_recognition.face_encodings(rgb_image, locations)