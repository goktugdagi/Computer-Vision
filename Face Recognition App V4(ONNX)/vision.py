import cv2
import numpy as np


def encode_jpg(bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Could not encode jpg")
    return buf.tobytes()


def draw_box_label(bgr: np.ndarray, bbox, label: str):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if label:
        cv2.putText(
            bgr,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )