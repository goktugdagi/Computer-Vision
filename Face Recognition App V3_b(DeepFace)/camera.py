import threading
import time
from typing import Optional

import cv2
import numpy as np

from .settings import SETTINGS


class CameraManager:
    """
    Single owner for the webcam (backend owns the camera).
    A background thread continuously grabs frames; API endpoints read latest frame.
    """

    def __init__(self, index: int):
        self.index = index
        self.cap: Optional[cv2.VideoCapture] = None
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.last_frame: Optional[np.ndarray] = None

    def start(self) -> None:
        if self.running:
            return

        # CAP_DSHOW helps many Windows drivers
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, SETTINGS.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SETTINGS.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, SETTINGS.camera_fps)

        if not self.cap.isOpened():
            raise RuntimeError("Camera could not be opened.")

        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.thread = None
        self.last_frame = None

    def _loop(self) -> None:
        while self.running and self.cap is not None:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.last_frame = frame
            else:
                time.sleep(0.01)

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.last_frame is None:
                return None
            return self.last_frame.copy()

    def snapshot(self) -> np.ndarray:
        frame = self.get_frame()
        if frame is None:
            raise RuntimeError("No frame available yet.")
        return frame