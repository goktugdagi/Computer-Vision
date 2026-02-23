import threading
import time
from typing import Optional
import cv2
import numpy as np


class CameraManager:
    """
    Single-owner camera reader.
    - Opens one VideoCapture
    - Continuously reads frames in a background thread
    - Stores the latest frame (BGR)
    """

    def __init__(self, camera_index: int = 0, target_fps: int = 25):
        self.camera_index = camera_index
        self.target_fps = max(1, int(target_fps))
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop.clear()
        self._cap = cv2.VideoCapture(self.camera_index)

        # Try to reduce buffering latency (not always honored)
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not self._cap.isOpened():
            raise RuntimeError("Could not open camera. Check camera index / permissions.")

        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._thread = None

        with self._lock:
            self._last_frame = None

    def _reader_loop(self) -> None:
        sleep_sec = 1.0 / float(self.target_fps)

        while not self._stop.is_set():
            if self._cap is None:
                time.sleep(0.1)
                continue

            ok, frame = self._cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._last_frame = frame
            time.sleep(sleep_sec)

    def get_frame(self) -> Optional[np.ndarray]:
        """Returns a copy of the latest frame (BGR)."""
        with self._lock:
            if self._last_frame is None:
                return None
            return self._last_frame.copy()

    def get_jpeg(self, quality: int = 80) -> Optional[bytes]:
        frame = self.get_frame()
        if frame is None:
            return None

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None
        return buf.tobytes()