import cv2
import threading
import time
from typing import Optional, Tuple


class Camera:
    def __init__(self, index: int, size: Tuple[int, int]):
        self.index = index
        self.width, self.height = size

        self.cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._frame = None
        self._last_ts = 0.0

    def start(self):
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera open failed (index={self.index})")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
                    self._last_ts = time.time()
            else:
                time.sleep(0.01)

    def read(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def last_timestamp(self) -> float:
        with self._lock:
            return float(self._last_ts)