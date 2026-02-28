from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from insightface.app import FaceAnalysis

from .settings import SETTINGS


@dataclass
class DetectedFace:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    det_score: float
    embedding: np.ndarray  # (512,)


class InsightFaceEngine:
    """
    InsightFace FaceAnalysis:
      - detector
      - ArcFace recognizer

    GPU/CPU selection:
      - Try GPU if prefer_gpu=True
      - If CUDA provider fails, fallback to CPU.
    """

    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l")

        if SETTINGS.prefer_gpu:
            try:
                self.app.prepare(ctx_id=0, det_size=SETTINGS.det_size)  # GPU
                self._backend = "GPU"
                return
            except Exception:
                pass

        self.app.prepare(ctx_id=-1, det_size=SETTINGS.det_size)  # CPU
        self._backend = "CPU"

    @property
    def backend(self) -> str:
        return self._backend

    def detect_and_embed(self, bgr: np.ndarray) -> List[DetectedFace]:
        faces = self.app.get(bgr)

        out: List[DetectedFace] = []
        for f in faces:
            score = float(getattr(f, "det_score", 1.0))
            if score < SETTINGS.min_det_score:
                continue

            x1, y1, x2, y2 = [int(v) for v in f.bbox]

            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                emb = getattr(f, "embedding", None)
            if emb is None:
                continue

            emb = np.asarray(emb, dtype=np.float32)
            out.append(DetectedFace(bbox=(x1, y1, x2, y2), det_score=score, embedding=emb))

        return out