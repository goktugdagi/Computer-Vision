from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from insightface.app import FaceAnalysis

from .settings import SETTINGS


@dataclass
class DetectedFace:
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2
    det_score: float
    embedding: np.ndarray


class InsightFaceEngineV4:
    """
    V4 engine:
      - Detector: SCRFD (from insightface)
      - Embedder: ArcFace (from insightface)
      - Runtime: ONNX (onnxruntime)
    """
    def __init__(self):
        self._backend = "InsightFace (SCRFD+ArcFace / ONNX)"

        self.app = FaceAnalysis(
            name="buffalo_l",  # standard pack: scrfd + arcface
            providers=None,    # let insightface pick default providers
        )
        self.app.prepare(
            ctx_id=SETTINGS.ctx_id,
            det_size=(SETTINGS.det_size_w, SETTINGS.det_size_h),
            det_thresh=float(SETTINGS.det_thresh),
        )

    @property
    def backend(self) -> str:
        return self._backend

    def detect_and_embed(self, bgr: np.ndarray) -> List[DetectedFace]:
        faces = self.app.get(bgr)
        out: List[DetectedFace] = []

        for f in faces:
            # f.bbox: [x1,y1,x2,y2]
            bb = f.bbox.astype(int).tolist()
            x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])

            det_score = float(getattr(f, "det_score", 1.0))
            emb = np.asarray(f.embedding, dtype=np.float32)

            out.append(
                DetectedFace(
                    bbox=(x1, y1, x2, y2),
                    det_score=det_score,
                    embedding=emb,
                )
            )

        return out