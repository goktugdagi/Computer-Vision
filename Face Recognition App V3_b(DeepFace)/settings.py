from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class Settings:
    # Storage
    data_dir: Path = Path("known_faces_v3")

    # Camera
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30

    # InsightFace
    det_size: Tuple[int, int] = (640, 640)
    min_det_score: float = 0.50

    # Matching (cosine distance): lower is stricter.
    match_threshold: float = 0.45

    # Prefer GPU if available; auto fallback to CPU if CUDA runtime missing
    prefer_gpu: bool = True


SETTINGS = Settings()