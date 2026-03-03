from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    # Server
    host: str = "127.0.0.1"
    port: int = 8000

    # Camera
    camera_index: int = 0
    camera_width: int = 960
    camera_height: int = 540

    # Matching
    match_threshold: float = 0.50  # cosine distance (lower is stricter)

    # Storage
    db_dir: str = "known_faces_v4"

    # InsightFace / ONNX (SCRFD + ArcFace)
    det_size_w: int = 640
    det_size_h: int = 640
    det_thresh: float = 0.5

    # GPU: 0 -> use GPU, -1 -> CPU
    ctx_id: int = -1  # default CPU


SETTINGS = Settings()
DB_PATH = Path(SETTINGS.db_dir)
DB_PATH.mkdir(parents=True, exist_ok=True)