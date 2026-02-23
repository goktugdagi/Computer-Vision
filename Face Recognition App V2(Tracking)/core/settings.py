from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    # Storage
    data_dir: Path = Path("known_faces")

    # Recognition
    tolerance: float = 0.55
    face_model: str = "hog"  # "hog" (fast CPU) or "cnn" (heavier)

    # Performance (V2)
    resize_scale: float = 0.6          # detection/encoding uses smaller frame
    detect_every_n_frames: int = 6     # run heavy detection every N frames
    jpeg_quality: int = 60             # MJPEG quality (lower => faster)
    stream_sleep_sec: float = 0.01

    # Tracking
    tracker_type: str = "CSRT"         # "CSRT" or "KCF"
    min_face_size: int = 35

    # Camera
    camera_index: int = 0
    camera_target_fps: int = 25        # camera reader thread pace

SETTINGS = Settings()