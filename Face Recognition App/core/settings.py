from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    """
        Central settings for the application. 
    """
    data_dir: Path = Path("known_faces") # where face encodings are stored

    # Accuracy
    tolerance: float = 0.55 # lower = stricter matching
    model: str = "hog" # "hog" (fast CPU) or "cnn" (more accuracy, heavy)

SETTINGS = Settings()