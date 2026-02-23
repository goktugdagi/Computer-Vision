from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import face_recognition


@dataclass
class Match:
    name: str
    distance: float


def recognize_one(
    known_names: List[str],
    known_encodings: np.ndarray,
    query_encoding: np.ndarray,
    tolerance: float,
) -> Optional[Match]:
    if known_encodings.shape[0] == 0:
        return None

    distances = face_recognition.face_distance(known_encodings, query_encoding)
    best_idx = int(distances.argmin())
    best_dist = float(distances[best_idx])

    if best_dist <= tolerance:
        return Match(name=known_names[best_idx], distance=best_dist)

    return None