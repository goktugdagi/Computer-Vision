from typing import List, Tuple

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    cosine distance = 1 - cosine similarity
    a: (D,) or (N,D)
    b: (M,D) or (D,)
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]

    a = l2_normalize(a)
    b = l2_normalize(b)

    sim = a @ b.T  # (N,M)
    dist = 1.0 - sim
    return dist


def match_one(
    emb: np.ndarray, known_names: List[str], known_embs: np.ndarray, threshold: float
) -> Tuple[str, float]:
    """
    Returns (best_name or 'Unknown', best_distance)
    """
    if len(known_names) == 0 or known_embs.size == 0:
        return "Unknown", 1.0

    dists = cosine_distance(emb, known_embs)[0]
    idx = int(np.argmin(dists))
    best_dist = float(dists[idx])
    best_name = known_names[idx] if best_dist <= threshold else "Unknown"
    return best_name, best_dist