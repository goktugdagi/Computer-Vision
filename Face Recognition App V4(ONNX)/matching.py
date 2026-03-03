from typing import Dict, List, Tuple, Optional
import numpy as np


def l2norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    n = float(np.linalg.norm(x) + 1e-9)
    return x / n


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = l2norm(a)
    b = l2norm(b)
    return float(1.0 - float(np.dot(a, b)))


def find_best_match(
    query_emb: np.ndarray,
    db: Dict[str, List[np.ndarray]],
) -> Tuple[Optional[str], float]:
    best_name = None
    best_dist = 999.0

    q = l2norm(query_emb)

    for name, embs in db.items():
        for e in embs:
            d = cosine_distance(q, e)
            if d < best_dist:
                best_dist = d
                best_name = name

    return best_name, best_dist