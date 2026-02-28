import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np


def ensure_dir(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)


def sanitize_name(name: str) -> str:
    name = name.strip()
    forbidden = '<>:"/\\|?*'
    for ch in forbidden:
        name = name.replace(ch, "")
    name = name.replace(" ", "_")
    return name


def encoding_path(data_dir: Path, person_name: str) -> Path:
    return data_dir / f"{person_name}.pkl"


def save_encoding(data_dir: Path, person_name: str, embedding: np.ndarray, overwrite: bool = True) -> Path:
    ensure_dir(data_dir)
    person_name = sanitize_name(person_name)
    p = encoding_path(data_dir, person_name)

    if p.exists() and not overwrite:
        raise FileExistsError(f"'{person_name}' already exists")

    with open(p, "wb") as f:
        pickle.dump(np.asarray(embedding, dtype=np.float32), f)

    return p


def delete_encoding(data_dir: Path, person_name: str) -> bool:
    ensure_dir(data_dir)
    person_name = sanitize_name(person_name)
    p = encoding_path(data_dir, person_name)
    if p.exists():
        p.unlink()
        return True
    return False


def list_people(data_dir: Path) -> List[str]:
    ensure_dir(data_dir)
    return [p.stem for p in sorted(data_dir.glob("*.pkl"))]


def load_database(data_dir: Path) -> Tuple[List[str], np.ndarray]:
    """
    Loads all embeddings into memory.
    Returns (names, embs) where embs shape is (N, 512)
    """
    ensure_dir(data_dir)
    names: List[str] = []
    embs: List[np.ndarray] = []

    for p in sorted(data_dir.glob("*.pkl")):
        try:
            with open(p, "rb") as f:
                emb = pickle.load(f)
            emb = np.asarray(emb, dtype=np.float32)
            if emb.ndim != 1:
                continue
            names.append(p.stem)
            embs.append(emb)
        except Exception:
            continue

    if len(embs) == 0:
        return [], np.empty((0, 512), dtype=np.float32)

    return names, np.stack(embs, axis=0).astype(np.float32)