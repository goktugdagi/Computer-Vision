import os
import pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np

def ensure_dir(data_dir: Path) -> None:
    """
        Create storage directly if it does not exist.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

def sanitize_name(name: str) -> str:
    """
        Make a Windows-safe filename-like label.
    """
    name = name.strip()
    forbidden = '<>:"/\\|?*'
    for ch in forbidden:
        name = name.replace(ch, "")
    name = name.replace(" ","_")
    return name

def encoding_path(data_dir: Path, person_name: str) -> Path:
    return data_dir / f"{person_name}.pkl"

def save_encoding(data_dir: Path, person_name: str, encoding: np.ndarray, overwrite: bool = True) -> Path:
    """
        Save a single 128-d face encodşng to disk(pickle).
    """
    ensure_dir(data_dir)
    person_name = sanitize_name(person_name)
    p = encoding_path(data_dir, person_name)

    if p.exists() and not overwrite:
        raise FileExistsError(f"'{person_name}' already exists")
    
    with open(p, "wb") as f:
        pickle.dump(encoding, f)

    return p 

def delete_encoding(data_dir: Path, person_name: str) -> bool:
    """
        Delete a person's encoding file if it exists.
    """
    ensure_dir(data_dir)
    person_name = sanitize_name(person_name)
    p = encoding_path(data_dir, person_name)

    if p.exists():
        p.unlink()
        return True
    return False

def list_people(data_dir: Path) -> List[str]:
    """
        List all enrolled people.
    """
    ensure_dir(data_dir)
    people: List[str] = []

    for fn in os.listdir(data_dir):
        if fn.endswith(".pkl"):
            people.append(Path(fn).stem)

    people.sort()
    return people

def load_database(data_dir: Path) -> Tuple[list[str], np.ndarray]:
    """
        Load all known encodings.

        Returns:
            names: list[str]
            encs: np.ndarray of shape (N, 128)
    """
    ensure_dir(data_dir)
    names: List[str] = []
    enc_list: List[np.ndarray] = []

    for fn in os.listdir(data_dir):
        if not fn.endswith(".pkl"):
            continue

        name = Path(fn).stem
        with open(data_dir / fn, "rb") as f:
            enc = pickle.load(f)

        names.append(name)
        enc_list.append(enc)
    
    if not enc_list:
        return [], np.empty((0, 128), dtype=np.float64)
    
    return names, np.vstack(enc_list)


