from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import numpy as np


def l2norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    n = float(np.linalg.norm(x) + 1e-9)
    return x / n


class FaceStore:
    """
    Disk store:
      known_faces_v4/
        index.json
        <name>__0.npy
        <name>__1.npy
        ...
    """
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.index_path = base_dir / "index.json"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self._save_index({"people": {}})

    def _load_index(self) -> Dict:
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save_index(self, data: Dict):
        self.index_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def list_people(self) -> List[str]:
        idx = self._load_index()
        return sorted(idx.get("people", {}).keys())

    def delete_person(self, name: str) -> bool:
        idx = self._load_index()
        people = idx.get("people", {})
        if name not in people:
            return False

        files = people[name].get("files", [])
        for f in files:
            p = self.base_dir / f
            if p.exists():
                p.unlink()

        people.pop(name, None)
        idx["people"] = people
        self._save_index(idx)
        return True

    def add_embedding(self, name: str, embedding: np.ndarray) -> int:
        idx = self._load_index()
        people = idx.get("people", {})
        if name not in people:
            people[name] = {"files": []}

        files = people[name]["files"]
        sample_id = len(files)
        fname = f"{name}__{sample_id}.npy"

        emb = l2norm(embedding)
        np.save(self.base_dir / fname, emb.astype(np.float32))
        files.append(fname)

        idx["people"] = people
        self._save_index(idx)
        return sample_id

    def load_all(self) -> Dict[str, List[np.ndarray]]:
        idx = self._load_index()
        out: Dict[str, List[np.ndarray]] = {}

        for name, meta in idx.get("people", {}).items():
            embs: List[np.ndarray] = []
            for f in meta.get("files", []):
                p = self.base_dir / f
                if p.exists():
                    e = np.load(p).astype(np.float32)
                    e = l2norm(e)  # normalize on load for robustness
                    embs.append(e)
            if embs:
                out[name] = embs

        return out

    def count(self) -> int:
        return len(self.list_people())