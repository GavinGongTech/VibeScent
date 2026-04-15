from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_dataframe(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_dataframe(path: str | Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def save_embeddings(path: str | Path, matrix: np.ndarray) -> None:
    np.save(path, matrix.astype(np.float32))


def load_embeddings(path: str | Path) -> np.ndarray:
    return np.load(path)


def guess_mime_type(path: str | Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type:
        raise ValueError(f"Could not guess MIME type for {path}")
    return mime_type
