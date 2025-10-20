# external_src/eval/io_utils.py
from __future__ import annotations
import os, ast, json, math, warnings
from pathlib import Path
from typing import Tuple, Optional, Iterable, Dict, Any, List
import glob
import re
import numpy as np
import pandas as pd

def _to_np(arr):
    if isinstance(arr, np.ndarray):
        return arr
    if isinstance(arr, (list, tuple)):
        try:
            return np.asarray(arr, dtype=np.float32)
        except Exception:
            return np.asarray(arr)
    if isinstance(arr, (bytes, bytearray, memoryview)):
        try:
            out = np.frombuffer(arr, dtype=np.float32)
            return out
        except Exception:
            pass
    if isinstance(arr, str):
        try:
            return np.asarray(ast.literal_eval(arr), dtype=np.float32)
        except Exception:
            pass
    return np.asarray(arr)

def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = logits - logits.max(axis=axis, keepdims=True)
    exps = np.exp(logits)
    return exps / exps.sum(axis=axis, keepdims=True)

def load_preds(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load predictions dumped by your evaluators.

    Supported:
    - Parquet with columns: 'labels' or 'y_true' or 'label'
      and one of: 'logits' or 'probs' (as vectors/strings/bytes)
    - Directory containing logits.npy and labels.npy
    - NPZ with 'logits' and 'labels'
    Returns: (labels[N], logits[N,C])  (if only probs present, convert to logits via log)
    """
    path = Path(path)
    if path.is_dir():
        lp = path / "logits.npy"
        yp = path / "labels.npy"
        if lp.exists() and yp.exists():
            logits = np.load(lp)
            labels = np.load(yp)
            return labels.astype(np.int64), logits.astype(np.float32)
        npz = next(path.glob("*.npz"), None)
        if npz:
            data = np.load(npz)
            logits = np.asarray(data["logits"], dtype=np.float32)
            labels = np.asarray(data["labels"], dtype=np.int64)
            return labels, logits
        # maybe the parquet sits here
        pqs = list(path.glob("*.parquet"))
        if pqs:
            return load_preds(pqs[0])

    if path.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            raise RuntimeError(f"Failed to read parquet: {path}\n{e}")
        # label column
        label_col = None
        for cand in ["labels", "y_true", "label", "targets", "y"]:
            if cand in df.columns:
                label_col = cand
                break
        if label_col is None:
            raise KeyError(f"No label column found in {path}. Expected one of labels,y_true,label,targets,y")
        labels = df[label_col].to_numpy().astype(np.int64)

        # probs/logits column
        vec_col = None
        for cand in ["logits", "probs", "probabilities", "y_pred_logits", "y_pred"]:
            if cand in df.columns:
                vec_col = cand
                break
        if vec_col is None:
            raise KeyError(f"No logits/probs column found in {path}. Expected logits/probs/...")

        # normalize to 2D array
        first = df[vec_col].iloc[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            mat = np.stack([_to_np(v) for v in df[vec_col].tolist()], axis=0)
        else:
            # string/bytes/object
            mat = np.stack([_to_np(v) for v in df[vec_col].tolist()], axis=0)

        mat = np.asarray(mat)
        # If probs provided, convert to logits (stable)
        if ("prob" in vec_col) or (np.all((mat >= 0) & (mat <= 1) & np.isfinite(mat)) and np.isclose(mat.sum(1), 1.0, atol=1e-3).all()):
            eps = 1e-8
            mat = np.log(np.clip(mat, eps, 1.0))  # pseudo-logits
        return labels, mat.astype(np.float32)

    if path.suffix.lower() == ".npz":
        data = np.load(path)
        logits = np.asarray(data["logits"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int64)
        return labels, logits

    raise FileNotFoundError(f"Unsupported prediction path: {path}")

def scan_pred_files(glob_or_paths: Iterable[str]) -> List[Path]:
    """
    Accepts:
      - absolute/relative file paths
      - glob patterns (absolute or relative)
      - multiple patterns in a single string separated by ';' or ','

    Returns a de-duplicated, stably-sorted list of Path objects.
    """
    tokens: List[str] = []
    for g in glob_or_paths:
        if g is None:
            continue
        # split on ; or , to support "pat1;pat2;pat3"
        parts = [t.strip() for t in re.split(r"[;,]+", str(g)) if t and t.strip()]
        tokens.extend(parts)

    matches: List[Path] = []
    for tok in tokens:
        p = Path(tok)
        if any(ch in tok for ch in "*?[]"):  # it's a glob pattern
            for hit in glob.glob(tok):
                matches.append(Path(hit))
        elif p.is_dir():
            # common filenames inside a dir
            for cand in ["test_preds.parquet", "preds.parquet", "preds.npz"]:
                q = p / cand
                if q.exists():
                    matches.append(q)
            # fallback to any parquet
            matches += list(sorted(p.glob("*.parquet")))
        elif p.exists():
            matches.append(p)

    # de-duplicate, stable order
    out: List[Path] = []
    seen = set()
    for m in sorted(matches):  # sorted gives deterministic order
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out
