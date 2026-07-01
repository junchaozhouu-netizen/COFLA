import json
import os
from pathlib import Path
from typing import Any, Dict


def str2bool(x):
    if isinstance(x, bool):
        return x
    x = str(x).strip().lower()
    if x in {"1", "true", "t", "yes", "y"}:
        return True
    if x in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value from: {x}")


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def save_text(text: str, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def experiment_dirs(result_root: str, exp_name: str) -> Dict[str, str]:
    exp_dir = os.path.join(result_root, exp_name)
    dirs = {
        "exp_dir": exp_dir,
        "checkpoints": os.path.join(exp_dir, "checkpoints"),
        "logs": os.path.join(exp_dir, "logs"),
        "metrics": os.path.join(exp_dir, "metrics"),
        "predictions": os.path.join(exp_dir, "predictions"),
    }
    for v in dirs.values():
        ensure_dir(v)
    return dirs
