from __future__ import annotations

import re
from pathlib import Path


def resolve_model_path(model_path: Path) -> Path:
    """Return the final model if it exists, otherwise the latest checkpoint."""
    if Path(str(model_path) + ".zip").exists():
        return model_path

    def _step_count(p: Path) -> int:
        m = re.search(r"_(\d+)_steps", p.stem)
        return int(m.group(1)) if m else 0

    checkpoints = sorted(
        (p for p in model_path.parent.glob(f"{model_path.name}_*_steps.zip")
         if re.search(r"_(\d+)_steps", p.stem)),
        key=_step_count,
    )
    if checkpoints:
        latest = checkpoints[-1]
        print(f"Final model not found — loading latest checkpoint: {latest.name}")
        return latest.with_suffix("")
    raise FileNotFoundError(
        f"No model found at {model_path}.zip and no checkpoints in {model_path.parent}. "
        "Run train.py first."
    )
