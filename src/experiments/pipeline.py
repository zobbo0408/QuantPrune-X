"""Small shared helpers for pipeline-owner experiment orchestration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .result_schema import validate_result


def write_result(run_dir: str | Path, result: Mapping[str, Any]) -> Path:
    """Validate and write the canonical ``results.json`` artifact for one run."""
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "results.json"
    output.write_text(json.dumps(validate_result(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
