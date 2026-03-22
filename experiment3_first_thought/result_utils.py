"""Small helpers for writing Experiment 3 result JSON files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def to_serializable(obj):
    """Convert numpy values to plain Python types for JSON output."""
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(key): to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(value) for value in obj]
    return obj


def write_json(data, output_path):
    """Write JSON with parent directory creation and numpy conversion."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as handle:
        json.dump(to_serializable(data), handle, indent=2)
