from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_yaml_config(path: str | Path, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
	"""Load YAML config from disk, returning default on missing/invalid content."""
	default_value = dict(default or {})
	file_path = Path(path)
	if not file_path.exists():
		return default_value

	try:
		import yaml  # type: ignore
	except Exception:
		return default_value

	try:
		data = yaml.safe_load(file_path.read_text(encoding="utf-8"))
	except Exception:
		return default_value

	if not isinstance(data, dict):
		return default_value
	return data
