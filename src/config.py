"""Load and resolve SpatialCell Lakehouse configuration."""

from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "dataset_config.yaml"

REQUIRED_SECTIONS = {
    "project",
    "dataset",
    "paths",
    "anndata",
    "extraction",
    "outputs",
}
DATA_PATH_NAMES = ("bronze", "silver", "gold", "reports")
OUTPUT_PATH_NAMES = ("silver", "gold", "reports")


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load, validate, and resolve a YAML configuration without filesystem writes."""
    path = _resolve_config_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Configuration path is not a regular file: {path}")

    try:
        with path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in configuration file {path}: {exc}") from exc

    if config is None:
        raise ValueError(f"Configuration file is empty: {path}")
    if not isinstance(config, dict):
        raise ValueError(
            f"Top-level YAML value must be a mapping, got {type(config).__name__}"
        )

    _validate_required_sections(config)
    resolved_paths = _resolve_data_paths(config["paths"])
    h5ad_filename = config["dataset"].get("h5ad_filename")
    if not isinstance(h5ad_filename, str) or not h5ad_filename.strip():
        raise ValueError("Configuration value 'dataset.h5ad_filename' must be a string")

    resolved_paths["source_h5ad"] = (
        resolved_paths["bronze"] / h5ad_filename
    ).resolve()
    config["resolved_paths"] = resolved_paths
    return config


def ensure_output_directories(config: Mapping[str, Any]) -> None:
    """Create the configured Silver, Gold, and Reports directories only."""
    resolved_paths = config.get("resolved_paths")
    if not isinstance(resolved_paths, Mapping):
        raise ValueError("Configuration must contain a 'resolved_paths' mapping")

    for name in OUTPUT_PATH_NAMES:
        directory = resolved_paths.get(name)
        if not isinstance(directory, Path):
            raise ValueError(f"Resolved path '{name}' must be a pathlib.Path")
        directory.mkdir(parents=True, exist_ok=True)


def _resolve_config_path(config_path: str | Path | None) -> Path:
    path = DEFAULT_CONFIG_PATH if config_path is None else Path(config_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _validate_required_sections(config: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_SECTIONS.difference(config))
    if missing:
        raise ValueError(
            "Configuration is missing required section(s): " + ", ".join(missing)
        )

    invalid = sorted(
        section
        for section in REQUIRED_SECTIONS
        if not isinstance(config[section], dict)
    )
    if invalid:
        raise ValueError(
            "Required configuration section(s) must be mappings: "
            + ", ".join(invalid)
        )


def _resolve_data_paths(paths: Mapping[str, Any]) -> dict[str, Path]:
    missing = sorted(set(DATA_PATH_NAMES).difference(paths))
    if missing:
        raise ValueError(
            "Configuration section 'paths' is missing required path(s): "
            + ", ".join(missing)
        )

    resolved: dict[str, Path] = {}
    for name in DATA_PATH_NAMES:
        raw_path = paths[name]
        if not isinstance(raw_path, (str, Path)):
            raise ValueError(f"Configuration path 'paths.{name}' must be a string")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        resolved[name] = path.resolve()
    return resolved
