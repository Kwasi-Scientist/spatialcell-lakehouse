"""Tests for repository configuration loading and path setup."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.config import (
    DEFAULT_CONFIG_PATH,
    REPO_ROOT,
    ensure_output_directories,
    load_config,
)


@pytest.fixture
def valid_config() -> dict:
    return {
        "project": {"name": "test-project"},
        "dataset": {"h5ad_filename": "source.h5ad"},
        "paths": {
            "bronze": "tmp-test/bronze",
            "silver": "tmp-test/silver",
            "gold": "tmp-test/gold",
            "reports": "tmp-test/reports",
        },
        "anndata": {"expression_source": "X"},
        "extraction": {"mode": "marker_panel"},
        "outputs": {"format": "parquet"},
    }


def write_config(path: Path, config: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_load_config_success_preserves_raw_paths_and_resolves_h5ad(
    tmp_path: Path, valid_config: dict
) -> None:
    config_path = write_config(tmp_path / "valid.yaml", valid_config)

    loaded = load_config(config_path)

    assert loaded["paths"] == valid_config["paths"]
    assert loaded["resolved_paths"]["bronze"] == (
        REPO_ROOT / valid_config["paths"]["bronze"]
    ).resolve()
    assert loaded["resolved_paths"]["source_h5ad"] == (
        loaded["resolved_paths"]["bronze"] / "source.h5ad"
    ).resolve()
    assert all(
        isinstance(path, Path) for path in loaded["resolved_paths"].values()
    )


def test_repository_root_and_default_config_path_contract() -> None:
    expected_root = Path(__file__).resolve().parents[1]

    assert REPO_ROOT == expected_root
    assert DEFAULT_CONFIG_PATH == expected_root / "configs" / "dataset_config.yaml"
    assert load_config()["project"]["name"] == "spatialcell-lakehouse"


def test_missing_configuration_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_config(missing)


def test_empty_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_config(config_path)


def test_non_mapping_top_level_yaml(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "list.yaml", ["not", "a", "mapping"])

    with pytest.raises(ValueError, match="Top-level YAML value must be a mapping"):
        load_config(config_path)


def test_missing_required_sections(tmp_path: Path, valid_config: dict) -> None:
    incomplete = deepcopy(valid_config)
    del incomplete["outputs"]
    config_path = write_config(tmp_path / "incomplete.yaml", incomplete)

    with pytest.raises(ValueError, match=r"missing required section\(s\): outputs"):
        load_config(config_path)


def test_relative_custom_config_path_is_resolved_from_repo_root(
    tmp_path: Path, valid_config: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = write_config(tmp_path / "relative.yaml", valid_config)
    relative_path = config_path.relative_to(REPO_ROOT, walk_up=True)
    monkeypatch.chdir(tmp_path)

    loaded = load_config(relative_path)

    assert loaded["project"]["name"] == "test-project"


def test_relative_and_absolute_data_paths(
    tmp_path: Path, valid_config: dict
) -> None:
    absolute_silver = tmp_path / "absolute-silver"
    configured = deepcopy(valid_config)
    configured["paths"]["silver"] = str(absolute_silver)
    config_path = write_config(tmp_path / "paths.yaml", configured)

    loaded = load_config(config_path)

    assert loaded["resolved_paths"]["bronze"] == (
        REPO_ROOT / configured["paths"]["bronze"]
    ).resolve()
    assert loaded["resolved_paths"]["silver"] == absolute_silver.resolve()
    assert loaded["paths"]["silver"] == str(absolute_silver)


def test_load_config_creates_no_directories(
    tmp_path: Path, valid_config: dict
) -> None:
    configured = deepcopy(valid_config)
    configured["paths"] = {
        name: str(tmp_path / name) for name in ("bronze", "silver", "gold", "reports")
    }
    config_path = write_config(tmp_path / "config.yaml", configured)

    load_config(config_path)

    assert all(not (tmp_path / name).exists() for name in configured["paths"])


def test_ensure_output_directories_excludes_bronze_and_h5ad(
    tmp_path: Path, valid_config: dict
) -> None:
    configured = deepcopy(valid_config)
    configured["paths"] = {
        name: str(tmp_path / name) for name in ("bronze", "silver", "gold", "reports")
    }
    config_path = write_config(tmp_path / "config.yaml", configured)
    loaded = load_config(config_path)

    ensure_output_directories(loaded)

    assert (tmp_path / "silver").is_dir()
    assert (tmp_path / "gold").is_dir()
    assert (tmp_path / "reports").is_dir()
    assert not (tmp_path / "bronze").exists()
    assert not loaded["resolved_paths"]["source_h5ad"].exists()


def test_ensure_output_directories_does_not_modify_existing_bronze_h5ad(
    tmp_path: Path, valid_config: dict
) -> None:
    configured = deepcopy(valid_config)
    configured["paths"] = {
        name: str(tmp_path / name) for name in ("bronze", "silver", "gold", "reports")
    }
    bronze = tmp_path / "bronze"
    bronze.mkdir()
    h5ad = bronze / configured["dataset"]["h5ad_filename"]
    original = b"immutable bronze content"
    h5ad.write_bytes(original)
    config_path = write_config(tmp_path / "config.yaml", configured)
    loaded = load_config(config_path)
    original_mtime = h5ad.stat().st_mtime_ns

    ensure_output_directories(loaded)

    assert h5ad.read_bytes() == original
    assert h5ad.stat().st_mtime_ns == original_mtime
