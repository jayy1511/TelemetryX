from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from telemetryx.config import ConfigurationError, load_settings


def make_valid_config() -> dict[str, Any]:
    """Return a complete valid TelemetryX configuration."""
    return {
        "project": {
            "name": "TelemetryX",
            "version": "0.1.0",
            "random_seed": 42,
            "timezone": "UTC",
        },
        "paths": {
            "raw_data_dir": "data/raw",
            "interim_data_dir": "data/interim",
            "processed_data_dir": "data/processed",
            "fastf1_cache_dir": "data/raw/fastf1_cache",
            "model_dir": "models",
            "artifact_dir": "artifacts",
        },
        "data": {
            "primary_source": "fastf1",
            "seasons": [2023, 2024],
            "default_session_type": "R",
            "load_laps": True,
            "load_telemetry": False,
            "load_weather": True,
            "load_race_control_messages": True,
            "cache_enabled": True,
        },
        "sample_race": {
            "season": 2024,
            "event": "Bahrain",
            "session_type": "R",
        },
        "dataset": {
            "snapshot_reference": "leader_lap_completion",
            "include_main_races_only": True,
            "minimum_snapshot_lap": 1,
            "exclude_inaccurate_laps_from_pace_features": True,
            "preserve_raw_missing_values": True,
            "processed_file_format": "parquet",
        },
        "splitting": {
            "strategy": "chronological_race_level",
            "validation": {
                "season": 2023,
                "last_n_races": 5,
            },
            "test": {
                "seasons": [2024],
            },
        },
        "logging": {
            "level": "INFO",
            "save_logs_to_file": True,
            "log_directory": "logs",
        },
    }


def write_config(
    project_root: Path,
    config: dict[str, Any] | str,
) -> Path:
    """
    Write a settings file inside a temporary project directory.

    A dictionary is serialized as YAML. A string is written directly, which
    allows tests to create intentionally malformed YAML.
    """
    config_directory = project_root / "config"
    config_directory.mkdir(parents=True, exist_ok=True)

    config_path = config_directory / "settings.yaml"

    if isinstance(config, str):
        content = config
    else:
        content = yaml.safe_dump(
            config,
            sort_keys=False,
        )

    config_path.write_text(
        content,
        encoding="utf-8",
    )

    return config_path


def test_load_default_project_settings() -> None:
    """The real project configuration should load successfully."""
    settings = load_settings()

    assert settings.project.name == "TelemetryX"
    assert settings.project.version == "0.1.0"
    assert settings.project.random_seed == 42
    assert settings.project.timezone == "UTC"

    assert settings.data.primary_source == "fastf1"
    assert settings.data.seasons == (2023, 2024)
    assert settings.data.load_laps is True
    assert settings.data.load_telemetry is False
    assert settings.data.cache_enabled is True

    assert settings.sample_race.season == 2024
    assert settings.sample_race.event == "Bahrain"
    assert settings.sample_race.session_type == "R"

    assert settings.dataset.snapshot_reference == "leader_lap_completion"
    assert settings.dataset.processed_file_format == "parquet"


def test_paths_are_resolved_from_project_root() -> None:
    """Configured paths should become absolute repository paths."""
    settings = load_settings()

    assert settings.project_root.is_absolute()
    assert settings.config_file.is_absolute()

    assert (
        settings.paths.raw_data_dir
        == (settings.project_root / "data" / "raw").resolve()
    )

    assert (
        settings.paths.fastf1_cache_dir
        == (settings.project_root / "data" / "raw" / "fastf1_cache").resolve()
    )

    assert (
        settings.paths.processed_data_dir
        == (settings.project_root / "data" / "processed").resolve()
    )

    assert settings.logging.log_directory == (settings.project_root / "logs").resolve()


def test_custom_project_root_can_be_used(tmp_path: Path) -> None:
    """Tests and tools should be able to provide another project root."""
    write_config(
        project_root=tmp_path,
        config=make_valid_config(),
    )

    settings = load_settings(project_root=tmp_path)

    assert settings.project_root == tmp_path.resolve()
    assert settings.config_file == (tmp_path / "config" / "settings.yaml").resolve()

    assert settings.paths.model_dir == (tmp_path / "models").resolve()


def test_missing_configuration_file_raises_error(
    tmp_path: Path,
) -> None:
    """A missing settings file should produce a clear error."""
    with pytest.raises(
        FileNotFoundError,
        match="TelemetryX configuration file was not found",
    ):
        load_settings(project_root=tmp_path)


def test_malformed_yaml_raises_configuration_error(
    tmp_path: Path,
) -> None:
    """Invalid YAML syntax should be reported as a configuration error."""
    write_config(
        project_root=tmp_path,
        config="project: [this YAML is not closed",
    )

    with pytest.raises(
        ConfigurationError,
        match="Invalid YAML",
    ):
        load_settings(project_root=tmp_path)


def test_missing_required_section_is_rejected(
    tmp_path: Path,
) -> None:
    """Every required top-level settings section must exist."""
    config = make_valid_config()
    del config["logging"]

    write_config(
        project_root=tmp_path,
        config=config,
    )

    with pytest.raises(
        ConfigurationError,
        match="Missing required configuration section: logging",
    ):
        load_settings(project_root=tmp_path)


def test_sample_race_must_belong_to_configured_seasons(
    tmp_path: Path,
) -> None:
    """The sample race cannot use an unconfigured season."""
    config = make_valid_config()
    config["sample_race"]["season"] = 2022

    write_config(
        project_root=tmp_path,
        config=config,
    )

    with pytest.raises(
        ConfigurationError,
        match="sample_race.season must be included in data.seasons",
    ):
        load_settings(project_root=tmp_path)


def test_absolute_project_path_is_rejected(
    tmp_path: Path,
) -> None:
    """Configuration paths must not depend on one machine."""
    config = make_valid_config()
    config["paths"]["raw_data_dir"] = str(
        (tmp_path / "absolute-data-directory").resolve()
    )

    write_config(
        project_root=tmp_path,
        config=config,
    )

    with pytest.raises(
        ConfigurationError,
        match="paths.raw_data_dir must be relative",
    ):
        load_settings(project_root=tmp_path)


def test_path_cannot_escape_project_root(
    tmp_path: Path,
) -> None:
    """Relative paths using parent traversal must remain inside the project."""
    config = make_valid_config()
    config["paths"]["raw_data_dir"] = "../outside-project"

    write_config(
        project_root=tmp_path,
        config=config,
    )

    with pytest.raises(
        ConfigurationError,
        match="must not point outside the project root",
    ):
        load_settings(project_root=tmp_path)


def test_validation_season_must_precede_test_season(
    tmp_path: Path,
) -> None:
    """Chronological evaluation requires validation before testing."""
    config = make_valid_config()
    config["splitting"]["validation"]["season"] = 2024

    write_config(
        project_root=tmp_path,
        config=config,
    )

    with pytest.raises(
        ConfigurationError,
        match="validation season must occur before",
    ):
        load_settings(project_root=tmp_path)


def test_duplicate_seasons_are_rejected(
    tmp_path: Path,
) -> None:
    """Season lists should not contain repeated values."""
    config = make_valid_config()
    config["data"]["seasons"] = [2023, 2024, 2024]

    write_config(
        project_root=tmp_path,
        config=config,
    )

    with pytest.raises(
        ConfigurationError,
        match="data.seasons must not contain duplicate values",
    ):
        load_settings(project_root=tmp_path)


def test_boolean_strings_are_rejected(
    tmp_path: Path,
) -> None:
    """A YAML string such as 'false' is not a real boolean."""
    config = make_valid_config()
    config["data"]["load_telemetry"] = "false"

    write_config(
        project_root=tmp_path,
        config=config,
    )

    with pytest.raises(
        ConfigurationError,
        match="data.load_telemetry must be true or false",
    ):
        load_settings(project_root=tmp_path)


def test_unsupported_snapshot_reference_is_rejected(
    tmp_path: Path,
) -> None:
    """The MVP must preserve the agreed snapshot definition."""
    config = make_valid_config()
    config["dataset"]["snapshot_reference"] = "driver_lap_number"

    write_config(
        project_root=tmp_path,
        config=config,
    )

    with pytest.raises(
        ConfigurationError,
        match="snapshot_reference must be",
    ):
        load_settings(project_root=tmp_path)
