"""Load and validate TelemetryX project configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml


class ConfigurationError(ValueError):
    """Raised when the TelemetryX configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class ProjectSettings:
    """General project-level settings."""

    name: str
    version: str
    random_seed: int
    timezone: str


@dataclass(frozen=True, slots=True)
class PathSettings:
    """Resolved filesystem locations used by TelemetryX."""

    raw_data_dir: Path
    interim_data_dir: Path
    processed_data_dir: Path
    fastf1_cache_dir: Path
    model_dir: Path
    artifact_dir: Path


@dataclass(frozen=True, slots=True)
class DataSettings:
    """Settings controlling race-data loading."""

    primary_source: str
    seasons: tuple[int, ...]
    default_session_type: str
    load_laps: bool
    load_telemetry: bool
    load_weather: bool
    load_race_control_messages: bool
    cache_enabled: bool


@dataclass(frozen=True, slots=True)
class SampleRaceSettings:
    """Race used during initial development and data inspection."""

    season: int
    event: str
    session_type: str


@dataclass(frozen=True, slots=True)
class DatasetSettings:
    """Settings controlling driver-lap snapshot construction."""

    snapshot_reference: str
    include_main_races_only: bool
    minimum_snapshot_lap: int
    exclude_inaccurate_laps_from_pace_features: bool
    preserve_raw_missing_values: bool
    processed_file_format: str


@dataclass(frozen=True, slots=True)
class ValidationSplitSettings:
    """Chronological validation-split settings."""

    season: int
    last_n_races: int


@dataclass(frozen=True, slots=True)
class TestSplitSettings:
    """Chronological test-split settings."""

    seasons: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SplittingSettings:
    """Dataset splitting configuration."""

    strategy: str
    validation: ValidationSplitSettings
    test: TestSplitSettings


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    """Application logging configuration."""

    level: str
    save_logs_to_file: bool
    log_directory: Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Complete validated TelemetryX configuration."""

    project_root: Path
    config_file: Path
    project: ProjectSettings
    paths: PathSettings
    data: DataSettings
    sample_race: SampleRaceSettings
    dataset: DatasetSettings
    splitting: SplittingSettings
    logging: LoggingSettings


def find_project_root() -> Path:
    """
    Return the TelemetryX repository root.

    This file lives at ``src/telemetryx/config.py``. Moving two parent
    directories upward from the package directory reaches the repository root.

    Returns
    -------
    Path
        Absolute path to the TelemetryX repository root.
    """
    return Path(__file__).resolve().parents[2]


def load_settings(
    config_path: Path | None = None,
    project_root: Path | None = None,
) -> Settings:
    """
    Load and validate TelemetryX settings from a YAML file.

    Parameters
    ----------
    config_path:
        Optional path to a YAML configuration file. Relative paths are resolved
        against the project root. When omitted, ``config/settings.yaml`` is
        used.
    project_root:
        Optional repository root override. This is mainly useful in automated
        tests. When omitted, the root is inferred from this module's location.

    Returns
    -------
    Settings
        Fully validated and typed TelemetryX settings.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ConfigurationError
        If the YAML is invalid or contains unsupported values.
    """
    root = project_root.resolve() if project_root is not None else find_project_root()

    settings_file = _resolve_config_file(
        root=root,
        config_path=config_path,
    )

    raw_config = _read_yaml(settings_file)
    settings = _build_settings(
        raw_config=raw_config,
        project_root=root,
        config_file=settings_file,
    )

    _validate_cross_section_rules(settings)

    return settings


def _resolve_config_file(
    root: Path,
    config_path: Path | None,
) -> Path:
    """Resolve the configuration file location."""
    if config_path is None:
        candidate = root / "config" / "settings.yaml"
    elif config_path.is_absolute():
        candidate = config_path
    else:
        candidate = root / config_path

    resolved = candidate.resolve()

    if not resolved.is_file():
        raise FileNotFoundError(
            f"TelemetryX configuration file was not found: {resolved}"
        )

    return resolved


def _read_yaml(config_file: Path) -> Mapping[str, object]:
    """Read a YAML file and ensure its root value is a mapping."""
    try:
        content = config_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(
            f"Could not read configuration file: {config_file}"
        ) from exc

    try:
        parsed: object = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Invalid YAML in configuration file: {config_file}"
        ) from exc

    return _require_mapping(parsed, "configuration root")


def _build_settings(
    raw_config: Mapping[str, object],
    project_root: Path,
    config_file: Path,
) -> Settings:
    """Convert raw YAML values into typed settings objects."""
    project_section = _require_section(raw_config, "project")
    paths_section = _require_section(raw_config, "paths")
    data_section = _require_section(raw_config, "data")
    sample_race_section = _require_section(raw_config, "sample_race")
    dataset_section = _require_section(raw_config, "dataset")
    splitting_section = _require_section(raw_config, "splitting")
    logging_section = _require_section(raw_config, "logging")

    validation_section = _require_section(
        splitting_section,
        "validation",
        parent="splitting",
    )
    test_section = _require_section(
        splitting_section,
        "test",
        parent="splitting",
    )

    project = ProjectSettings(
        name=_require_string(project_section, "name", "project"),
        version=_require_string(project_section, "version", "project"),
        random_seed=_require_int(
            project_section,
            "random_seed",
            "project",
            minimum=0,
        ),
        timezone=_require_string(project_section, "timezone", "project"),
    )

    paths = PathSettings(
        raw_data_dir=_require_project_path(
            paths_section,
            "raw_data_dir",
            "paths",
            project_root,
        ),
        interim_data_dir=_require_project_path(
            paths_section,
            "interim_data_dir",
            "paths",
            project_root,
        ),
        processed_data_dir=_require_project_path(
            paths_section,
            "processed_data_dir",
            "paths",
            project_root,
        ),
        fastf1_cache_dir=_require_project_path(
            paths_section,
            "fastf1_cache_dir",
            "paths",
            project_root,
        ),
        model_dir=_require_project_path(
            paths_section,
            "model_dir",
            "paths",
            project_root,
        ),
        artifact_dir=_require_project_path(
            paths_section,
            "artifact_dir",
            "paths",
            project_root,
        ),
    )

    data = DataSettings(
        primary_source=_require_string(
            data_section,
            "primary_source",
            "data",
        ),
        seasons=_require_int_tuple(
            data_section,
            "seasons",
            "data",
            minimum=1950,
        ),
        default_session_type=_require_string(
            data_section,
            "default_session_type",
            "data",
        ),
        load_laps=_require_bool(
            data_section,
            "load_laps",
            "data",
        ),
        load_telemetry=_require_bool(
            data_section,
            "load_telemetry",
            "data",
        ),
        load_weather=_require_bool(
            data_section,
            "load_weather",
            "data",
        ),
        load_race_control_messages=_require_bool(
            data_section,
            "load_race_control_messages",
            "data",
        ),
        cache_enabled=_require_bool(
            data_section,
            "cache_enabled",
            "data",
        ),
    )

    sample_race = SampleRaceSettings(
        season=_require_int(
            sample_race_section,
            "season",
            "sample_race",
            minimum=1950,
        ),
        event=_require_string(
            sample_race_section,
            "event",
            "sample_race",
        ),
        session_type=_require_string(
            sample_race_section,
            "session_type",
            "sample_race",
        ),
    )

    dataset = DatasetSettings(
        snapshot_reference=_require_string(
            dataset_section,
            "snapshot_reference",
            "dataset",
        ),
        include_main_races_only=_require_bool(
            dataset_section,
            "include_main_races_only",
            "dataset",
        ),
        minimum_snapshot_lap=_require_int(
            dataset_section,
            "minimum_snapshot_lap",
            "dataset",
            minimum=0,
        ),
        exclude_inaccurate_laps_from_pace_features=_require_bool(
            dataset_section,
            "exclude_inaccurate_laps_from_pace_features",
            "dataset",
        ),
        preserve_raw_missing_values=_require_bool(
            dataset_section,
            "preserve_raw_missing_values",
            "dataset",
        ),
        processed_file_format=_require_string(
            dataset_section,
            "processed_file_format",
            "dataset",
        ),
    )

    validation = ValidationSplitSettings(
        season=_require_int(
            validation_section,
            "season",
            "splitting.validation",
            minimum=1950,
        ),
        last_n_races=_require_int(
            validation_section,
            "last_n_races",
            "splitting.validation",
            minimum=1,
        ),
    )

    test = TestSplitSettings(
        seasons=_require_int_tuple(
            test_section,
            "seasons",
            "splitting.test",
            minimum=1950,
        )
    )

    splitting = SplittingSettings(
        strategy=_require_string(
            splitting_section,
            "strategy",
            "splitting",
        ),
        validation=validation,
        test=test,
    )

    logging = LoggingSettings(
        level=_require_string(
            logging_section,
            "level",
            "logging",
        ).upper(),
        save_logs_to_file=_require_bool(
            logging_section,
            "save_logs_to_file",
            "logging",
        ),
        log_directory=_require_project_path(
            logging_section,
            "log_directory",
            "logging",
            project_root,
        ),
    )

    return Settings(
        project_root=project_root,
        config_file=config_file,
        project=project,
        paths=paths,
        data=data,
        sample_race=sample_race,
        dataset=dataset,
        splitting=splitting,
        logging=logging,
    )


def _validate_cross_section_rules(settings: Settings) -> None:
    """Validate rules involving values from multiple configuration sections."""
    allowed_session_types = {
        "R",
        "Q",
        "S",
        "FP1",
        "FP2",
        "FP3",
    }
    allowed_log_levels = {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }

    if settings.sample_race.season not in settings.data.seasons:
        raise ConfigurationError("sample_race.season must be included in data.seasons.")

    if settings.splitting.validation.season not in settings.data.seasons:
        raise ConfigurationError(
            "splitting.validation.season must be included in data.seasons."
        )

    missing_test_seasons = set(settings.splitting.test.seasons).difference(
        settings.data.seasons
    )
    if missing_test_seasons:
        missing = ", ".join(str(season) for season in sorted(missing_test_seasons))
        raise ConfigurationError(
            "Every splitting.test season must be included in data.seasons. "
            f"Missing: {missing}."
        )

    earliest_test_season = min(settings.splitting.test.seasons)
    if settings.splitting.validation.season >= earliest_test_season:
        raise ConfigurationError(
            "The validation season must occur before the earliest test season."
        )

    if settings.data.default_session_type not in allowed_session_types:
        raise ConfigurationError(
            "data.default_session_type is unsupported: "
            f"{settings.data.default_session_type!r}."
        )

    if settings.sample_race.session_type not in allowed_session_types:
        raise ConfigurationError(
            "sample_race.session_type is unsupported: "
            f"{settings.sample_race.session_type!r}."
        )

    if settings.dataset.processed_file_format.lower() != "parquet":
        raise ConfigurationError(
            "The MVP currently supports only Parquet processed datasets."
        )

    if settings.dataset.snapshot_reference != "leader_lap_completion":
        raise ConfigurationError(
            "The MVP snapshot_reference must be 'leader_lap_completion'."
        )

    if settings.logging.level not in allowed_log_levels:
        raise ConfigurationError(
            f"Unsupported logging level: {settings.logging.level!r}."
        )


def _require_section(
    mapping: Mapping[str, object],
    key: str,
    parent: str | None = None,
) -> Mapping[str, object]:
    """Return a required nested configuration section."""
    location = f"{parent}.{key}" if parent else key

    if key not in mapping:
        raise ConfigurationError(f"Missing required configuration section: {location}.")

    return _require_mapping(mapping[key], location)


def _require_mapping(
    value: object,
    location: str,
) -> Mapping[str, object]:
    """Ensure a configuration value is a string-keyed mapping."""
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{location} must be a YAML mapping.")

    for key in value:
        if not isinstance(key, str):
            raise ConfigurationError(f"Every key in {location} must be a string.")

    return cast(Mapping[str, object], value)


def _require_string(
    mapping: Mapping[str, object],
    key: str,
    section: str,
) -> str:
    """Return a required non-empty string value."""
    value = _require_key(mapping, key, section)

    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{section}.{key} must be a non-empty string.")

    return value.strip()


def _require_bool(
    mapping: Mapping[str, object],
    key: str,
    section: str,
) -> bool:
    """Return a required boolean value."""
    value = _require_key(mapping, key, section)

    if not isinstance(value, bool):
        raise ConfigurationError(f"{section}.{key} must be true or false.")

    return value


def _require_int(
    mapping: Mapping[str, object],
    key: str,
    section: str,
    minimum: int | None = None,
) -> int:
    """Return a required integer value with an optional minimum."""
    value = _require_key(mapping, key, section)

    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{section}.{key} must be an integer.")

    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{section}.{key} must be at least {minimum}.")

    return value


def _require_int_tuple(
    mapping: Mapping[str, object],
    key: str,
    section: str,
    minimum: int | None = None,
) -> tuple[int, ...]:
    """Return a non-empty YAML list converted to a tuple of integers."""
    value = _require_key(mapping, key, section)

    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{section}.{key} must be a non-empty YAML list.")

    integers: list[int] = []

    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ConfigurationError(f"{section}.{key}[{index}] must be an integer.")

        if minimum is not None and item < minimum:
            raise ConfigurationError(
                f"{section}.{key}[{index}] must be at least {minimum}."
            )

        integers.append(item)

    if len(integers) != len(set(integers)):
        raise ConfigurationError(f"{section}.{key} must not contain duplicate values.")

    return tuple(integers)


def _require_project_path(
    mapping: Mapping[str, object],
    key: str,
    section: str,
    project_root: Path,
) -> Path:
    """
    Resolve a configured path and ensure it remains inside the repository.

    Absolute paths are rejected because they would make the configuration
    specific to one developer's machine.
    """
    raw_path = _require_string(mapping, key, section)
    relative_path = Path(raw_path)

    if relative_path.is_absolute():
        raise ConfigurationError(
            f"{section}.{key} must be relative to the project root."
        )

    resolved_path = (project_root / relative_path).resolve()

    try:
        resolved_path.relative_to(project_root)
    except ValueError as exc:
        raise ConfigurationError(
            f"{section}.{key} must not point outside the project root."
        ) from exc

    return resolved_path


def _require_key(
    mapping: Mapping[str, object],
    key: str,
    section: str,
) -> object:
    """Return a required configuration value."""
    if key not in mapping:
        raise ConfigurationError(
            f"Missing required configuration value: {section}.{key}."
        )

    return mapping[key]
