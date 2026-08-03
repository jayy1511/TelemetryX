from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from telemetryx.config import Settings, load_settings
from telemetryx.data.dataset import validate_race_dataset
from telemetryx.data.targets import TARGET_COLUMN

DATASET_SCHEMA_VERSION: Final[str] = "1.0"

DEFAULT_DATASET_DIRECTORY_NAME: Final[str] = "race_datasets"


class RaceDatasetArtifactError(RuntimeError):
    """Raised when processed race-dataset artifacts cannot be handled."""


@dataclass(frozen=True, slots=True)
class RaceDatasetArtifacts:
    """Paths and summary information for one saved race dataset."""

    dataset_path: Path
    manifest_path: Path
    race_id: str
    row_count: int
    snapshot_count: int
    driver_count: int


def save_race_dataset_artifacts(
    dataset: pd.DataFrame,
    settings: Settings | None = None,
    *,
    output_directory: Path | None = None,
    overwrite: bool = False,
) -> RaceDatasetArtifacts:
    """
    Validate and save one processed race dataset and its manifest.

    The dataset is written as Parquet so pandas data types are preserved. A
    JSON manifest records its schema, dimensions and SHA-256 checksum.

    Parameters
    ----------
    dataset:
        Valid race-level supervised-learning dataset.
    settings:
        Optional TelemetryX settings. Defaults are loaded when omitted.
    output_directory:
        Optional destination directory. By default, artifacts are written
        beneath ``data/processed/race_datasets``.
    overwrite:
        Whether existing artifacts may be replaced.

    Returns
    -------
    RaceDatasetArtifacts
        Generated artifact paths and dataset summary information.

    Raises
    ------
    TypeError
        If ``dataset`` is not a pandas DataFrame.
    RaceDatasetArtifactError
        If output paths already exist or artifact writing fails.
    """
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("dataset must be provided as a pandas DataFrame.")

    validate_race_dataset(dataset)

    race_id = _extract_single_text_value(
        dataset,
        column="RaceId",
    )

    active_settings = settings if settings is not None else load_settings()

    destination = _resolve_output_directory(
        settings=active_settings,
        output_directory=output_directory,
    )

    dataset_path = destination / f"{race_id}_dataset.parquet"

    manifest_path = destination / f"{race_id}_manifest.json"

    _reject_existing_artifacts(
        paths=(
            dataset_path,
            manifest_path,
        ),
        overwrite=overwrite,
    )

    temporary_dataset_path = dataset_path.with_suffix(".temporary.parquet")

    temporary_manifest_path = manifest_path.with_suffix(".temporary.json")

    snapshot_laps = [int(value) for value in dataset["SnapshotLap"].tolist()]

    snapshot_count = len(set(snapshot_laps))

    driver_count = len({str(value) for value in dataset["Driver"].tolist()})

    try:
        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataset.to_parquet(
            temporary_dataset_path,
            index=False,
        )

        dataset_checksum = _sha256_file(temporary_dataset_path)

        manifest = _build_manifest(
            dataset=dataset,
            dataset_filename=dataset_path.name,
            dataset_checksum=dataset_checksum,
            race_id=race_id,
            snapshot_laps=snapshot_laps,
            driver_count=driver_count,
        )

        temporary_manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=False,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary_dataset_path.replace(dataset_path)

        temporary_manifest_path.replace(manifest_path)
    except Exception as exc:
        _remove_file_quietly(temporary_dataset_path)

        _remove_file_quietly(temporary_manifest_path)

        raise RaceDatasetArtifactError(
            "Could not save processed race-dataset artifacts."
        ) from exc

    return RaceDatasetArtifacts(
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        race_id=race_id,
        row_count=len(dataset),
        snapshot_count=snapshot_count,
        driver_count=driver_count,
    )


def load_race_dataset_artifact(
    dataset_path: Path,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Load a processed race dataset from Parquet.

    Parameters
    ----------
    dataset_path:
        Path to a race-dataset Parquet artifact.
    validate:
        Whether to run the full race-dataset validator after loading.

    Returns
    -------
    pd.DataFrame
        Loaded race dataset.

    Raises
    ------
    TypeError
        If ``dataset_path`` is not a pathlib Path.
    RaceDatasetArtifactError
        If the artifact does not exist, cannot be read or fails validation.
    """
    if not isinstance(dataset_path, Path):
        raise TypeError("dataset_path must be provided as a pathlib Path.")

    resolved_path = dataset_path.resolve()

    if not resolved_path.is_file():
        raise RaceDatasetArtifactError(
            f"Race-dataset artifact does not exist: {resolved_path}"
        )

    try:
        dataset = pd.read_parquet(resolved_path)
    except Exception as exc:
        raise RaceDatasetArtifactError(
            f"Could not read race-dataset artifact: {resolved_path}"
        ) from exc

    if validate:
        try:
            validate_race_dataset(dataset)
        except (TypeError, ValueError) as exc:
            raise RaceDatasetArtifactError(
                f"Loaded race-dataset artifact failed validation: {resolved_path}"
            ) from exc

    return dataset


def load_race_dataset_manifest(
    manifest_path: Path,
) -> dict[str, object]:
    """
    Load and minimally validate a race-dataset JSON manifest.

    Parameters
    ----------
    manifest_path:
        Path to a saved race-dataset manifest.

    Returns
    -------
    dict[str, object]
        Parsed manifest values.

    Raises
    ------
    TypeError
        If ``manifest_path`` is not a pathlib Path.
    RaceDatasetArtifactError
        If the manifest does not exist, is invalid JSON or is not an object.
    """
    if not isinstance(manifest_path, Path):
        raise TypeError("manifest_path must be provided as a pathlib Path.")

    resolved_path = manifest_path.resolve()

    if not resolved_path.is_file():
        raise RaceDatasetArtifactError(
            f"Race-dataset manifest does not exist: {resolved_path}"
        )

    try:
        parsed = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RaceDatasetArtifactError(
            f"Could not read race-dataset manifest: {resolved_path}"
        ) from exc

    if not isinstance(parsed, dict):
        raise RaceDatasetArtifactError(
            "Race-dataset manifest must contain a JSON object."
        )

    return {str(key): value for key, value in parsed.items()}


def verify_race_dataset_checksum(
    artifacts: RaceDatasetArtifacts,
) -> bool:
    """
    Verify that a dataset file matches the checksum stored in its manifest.

    Parameters
    ----------
    artifacts:
        Saved race-dataset artifact paths.

    Returns
    -------
    bool
        ``True`` when the current Parquet checksum matches the manifest.
    """
    manifest = load_race_dataset_manifest(artifacts.manifest_path)

    expected_checksum = manifest.get("dataset_sha256")

    if not isinstance(expected_checksum, str):
        raise RaceDatasetArtifactError(
            "The manifest does not contain a valid dataset_sha256 value."
        )

    if not artifacts.dataset_path.is_file():
        raise RaceDatasetArtifactError(
            f"Race-dataset artifact does not exist: {artifacts.dataset_path}"
        )

    actual_checksum = _sha256_file(artifacts.dataset_path)

    return actual_checksum == expected_checksum


def _build_manifest(
    *,
    dataset: pd.DataFrame,
    dataset_filename: str,
    dataset_checksum: str,
    race_id: str,
    snapshot_laps: list[int],
    driver_count: int,
) -> dict[str, object]:
    """Build a JSON-compatible manifest for one processed race dataset."""
    if not snapshot_laps:
        raise RaceDatasetArtifactError(
            "Cannot create a manifest without replay snapshots."
        )

    winner_row_count = int(dataset[TARGET_COLUMN].sum())

    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "race_id": race_id,
        "dataset_file": dataset_filename,
        "dataset_sha256": dataset_checksum,
        "row_count": len(dataset),
        "snapshot_count": len(set(snapshot_laps)),
        "driver_count": driver_count,
        "snapshot_lap_min": min(snapshot_laps),
        "snapshot_lap_max": max(snapshot_laps),
        "winner_row_count": winner_row_count,
        "target_column": TARGET_COLUMN,
        "columns": [str(column) for column in dataset.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in dataset.dtypes.items()},
    }


def _resolve_output_directory(
    *,
    settings: Settings,
    output_directory: Path | None,
) -> Path:
    """Return the resolved destination for processed race datasets."""
    if output_directory is not None:
        return output_directory.resolve()

    return (
        settings.paths.processed_data_dir / DEFAULT_DATASET_DIRECTORY_NAME
    ).resolve()


def _extract_single_text_value(
    dataset: pd.DataFrame,
    *,
    column: str,
) -> str:
    """Return one non-empty text value shared by every dataset row."""
    values = dataset[column].dropna().astype("string").unique().tolist()

    if len(values) != 1:
        raise RaceDatasetArtifactError(f"{column} must contain exactly one value.")

    normalized = str(values[0]).strip()

    if not normalized:
        raise RaceDatasetArtifactError(f"{column} cannot be blank.")

    return normalized


def _reject_existing_artifacts(
    *,
    paths: tuple[Path, ...],
    overwrite: bool,
) -> None:
    """Prevent accidental replacement of processed dataset artifacts."""
    if overwrite:
        return

    existing_paths = [path for path in paths if path.exists()]

    if not existing_paths:
        return

    formatted_paths = ", ".join(str(path) for path in existing_paths)

    raise RaceDatasetArtifactError(
        "One or more race-dataset artifacts already exist. "
        "Use overwrite=True to replace them: "
        f"{formatted_paths}"
    )


def _sha256_file(
    path: Path,
) -> str:
    """Return the hexadecimal SHA-256 digest of one file."""
    digest = hashlib.sha256()

    try:
        with path.open("rb") as file_handle:
            while chunk := file_handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise RaceDatasetArtifactError(
            f"Could not calculate file checksum: {path}"
        ) from exc

    return digest.hexdigest()


def _remove_file_quietly(
    path: Path,
) -> None:
    """Remove a temporary artifact without hiding the original failure."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
