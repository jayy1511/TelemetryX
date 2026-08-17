"""Build and persist a multi-race TelemetryX training corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from telemetryx.config import Settings, load_settings
from telemetryx.data.corpus import (
    RaceCorpusSummary,
    combine_race_datasets,
    summarize_race_corpus,
    validate_race_corpus,
)
from telemetryx.data.dataset_artifacts import (
    RaceDatasetArtifactError,
    RaceDatasetArtifacts,
    load_race_dataset_artifact,
    load_race_dataset_manifest,
    verify_race_dataset_checksum,
)

CORPUS_SCHEMA_VERSION: Final[str] = "1.0"

DEFAULT_RACE_DATASET_DIRECTORY_NAME: Final[str] = "race_datasets"

DEFAULT_CORPUS_DIRECTORY_NAME: Final[str] = "corpora"

DEFAULT_CORPUS_FILENAME: Final[str] = "telemetryx_race_corpus.parquet"

DEFAULT_CORPUS_MANIFEST_FILENAME: Final[str] = "telemetryx_race_corpus_manifest.json"


class RaceCorpusBuildError(RuntimeError):
    """Raised when the processed multi-race corpus cannot be built."""


@dataclass(frozen=True, slots=True)
class RaceCorpusArtifacts:
    """Paths and summary information for one saved race corpus."""

    corpus_path: Path
    manifest_path: Path
    race_count: int
    season_count: int
    row_count: int
    snapshot_count: int
    driver_count: int
    first_race_id: str
    last_race_id: str


def build_and_save_race_corpus(
    settings: Settings | None = None,
    *,
    input_directory: Path | None = None,
    output_directory: Path | None = None,
    overwrite: bool = False,
) -> RaceCorpusArtifacts:
    """
    Discover processed race datasets, combine them and save one corpus.

    Every race artifact must have a matching manifest and valid SHA-256
    checksum before it is admitted to the corpus.

    Parameters
    ----------
    settings:
        Optional validated TelemetryX settings.
    input_directory:
        Optional directory containing individual race dataset artifacts.
        Defaults to ``data/processed/race_datasets``.
    output_directory:
        Optional corpus destination. Defaults to
        ``data/processed/corpora``.
    overwrite:
        Whether existing corpus artifacts may be replaced.

    Returns
    -------
    RaceCorpusArtifacts
        Paths and summary information for the generated corpus.

    Raises
    ------
    RaceCorpusBuildError
        If discovery, integrity checking, loading, combination or persistence
        fails.
    """
    active_settings = settings if settings is not None else load_settings()

    race_artifacts = discover_race_dataset_artifacts(
        settings=active_settings,
        input_directory=input_directory,
    )

    corpus = build_race_corpus_from_artifacts(race_artifacts)

    return save_race_corpus_artifacts(
        corpus,
        settings=active_settings,
        output_directory=output_directory,
        overwrite=overwrite,
    )


def discover_race_dataset_artifacts(
    settings: Settings | None = None,
    *,
    input_directory: Path | None = None,
) -> tuple[RaceDatasetArtifacts, ...]:
    """
    Discover saved race datasets and reconstruct their artifact metadata.

    Every ``*_dataset.parquet`` file must have a corresponding
    ``*_manifest.json`` file.

    Parameters
    ----------
    settings:
        Optional validated TelemetryX settings.
    input_directory:
        Optional race-artifact directory.

    Returns
    -------
    tuple[RaceDatasetArtifacts, ...]
        Discovered artifacts sorted deterministically by race identifier.

    Raises
    ------
    RaceCorpusBuildError
        If no race datasets exist or a manifest is missing or malformed.
    """
    active_settings = settings if settings is not None else load_settings()

    source_directory = _resolve_race_dataset_directory(
        settings=active_settings,
        input_directory=input_directory,
    )

    if not source_directory.is_dir():
        raise RaceCorpusBuildError(
            f"Race-dataset directory does not exist: {source_directory}"
        )

    dataset_paths = sorted(source_directory.glob("*_dataset.parquet"))

    if not dataset_paths:
        raise RaceCorpusBuildError(
            f"No processed race-dataset Parquet files were found in: {source_directory}"
        )

    artifacts: list[RaceDatasetArtifacts] = []

    seen_race_ids: set[str] = set()

    for dataset_path in dataset_paths:
        manifest_path = _manifest_path_for_dataset(dataset_path)

        if not manifest_path.is_file():
            raise RaceCorpusBuildError(
                f"A race dataset is missing its manifest: {dataset_path.name}"
            )

        try:
            manifest = load_race_dataset_manifest(manifest_path)
        except RaceDatasetArtifactError as exc:
            raise RaceCorpusBuildError(
                f"Could not load the manifest for race dataset: {dataset_path.name}"
            ) from exc

        race_id = _require_manifest_text(
            manifest,
            key="race_id",
            manifest_path=manifest_path,
        )

        dataset_filename = _require_manifest_text(
            manifest,
            key="dataset_file",
            manifest_path=manifest_path,
        )

        if dataset_filename != dataset_path.name:
            raise RaceCorpusBuildError(
                "Race-dataset manifest references a different "
                "dataset filename: "
                f"{manifest_path.name}."
            )

        if race_id in seen_race_ids:
            raise RaceCorpusBuildError(
                f"Multiple discovered artifacts use the same RaceId: {race_id}."
            )

        row_count = _require_manifest_non_negative_integer(
            manifest,
            key="row_count",
            manifest_path=manifest_path,
        )

        snapshot_count = _require_manifest_non_negative_integer(
            manifest,
            key="snapshot_count",
            manifest_path=manifest_path,
        )

        driver_count = _require_manifest_non_negative_integer(
            manifest,
            key="driver_count",
            manifest_path=manifest_path,
        )

        artifacts.append(
            RaceDatasetArtifacts(
                dataset_path=dataset_path.resolve(),
                manifest_path=manifest_path.resolve(),
                race_id=race_id,
                row_count=row_count,
                snapshot_count=snapshot_count,
                driver_count=driver_count,
            )
        )

        seen_race_ids.add(race_id)

    artifacts.sort(key=lambda artifact: artifact.race_id)

    return tuple(artifacts)


def build_race_corpus_from_artifacts(
    artifacts: tuple[RaceDatasetArtifacts, ...],
) -> pd.DataFrame:
    """
    Verify and load individual race artifacts into one validated corpus.

    Parameters
    ----------
    artifacts:
        Race dataset artifacts to combine.

    Returns
    -------
    pd.DataFrame
        Validated, chronologically ordered multi-race corpus.

    Raises
    ------
    RaceCorpusBuildError
        If no artifacts are supplied or an artifact fails integrity,
        validation or manifest consistency checks.
    """
    if not artifacts:
        raise RaceCorpusBuildError("At least one race-dataset artifact is required.")

    race_datasets: list[pd.DataFrame] = []

    seen_race_ids: set[str] = set()

    for artifact in artifacts:
        if not isinstance(
            artifact,
            RaceDatasetArtifacts,
        ):
            raise TypeError(
                "Every corpus artifact must be a RaceDatasetArtifacts instance."
            )

        if artifact.race_id in seen_race_ids:
            raise RaceCorpusBuildError(
                "The same race artifact was supplied more than once: "
                f"{artifact.race_id}."
            )

        try:
            checksum_valid = verify_race_dataset_checksum(artifact)
        except RaceDatasetArtifactError as exc:
            raise RaceCorpusBuildError(
                f"Could not verify race-dataset checksum: {artifact.race_id}."
            ) from exc

        if not checksum_valid:
            raise RaceCorpusBuildError(
                "Race-dataset checksum does not match its manifest: "
                f"{artifact.race_id}."
            )

        try:
            dataset = load_race_dataset_artifact(artifact.dataset_path)
        except RaceDatasetArtifactError as exc:
            raise RaceCorpusBuildError(
                f"Could not load validated race dataset: {artifact.race_id}."
            ) from exc

        race_id = _extract_dataset_race_id(dataset)

        if race_id != artifact.race_id:
            raise RaceCorpusBuildError(
                "RaceId in the dataset does not match its manifest: "
                f"manifest={artifact.race_id}, dataset={race_id}."
            )

        if len(dataset) != artifact.row_count:
            raise RaceCorpusBuildError(
                "Race-dataset row count does not match its manifest: "
                f"{artifact.race_id}."
            )

        actual_snapshot_count = len(
            dataset.loc[
                :,
                [
                    "RaceId",
                    "SnapshotLap",
                ],
            ].drop_duplicates()
        )

        if actual_snapshot_count != artifact.snapshot_count:
            raise RaceCorpusBuildError(
                "Race-dataset snapshot count does not match its manifest: "
                f"{artifact.race_id}."
            )

        actual_driver_count = int(dataset["Driver"].astype("string").nunique())

        if actual_driver_count != artifact.driver_count:
            raise RaceCorpusBuildError(
                "Race-dataset driver count does not match its manifest: "
                f"{artifact.race_id}."
            )

        race_datasets.append(dataset)

        seen_race_ids.add(artifact.race_id)

    try:
        corpus = combine_race_datasets(race_datasets)
    except (TypeError, ValueError) as exc:
        raise RaceCorpusBuildError(
            "Could not combine race datasets into a corpus."
        ) from exc

    return corpus


def save_race_corpus_artifacts(
    corpus: pd.DataFrame,
    settings: Settings | None = None,
    *,
    output_directory: Path | None = None,
    overwrite: bool = False,
) -> RaceCorpusArtifacts:
    """
    Save a validated race corpus as Parquet plus a JSON manifest.

    Parameters
    ----------
    corpus:
        Valid combined race corpus.
    settings:
        Optional validated TelemetryX settings.
    output_directory:
        Optional corpus destination.
    overwrite:
        Whether existing corpus artifacts may be replaced.

    Returns
    -------
    RaceCorpusArtifacts
        Generated corpus paths and summary information.

    Raises
    ------
    TypeError
        If ``corpus`` is not a pandas DataFrame.
    RaceCorpusBuildError
        If validation or persistence fails.
    """
    if not isinstance(
        corpus,
        pd.DataFrame,
    ):
        raise TypeError("corpus must be provided as a pandas DataFrame.")

    try:
        validate_race_corpus(corpus)

        summary = summarize_race_corpus(corpus)
    except (TypeError, ValueError) as exc:
        raise RaceCorpusBuildError("The race corpus failed validation.") from exc

    active_settings = settings if settings is not None else load_settings()

    destination = _resolve_corpus_directory(
        settings=active_settings,
        output_directory=output_directory,
    )

    corpus_path = destination / DEFAULT_CORPUS_FILENAME

    manifest_path = destination / DEFAULT_CORPUS_MANIFEST_FILENAME

    _reject_existing_corpus_artifacts(
        paths=(
            corpus_path,
            manifest_path,
        ),
        overwrite=overwrite,
    )

    temporary_corpus_path = corpus_path.with_name(
        f"{corpus_path.stem}.temporary{corpus_path.suffix}"
    )

    temporary_manifest_path = manifest_path.with_name(
        f"{manifest_path.stem}.temporary{manifest_path.suffix}"
    )

    try:
        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        corpus.to_parquet(
            temporary_corpus_path,
            index=False,
        )

        corpus_checksum = _sha256_file(temporary_corpus_path)

        manifest = _build_corpus_manifest(
            corpus=corpus,
            summary=summary,
            corpus_filename=corpus_path.name,
            corpus_checksum=corpus_checksum,
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

        temporary_corpus_path.replace(corpus_path)

        temporary_manifest_path.replace(manifest_path)
    except Exception as exc:
        _remove_file_quietly(temporary_corpus_path)

        _remove_file_quietly(temporary_manifest_path)

        raise RaceCorpusBuildError("Could not save race-corpus artifacts.") from exc

    return RaceCorpusArtifacts(
        corpus_path=corpus_path,
        manifest_path=manifest_path,
        race_count=summary.race_count,
        season_count=summary.season_count,
        row_count=summary.row_count,
        snapshot_count=summary.snapshot_count,
        driver_count=summary.driver_count,
        first_race_id=summary.first_race_id,
        last_race_id=summary.last_race_id,
    )


def load_race_corpus_artifact(
    corpus_path: Path,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Load a saved race-corpus Parquet artifact.

    Parameters
    ----------
    corpus_path:
        Corpus Parquet path.
    validate:
        Whether to run corpus validation after loading.

    Returns
    -------
    pd.DataFrame
        Loaded corpus.

    Raises
    ------
    TypeError
        If ``corpus_path`` is not a pathlib Path.
    RaceCorpusBuildError
        If the corpus cannot be read or validated.
    """
    if not isinstance(
        corpus_path,
        Path,
    ):
        raise TypeError("corpus_path must be provided as a pathlib Path.")

    resolved_path = corpus_path.resolve()

    if not resolved_path.is_file():
        raise RaceCorpusBuildError(
            f"Race-corpus artifact does not exist: {resolved_path}"
        )

    try:
        corpus = pd.read_parquet(resolved_path)
    except Exception as exc:
        raise RaceCorpusBuildError(
            f"Could not read race-corpus artifact: {resolved_path}"
        ) from exc

    if validate:
        try:
            validate_race_corpus(corpus)
        except (TypeError, ValueError) as exc:
            raise RaceCorpusBuildError(
                f"Loaded race-corpus artifact failed validation: {resolved_path}"
            ) from exc

    return corpus


def verify_race_corpus_checksum(
    artifacts: RaceCorpusArtifacts,
) -> bool:
    """
    Verify the saved corpus against its manifest checksum.

    Parameters
    ----------
    artifacts:
        Saved corpus artifact metadata.

    Returns
    -------
    bool
        Whether the current Parquet bytes match the saved digest.
    """
    manifest = _load_corpus_manifest(artifacts.manifest_path)

    expected_checksum = manifest.get("corpus_sha256")

    if not isinstance(
        expected_checksum,
        str,
    ):
        raise RaceCorpusBuildError(
            "Corpus manifest does not contain a valid corpus_sha256 value."
        )

    if not artifacts.corpus_path.is_file():
        raise RaceCorpusBuildError(
            f"Race-corpus artifact does not exist: {artifacts.corpus_path}"
        )

    actual_checksum = _sha256_file(artifacts.corpus_path)

    return actual_checksum == expected_checksum


def create_argument_parser() -> argparse.ArgumentParser:
    """Create the multi-race corpus build command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Combine processed TelemetryX race datasets "
            "into one validated training corpus."
        )
    )

    parser.add_argument(
        "--input-directory",
        type=Path,
        default=None,
        help=(
            "Directory containing individual race datasets. "
            "Defaults to data/processed/race_datasets."
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help=("Corpus destination. Defaults to data/processed/corpora."),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing corpus artifacts.",
    )

    return parser


def print_corpus_build_summary(
    artifacts: RaceCorpusArtifacts,
) -> None:
    """Print a concise summary of a completed corpus build."""
    print()
    print("TelemetryX race corpus")
    print("=" * 60)
    print(f"Races: {artifacts.race_count}")
    print(f"Seasons: {artifacts.season_count}")
    print(f"Rows: {artifacts.row_count}")
    print(f"Snapshots: {artifacts.snapshot_count}")
    print(f"Drivers: {artifacts.driver_count}")
    print(f"First race: {artifacts.first_race_id}")
    print(f"Last race: {artifacts.last_race_id}")
    print(f"Corpus: {artifacts.corpus_path}")
    print(f"Manifest: {artifacts.manifest_path}")


def main() -> None:
    """Run the processed race-corpus build command."""
    parser = create_argument_parser()
    arguments = parser.parse_args()

    artifacts = build_and_save_race_corpus(
        input_directory=arguments.input_directory,
        output_directory=arguments.output_directory,
        overwrite=arguments.overwrite,
    )

    print_corpus_build_summary(artifacts)


def _build_corpus_manifest(
    *,
    corpus: pd.DataFrame,
    summary: RaceCorpusSummary,
    corpus_filename: str,
    corpus_checksum: str,
) -> dict[str, object]:
    """Return the JSON-compatible manifest for one race corpus."""
    race_metadata = (
        corpus.loc[
            :,
            [
                "RaceId",
                "Season",
                "RoundNumber",
            ],
        ]
        .drop_duplicates()
        .sort_values(
            by=[
                "Season",
                "RoundNumber",
                "RaceId",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    race_ids = [str(value) for value in race_metadata["RaceId"].tolist()]

    seasons = sorted({int(value) for value in race_metadata["Season"].tolist()})

    return {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "corpus_file": corpus_filename,
        "corpus_sha256": corpus_checksum,
        "race_count": summary.race_count,
        "season_count": summary.season_count,
        "row_count": summary.row_count,
        "snapshot_count": summary.snapshot_count,
        "driver_count": summary.driver_count,
        "first_race_id": summary.first_race_id,
        "last_race_id": summary.last_race_id,
        "seasons": seasons,
        "race_ids": race_ids,
        "columns": [str(column) for column in corpus.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in corpus.dtypes.items()},
    }


def _load_corpus_manifest(
    manifest_path: Path,
) -> dict[str, object]:
    """Load a corpus JSON manifest."""
    if not isinstance(
        manifest_path,
        Path,
    ):
        raise TypeError("manifest_path must be provided as a pathlib Path.")

    resolved_path = manifest_path.resolve()

    if not resolved_path.is_file():
        raise RaceCorpusBuildError(
            f"Race-corpus manifest does not exist: {resolved_path}"
        )

    try:
        parsed = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise RaceCorpusBuildError(
            f"Could not read race-corpus manifest: {resolved_path}"
        ) from exc

    if not isinstance(
        parsed,
        dict,
    ):
        raise RaceCorpusBuildError("Race-corpus manifest must contain a JSON object.")

    return {str(key): value for key, value in parsed.items()}


def _resolve_race_dataset_directory(
    *,
    settings: Settings,
    input_directory: Path | None,
) -> Path:
    """Return the directory containing processed race datasets."""
    if input_directory is not None:
        return input_directory.resolve()

    return (
        settings.paths.processed_data_dir / DEFAULT_RACE_DATASET_DIRECTORY_NAME
    ).resolve()


def _resolve_corpus_directory(
    *,
    settings: Settings,
    output_directory: Path | None,
) -> Path:
    """Return the destination for corpus artifacts."""
    if output_directory is not None:
        return output_directory.resolve()

    return (settings.paths.processed_data_dir / DEFAULT_CORPUS_DIRECTORY_NAME).resolve()


def _manifest_path_for_dataset(
    dataset_path: Path,
) -> Path:
    """Return the expected manifest path for a race-dataset file."""
    suffix = "_dataset.parquet"

    if not dataset_path.name.endswith(suffix):
        raise RaceCorpusBuildError(
            "Race-dataset filename does not use the expected suffix: "
            f"{dataset_path.name}"
        )

    race_prefix = dataset_path.name[: -len(suffix)]

    return dataset_path.with_name(f"{race_prefix}_manifest.json")


def _require_manifest_text(
    manifest: dict[str, object],
    *,
    key: str,
    manifest_path: Path,
) -> str:
    """Return one required non-empty text manifest value."""
    value = manifest.get(key)

    if not isinstance(
        value,
        str,
    ):
        raise RaceCorpusBuildError(
            f"Manifest {manifest_path.name} does not contain a valid {key} value."
        )

    normalized = value.strip()

    if not normalized:
        raise RaceCorpusBuildError(
            f"Manifest {manifest_path.name} contains a blank {key} value."
        )

    return normalized


def _require_manifest_non_negative_integer(
    manifest: dict[str, object],
    *,
    key: str,
    manifest_path: Path,
) -> int:
    """Return one required non-negative integer manifest value."""
    value = manifest.get(key)

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RaceCorpusBuildError(
            f"Manifest {manifest_path.name} does not contain "
            f"a valid non-negative {key} value."
        )

    return value


def _extract_dataset_race_id(
    dataset: pd.DataFrame,
) -> str:
    """Return the single RaceId represented by one loaded race dataset."""
    values = dataset["RaceId"].astype("string").dropna().unique().tolist()

    if len(values) != 1:
        raise RaceCorpusBuildError(
            "Loaded race dataset must contain exactly one RaceId."
        )

    race_id = str(values[0]).strip()

    if not race_id:
        raise RaceCorpusBuildError("Loaded race dataset contains a blank RaceId.")

    return race_id


def _reject_existing_corpus_artifacts(
    *,
    paths: tuple[Path, ...],
    overwrite: bool,
) -> None:
    """Prevent accidental replacement of corpus artifacts."""
    if overwrite:
        return

    existing_paths = [path for path in paths if path.exists()]

    if not existing_paths:
        return

    formatted_paths = ", ".join(str(path) for path in existing_paths)

    raise RaceCorpusBuildError(
        "One or more race-corpus artifacts already exist. "
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
        raise RaceCorpusBuildError(
            f"Could not calculate file checksum: {path}"
        ) from exc

    return digest.hexdigest()


def _remove_file_quietly(
    path: Path,
) -> None:
    """Remove an incomplete temporary artifact if it exists."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


if __name__ == "__main__":
    main()
