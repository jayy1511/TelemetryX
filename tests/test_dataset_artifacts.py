from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from telemetryx.config import Settings, load_settings
from telemetryx.data.dataset import build_race_dataset
from telemetryx.data.dataset_artifacts import (
    DATASET_SCHEMA_VERSION,
    RaceDatasetArtifactError,
    RaceDatasetArtifacts,
    load_race_dataset_artifact,
    load_race_dataset_manifest,
    save_race_dataset_artifacts,
    verify_race_dataset_checksum,
)


def make_cleaned_laps() -> pd.DataFrame:
    """Return cleaned lap data for a small three-driver race."""
    return pd.DataFrame(
        {
            "Driver": [
                "VER",
                "VER",
                "VER",
                "NOR",
                "NOR",
                "NOR",
                "BOT",
                "BOT",
            ],
            "LapNumber": [
                1,
                2,
                3,
                1,
                2,
                3,
                1,
                2,
            ],
            "Position": [
                1,
                1,
                1,
                2,
                2,
                2,
                3,
                3,
            ],
            "Stint": [
                1,
                1,
                2,
                1,
                1,
                1,
                1,
                1,
            ],
            "Compound": [
                "SOFT",
                "SOFT",
                "MEDIUM",
                "SOFT",
                "SOFT",
                "SOFT",
                "SOFT",
                "SOFT",
            ],
            "TyreLife": [
                1.0,
                2.0,
                1.0,
                1.0,
                2.0,
                3.0,
                1.0,
                2.0,
            ],
            "TrackStatus": [
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
            ],
            "LapTimeSeconds": [
                90.0,
                89.0,
                88.0,
                91.0,
                90.0,
                89.0,
                92.0,
                91.0,
            ],
        }
    )


def make_results() -> pd.DataFrame:
    """Return final race results containing exactly one winner."""
    return pd.DataFrame(
        {
            "Abbreviation": [
                "VER",
                "NOR",
                "BOT",
            ],
            "Position": [
                1,
                2,
                3,
            ],
            "Status": [
                "Finished",
                "Finished",
                "+1 Lap",
            ],
        }
    )


def make_race_dataset() -> pd.DataFrame:
    """Return one valid assembled race dataset."""
    return build_race_dataset(
        make_cleaned_laps(),
        make_results(),
        season=2024,
        round_number=1,
        event_name="Bahrain Grand Prix",
        session_name="Race",
    )


def settings_with_temporary_processed_directory(
    tmp_path: Path,
) -> Settings:
    """Return settings using a temporary processed-data directory."""
    settings = load_settings()

    temporary_paths = replace(
        settings.paths,
        processed_data_dir=tmp_path / "processed",
    )

    return replace(
        settings,
        paths=temporary_paths,
    )


def test_save_artifacts_creates_parquet_and_manifest(
    tmp_path: Path,
) -> None:
    """Saving should create both processed dataset artifacts."""
    dataset = make_race_dataset()

    artifacts = save_race_dataset_artifacts(
        dataset,
        output_directory=tmp_path,
    )

    assert (
        artifacts.dataset_path
        == (tmp_path / "2024_01_bahrain_grand_prix_dataset.parquet").resolve()
    )

    assert (
        artifacts.manifest_path
        == (tmp_path / "2024_01_bahrain_grand_prix_manifest.json").resolve()
    )

    assert artifacts.dataset_path.is_file()
    assert artifacts.manifest_path.is_file()

    assert artifacts.race_id == ("2024_01_bahrain_grand_prix")

    assert artifacts.row_count == 9
    assert artifacts.snapshot_count == 3
    assert artifacts.driver_count == 3


def test_default_output_directory_uses_project_settings(
    tmp_path: Path,
) -> None:
    """Default artifacts should be written under processed race datasets."""
    settings = settings_with_temporary_processed_directory(tmp_path)

    artifacts = save_race_dataset_artifacts(
        make_race_dataset(),
        settings=settings,
    )

    expected_directory = (tmp_path / "processed" / "race_datasets").resolve()

    assert artifacts.dataset_path.parent == expected_directory
    assert artifacts.manifest_path.parent == expected_directory


def test_parquet_round_trip_preserves_dataset(
    tmp_path: Path,
) -> None:
    """A saved and loaded Parquet dataset should preserve its values."""
    dataset = make_race_dataset()

    artifacts = save_race_dataset_artifacts(
        dataset,
        output_directory=tmp_path,
    )

    loaded = load_race_dataset_artifact(artifacts.dataset_path)

    pd.testing.assert_frame_equal(
        loaded,
        dataset,
    )


def test_load_with_validation_disabled_returns_table(
    tmp_path: Path,
) -> None:
    """Validation may be disabled for controlled diagnostic loading."""
    malformed = pd.DataFrame(
        {
            "Example": [
                1,
                2,
            ]
        }
    )

    dataset_path = tmp_path / "malformed.parquet"

    malformed.to_parquet(
        dataset_path,
        index=False,
    )

    loaded = load_race_dataset_artifact(
        dataset_path,
        validate=False,
    )

    pd.testing.assert_frame_equal(
        loaded,
        malformed,
    )


def test_manifest_contains_dataset_metadata(
    tmp_path: Path,
) -> None:
    """The manifest should document schema and dataset dimensions."""
    dataset = make_race_dataset()

    artifacts = save_race_dataset_artifacts(
        dataset,
        output_directory=tmp_path,
    )

    manifest = load_race_dataset_manifest(artifacts.manifest_path)

    assert manifest["schema_version"] == DATASET_SCHEMA_VERSION

    assert manifest["race_id"] == ("2024_01_bahrain_grand_prix")

    assert manifest["dataset_file"] == ("2024_01_bahrain_grand_prix_dataset.parquet")

    assert manifest["row_count"] == 9
    assert manifest["snapshot_count"] == 3
    assert manifest["driver_count"] == 3
    assert manifest["snapshot_lap_min"] == 1
    assert manifest["snapshot_lap_max"] == 3

    assert manifest["winner_row_count"] == 3
    assert manifest["target_column"] == "WonRace"

    assert manifest["columns"] == [str(column) for column in dataset.columns]

    assert manifest["dtypes"] == {
        str(column): str(dtype) for column, dtype in dataset.dtypes.items()
    }

    checksum = manifest["dataset_sha256"]

    assert isinstance(checksum, str)
    assert len(checksum) == 64


def test_checksum_matches_unchanged_dataset(
    tmp_path: Path,
) -> None:
    """An unchanged Parquet artifact should match its manifest checksum."""
    artifacts = save_race_dataset_artifacts(
        make_race_dataset(),
        output_directory=tmp_path,
    )

    assert verify_race_dataset_checksum(artifacts) is True


def test_checksum_detects_modified_dataset_file(
    tmp_path: Path,
) -> None:
    """Changing the Parquet bytes should invalidate its checksum."""
    artifacts = save_race_dataset_artifacts(
        make_race_dataset(),
        output_directory=tmp_path,
    )

    with artifacts.dataset_path.open("ab") as file_handle:
        file_handle.write(b"telemetryx-test-modification")

    assert verify_race_dataset_checksum(artifacts) is False


def test_existing_artifacts_are_protected_by_default(
    tmp_path: Path,
) -> None:
    """A second save should not replace existing files accidentally."""
    dataset = make_race_dataset()

    save_race_dataset_artifacts(
        dataset,
        output_directory=tmp_path,
    )

    with pytest.raises(
        RaceDatasetArtifactError,
        match="already exist",
    ):
        save_race_dataset_artifacts(
            dataset,
            output_directory=tmp_path,
        )


def test_overwrite_replaces_existing_artifacts(
    tmp_path: Path,
) -> None:
    """Explicit overwrite mode should allow repeated artifact creation."""
    dataset = make_race_dataset()

    first = save_race_dataset_artifacts(
        dataset,
        output_directory=tmp_path,
    )

    second = save_race_dataset_artifacts(
        dataset,
        output_directory=tmp_path,
        overwrite=True,
    )

    assert second.dataset_path == first.dataset_path
    assert second.manifest_path == first.manifest_path

    assert second.dataset_path.is_file()
    assert second.manifest_path.is_file()

    assert verify_race_dataset_checksum(second) is True


def test_save_rejects_invalid_dataset(
    tmp_path: Path,
) -> None:
    """Invalid race datasets should fail before files are written."""
    dataset = make_race_dataset().drop(columns=["SnapshotLap"])

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        save_race_dataset_artifacts(
            dataset,
            output_directory=tmp_path,
        )

    assert list(tmp_path.iterdir()) == []


def test_save_rejects_non_dataframe_input(
    tmp_path: Path,
) -> None:
    """The persistence API requires a pandas DataFrame."""
    invalid_dataset: Any = []

    with pytest.raises(
        TypeError,
        match="dataset must be provided as a pandas DataFrame",
    ):
        save_race_dataset_artifacts(
            invalid_dataset,
            output_directory=tmp_path,
        )


def test_load_rejects_non_path_input() -> None:
    """Dataset loading requires a pathlib Path."""
    invalid_path: Any = "dataset.parquet"

    with pytest.raises(
        TypeError,
        match="dataset_path must be provided as a pathlib Path",
    ):
        load_race_dataset_artifact(invalid_path)


def test_load_rejects_missing_dataset(
    tmp_path: Path,
) -> None:
    """Loading should fail clearly when the Parquet file is absent."""
    missing_path = tmp_path / "missing_dataset.parquet"

    with pytest.raises(
        RaceDatasetArtifactError,
        match="does not exist",
    ):
        load_race_dataset_artifact(missing_path)


def test_load_rejects_invalid_parquet_file(
    tmp_path: Path,
) -> None:
    """Unreadable Parquet content should be wrapped in an artifact error."""
    invalid_path = tmp_path / "invalid_dataset.parquet"

    invalid_path.write_text(
        "this is not parquet",
        encoding="utf-8",
    )

    with pytest.raises(
        RaceDatasetArtifactError,
        match="Could not read race-dataset artifact",
    ):
        load_race_dataset_artifact(invalid_path)


def test_load_wraps_dataset_validation_failure(
    tmp_path: Path,
) -> None:
    """A readable but malformed dataset should fail post-load validation."""
    malformed = pd.DataFrame(
        {
            "RaceId": [
                "example_race",
            ]
        }
    )

    dataset_path = tmp_path / "malformed_dataset.parquet"

    malformed.to_parquet(
        dataset_path,
        index=False,
    )

    with pytest.raises(
        RaceDatasetArtifactError,
        match="failed validation",
    ):
        load_race_dataset_artifact(dataset_path)


def test_manifest_loader_rejects_non_path_input() -> None:
    """Manifest loading requires a pathlib Path."""
    invalid_path: Any = "manifest.json"

    with pytest.raises(
        TypeError,
        match="manifest_path must be provided as a pathlib Path",
    ):
        load_race_dataset_manifest(invalid_path)


def test_manifest_loader_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """A missing manifest should produce a clear artifact error."""
    missing_path = tmp_path / "missing_manifest.json"

    with pytest.raises(
        RaceDatasetArtifactError,
        match="manifest does not exist",
    ):
        load_race_dataset_manifest(missing_path)


def test_manifest_loader_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    """Malformed JSON should not be accepted as a manifest."""
    manifest_path = tmp_path / "invalid_manifest.json"

    manifest_path.write_text(
        "{not-valid-json",
        encoding="utf-8",
    )

    with pytest.raises(
        RaceDatasetArtifactError,
        match="Could not read race-dataset manifest",
    ):
        load_race_dataset_manifest(manifest_path)


@pytest.mark.parametrize(
    "json_value",
    [
        [],
        [
            "not",
            "an",
            "object",
        ],
        "manifest",
        42,
        True,
    ],
)
def test_manifest_loader_requires_json_object(
    tmp_path: Path,
    json_value: object,
) -> None:
    """A manifest root value must be a JSON object."""
    manifest_path = tmp_path / "non_object_manifest.json"

    manifest_path.write_text(
        json.dumps(json_value),
        encoding="utf-8",
    )

    with pytest.raises(
        RaceDatasetArtifactError,
        match="must contain a JSON object",
    ):
        load_race_dataset_manifest(manifest_path)


def test_checksum_verification_requires_manifest_checksum(
    tmp_path: Path,
) -> None:
    """Checksum verification should reject a manifest without a digest."""
    dataset_path = tmp_path / "dataset.parquet"
    manifest_path = tmp_path / "manifest.json"

    make_race_dataset().to_parquet(
        dataset_path,
        index=False,
    )

    manifest_path.write_text(
        json.dumps(
            {
                "race_id": "example",
            }
        ),
        encoding="utf-8",
    )

    artifacts = RaceDatasetArtifacts(
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        race_id="example",
        row_count=9,
        snapshot_count=3,
        driver_count=3,
    )

    with pytest.raises(
        RaceDatasetArtifactError,
        match="valid dataset_sha256",
    ):
        verify_race_dataset_checksum(artifacts)


def test_checksum_verification_requires_dataset_file(
    tmp_path: Path,
) -> None:
    """Checksum verification should fail when the dataset is missing."""
    dataset_path = tmp_path / "missing.parquet"
    manifest_path = tmp_path / "manifest.json"

    manifest_path.write_text(
        json.dumps(
            {
                "dataset_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    artifacts = RaceDatasetArtifacts(
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        race_id="example",
        row_count=0,
        snapshot_count=0,
        driver_count=0,
    )

    with pytest.raises(
        RaceDatasetArtifactError,
        match="artifact does not exist",
    ):
        verify_race_dataset_checksum(artifacts)


def test_failed_parquet_write_leaves_no_temporary_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed Parquet write should not leave incomplete files."""

    def fail_to_parquet(
        self: pd.DataFrame,
        path: Path,
        *,
        index: bool,
    ) -> None:
        raise OSError(f"Simulated Parquet failure: {path}; index={index}")

    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        fail_to_parquet,
    )

    with pytest.raises(
        RaceDatasetArtifactError,
        match="Could not save processed race-dataset artifacts",
    ):
        save_race_dataset_artifacts(
            make_race_dataset(),
            output_directory=tmp_path,
        )

    assert list(tmp_path.iterdir()) == []
