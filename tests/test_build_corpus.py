from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from telemetryx.config import Settings, load_settings
from telemetryx.data.build_corpus import (
    CORPUS_SCHEMA_VERSION,
    RaceCorpusArtifacts,
    RaceCorpusBuildError,
    build_and_save_race_corpus,
    build_race_corpus_from_artifacts,
    create_argument_parser,
    discover_race_dataset_artifacts,
    load_race_corpus_artifact,
    print_corpus_build_summary,
    save_race_corpus_artifacts,
    verify_race_corpus_checksum,
)
from telemetryx.data.dataset import build_race_dataset
from telemetryx.data.dataset_artifacts import (
    RaceDatasetArtifacts,
    save_race_dataset_artifacts,
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


def make_results(
    *,
    winner: str = "VER",
) -> pd.DataFrame:
    """Return final results containing exactly one winner."""
    drivers = [
        "VER",
        "NOR",
        "BOT",
    ]

    ordered_drivers = [
        winner,
        *[driver for driver in drivers if driver != winner],
    ]

    return pd.DataFrame(
        {
            "Abbreviation": ordered_drivers,
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


def make_race_dataset(
    *,
    season: int,
    round_number: int,
    event_name: str,
    winner: str = "VER",
) -> pd.DataFrame:
    """Return one valid race-level supervised dataset."""
    return build_race_dataset(
        make_cleaned_laps(),
        make_results(
            winner=winner,
        ),
        season=season,
        round_number=round_number,
        event_name=event_name,
        session_name="Race",
    )


def save_sample_races(
    directory: Path,
) -> tuple[RaceDatasetArtifacts, ...]:
    """Save three valid race artifacts across two seasons."""
    race_2023_01 = save_race_dataset_artifacts(
        make_race_dataset(
            season=2023,
            round_number=1,
            event_name="Bahrain Grand Prix",
            winner="VER",
        ),
        output_directory=directory,
    )

    race_2023_02 = save_race_dataset_artifacts(
        make_race_dataset(
            season=2023,
            round_number=2,
            event_name="Saudi Arabian Grand Prix",
            winner="NOR",
        ),
        output_directory=directory,
    )

    race_2024_01 = save_race_dataset_artifacts(
        make_race_dataset(
            season=2024,
            round_number=1,
            event_name="Bahrain Grand Prix",
            winner="VER",
        ),
        output_directory=directory,
    )

    return (
        race_2023_01,
        race_2023_02,
        race_2024_01,
    )


def settings_with_temporary_processed_directory(
    tmp_path: Path,
) -> Settings:
    """Return project settings with temporary processed storage."""
    settings = load_settings()

    paths = replace(
        settings.paths,
        processed_data_dir=tmp_path / "processed",
    )

    return replace(
        settings,
        paths=paths,
    )


def test_discover_race_dataset_artifacts_finds_saved_races(
    tmp_path: Path,
) -> None:
    """Discovery should reconstruct metadata for every saved race."""
    save_sample_races(tmp_path)

    artifacts = discover_race_dataset_artifacts(input_directory=tmp_path)

    assert [artifact.race_id for artifact in artifacts] == [
        "2023_01_bahrain_grand_prix",
        "2023_02_saudi_arabian_grand_prix",
        "2024_01_bahrain_grand_prix",
    ]

    assert all(artifact.dataset_path.is_file() for artifact in artifacts)

    assert all(artifact.manifest_path.is_file() for artifact in artifacts)


def test_discovery_uses_default_processed_directory(
    tmp_path: Path,
) -> None:
    """Default discovery should use processed/race_datasets."""
    settings = settings_with_temporary_processed_directory(tmp_path)

    race_directory = settings.paths.processed_data_dir / "race_datasets"

    save_sample_races(race_directory)

    artifacts = discover_race_dataset_artifacts(settings=settings)

    assert len(artifacts) == 3

    assert all(
        artifact.dataset_path.parent == race_directory.resolve()
        for artifact in artifacts
    )


def test_discovery_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    """A missing race-artifact directory should fail clearly."""
    missing_directory = tmp_path / "missing"

    with pytest.raises(
        RaceCorpusBuildError,
        match="does not exist",
    ):
        discover_race_dataset_artifacts(input_directory=missing_directory)


def test_discovery_rejects_directory_without_datasets(
    tmp_path: Path,
) -> None:
    """At least one race Parquet artifact must exist."""
    with pytest.raises(
        RaceCorpusBuildError,
        match="No processed race-dataset",
    ):
        discover_race_dataset_artifacts(input_directory=tmp_path)


def test_discovery_rejects_missing_manifest(
    tmp_path: Path,
) -> None:
    """Every race Parquet file must have a matching manifest."""
    dataset_path = tmp_path / "2023_01_example_dataset.parquet"

    pd.DataFrame(
        {
            "Example": [
                1,
            ]
        }
    ).to_parquet(
        dataset_path,
        index=False,
    )

    with pytest.raises(
        RaceCorpusBuildError,
        match="missing its manifest",
    ):
        discover_race_dataset_artifacts(input_directory=tmp_path)


def test_discovery_rejects_invalid_manifest_json(
    tmp_path: Path,
) -> None:
    """Malformed race manifests should fail discovery."""
    dataset_path = tmp_path / "2023_01_example_dataset.parquet"

    dataset_path.write_bytes(b"placeholder")

    manifest_path = tmp_path / "2023_01_example_manifest.json"

    manifest_path.write_text(
        "{not-valid-json",
        encoding="utf-8",
    )

    with pytest.raises(
        RaceCorpusBuildError,
        match="Could not load the manifest",
    ):
        discover_race_dataset_artifacts(input_directory=tmp_path)


def test_discovery_rejects_manifest_dataset_filename_mismatch(
    tmp_path: Path,
) -> None:
    """A manifest must reference the Parquet file beside it."""
    artifacts = save_sample_races(tmp_path)

    first = artifacts[0]

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))

    manifest["dataset_file"] = "different_dataset.parquet"

    first.manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(
        RaceCorpusBuildError,
        match="references a different dataset filename",
    ):
        discover_race_dataset_artifacts(input_directory=tmp_path)


def test_discovery_rejects_invalid_manifest_counts(
    tmp_path: Path,
) -> None:
    """Manifest dimensional values must be non-negative integers."""
    artifacts = save_sample_races(tmp_path)

    first = artifacts[0]

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))

    manifest["row_count"] = "nine"

    first.manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(
        RaceCorpusBuildError,
        match="valid non-negative row_count",
    ):
        discover_race_dataset_artifacts(input_directory=tmp_path)


def test_build_corpus_from_artifacts_combines_races(
    tmp_path: Path,
) -> None:
    """Verified race artifacts should combine chronologically."""
    artifacts = save_sample_races(tmp_path)

    corpus = build_race_corpus_from_artifacts(artifacts)

    assert len(corpus) == 27
    assert corpus["RaceId"].nunique() == 3
    assert corpus["Season"].nunique() == 2

    race_order = (
        corpus.loc[
            :,
            [
                "RaceId",
                "Season",
                "RoundNumber",
            ],
        ]
        .drop_duplicates()["RaceId"]
        .tolist()
    )

    assert race_order == [
        "2023_01_bahrain_grand_prix",
        "2023_02_saudi_arabian_grand_prix",
        "2024_01_bahrain_grand_prix",
    ]


def test_build_corpus_requires_artifacts() -> None:
    """A corpus cannot be built from an empty artifact collection."""
    with pytest.raises(
        RaceCorpusBuildError,
        match="At least one race-dataset artifact",
    ):
        build_race_corpus_from_artifacts(())


def test_build_corpus_rejects_non_artifact_item() -> None:
    """Every supplied item must be RaceDatasetArtifacts metadata."""
    invalid_artifacts: Any = (object(),)

    with pytest.raises(
        TypeError,
        match="RaceDatasetArtifacts instance",
    ):
        build_race_corpus_from_artifacts(invalid_artifacts)


def test_build_corpus_rejects_duplicate_artifact(
    tmp_path: Path,
) -> None:
    """The same race artifact must not enter the corpus twice."""
    artifact = save_race_dataset_artifacts(
        make_race_dataset(
            season=2023,
            round_number=1,
            event_name="Bahrain Grand Prix",
        ),
        output_directory=tmp_path,
    )

    with pytest.raises(
        RaceCorpusBuildError,
        match="supplied more than once",
    ):
        build_race_corpus_from_artifacts(
            (
                artifact,
                artifact,
            )
        )


def test_build_corpus_rejects_checksum_mismatch(
    tmp_path: Path,
) -> None:
    """Modified race bytes must be rejected before corpus assembly."""
    artifact = save_race_dataset_artifacts(
        make_race_dataset(
            season=2023,
            round_number=1,
            event_name="Bahrain Grand Prix",
        ),
        output_directory=tmp_path,
    )

    with artifact.dataset_path.open("ab") as file_handle:
        file_handle.write(b"modified-after-manifest")

    with pytest.raises(
        RaceCorpusBuildError,
        match="checksum does not match",
    ):
        build_race_corpus_from_artifacts((artifact,))


def test_build_corpus_rejects_artifact_race_id_mismatch(
    tmp_path: Path,
) -> None:
    """Loaded dataset identity must match artifact metadata."""
    artifact = save_race_dataset_artifacts(
        make_race_dataset(
            season=2023,
            round_number=1,
            event_name="Bahrain Grand Prix",
        ),
        output_directory=tmp_path,
    )

    mismatched = replace(
        artifact,
        race_id="2023_01_different_race",
    )

    with pytest.raises(
        RaceCorpusBuildError,
        match="RaceId in the dataset does not match",
    ):
        build_race_corpus_from_artifacts((mismatched,))


def test_build_corpus_rejects_row_count_mismatch(
    tmp_path: Path,
) -> None:
    """Artifact row-count metadata must match loaded data."""
    artifact = save_race_dataset_artifacts(
        make_race_dataset(
            season=2023,
            round_number=1,
            event_name="Bahrain Grand Prix",
        ),
        output_directory=tmp_path,
    )

    mismatched = replace(
        artifact,
        row_count=artifact.row_count + 1,
    )

    with pytest.raises(
        RaceCorpusBuildError,
        match="row count does not match",
    ):
        build_race_corpus_from_artifacts((mismatched,))


def test_build_corpus_rejects_snapshot_count_mismatch(
    tmp_path: Path,
) -> None:
    """Artifact snapshot metadata must match loaded data."""
    artifact = save_race_dataset_artifacts(
        make_race_dataset(
            season=2023,
            round_number=1,
            event_name="Bahrain Grand Prix",
        ),
        output_directory=tmp_path,
    )

    mismatched = replace(
        artifact,
        snapshot_count=artifact.snapshot_count + 1,
    )

    with pytest.raises(
        RaceCorpusBuildError,
        match="snapshot count does not match",
    ):
        build_race_corpus_from_artifacts((mismatched,))


def test_build_corpus_rejects_driver_count_mismatch(
    tmp_path: Path,
) -> None:
    """Artifact driver-count metadata must match loaded data."""
    artifact = save_race_dataset_artifacts(
        make_race_dataset(
            season=2023,
            round_number=1,
            event_name="Bahrain Grand Prix",
        ),
        output_directory=tmp_path,
    )

    mismatched = replace(
        artifact,
        driver_count=artifact.driver_count + 1,
    )

    with pytest.raises(
        RaceCorpusBuildError,
        match="driver count does not match",
    ):
        build_race_corpus_from_artifacts((mismatched,))


def test_save_corpus_creates_parquet_and_manifest(
    tmp_path: Path,
) -> None:
    """A valid corpus should produce both durable artifacts."""
    input_directory = tmp_path / "races"

    output_directory = tmp_path / "corpus"

    artifacts = save_sample_races(input_directory)

    corpus = build_race_corpus_from_artifacts(artifacts)

    saved = save_race_corpus_artifacts(
        corpus,
        output_directory=output_directory,
    )

    assert saved.corpus_path.is_file()
    assert saved.manifest_path.is_file()

    assert saved.corpus_path.name == ("telemetryx_race_corpus.parquet")

    assert saved.manifest_path.name == ("telemetryx_race_corpus_manifest.json")

    assert saved.race_count == 3
    assert saved.season_count == 2
    assert saved.row_count == 27
    assert saved.snapshot_count == 9
    assert saved.driver_count == 3

    assert saved.first_race_id == ("2023_01_bahrain_grand_prix")

    assert saved.last_race_id == ("2024_01_bahrain_grand_prix")


def test_corpus_parquet_round_trip_preserves_data(
    tmp_path: Path,
) -> None:
    """Saved corpus data should survive a Parquet round trip."""
    race_directory = tmp_path / "races"

    corpus_directory = tmp_path / "corpus"

    race_artifacts = save_sample_races(race_directory)

    corpus = build_race_corpus_from_artifacts(race_artifacts)

    artifacts = save_race_corpus_artifacts(
        corpus,
        output_directory=corpus_directory,
    )

    loaded = load_race_corpus_artifact(artifacts.corpus_path)

    pd.testing.assert_frame_equal(
        loaded,
        corpus,
    )


def test_corpus_manifest_contains_expected_metadata(
    tmp_path: Path,
) -> None:
    """Corpus manifest should describe the complete training corpus."""
    race_directory = tmp_path / "races"

    corpus_directory = tmp_path / "corpus"

    race_artifacts = save_sample_races(race_directory)

    corpus = build_race_corpus_from_artifacts(race_artifacts)

    artifacts = save_race_corpus_artifacts(
        corpus,
        output_directory=corpus_directory,
    )

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == (CORPUS_SCHEMA_VERSION)

    assert manifest["corpus_file"] == ("telemetryx_race_corpus.parquet")

    assert manifest["race_count"] == 3
    assert manifest["season_count"] == 2
    assert manifest["row_count"] == 27
    assert manifest["snapshot_count"] == 9
    assert manifest["driver_count"] == 3

    assert manifest["seasons"] == [
        2023,
        2024,
    ]

    assert manifest["race_ids"] == [
        "2023_01_bahrain_grand_prix",
        "2023_02_saudi_arabian_grand_prix",
        "2024_01_bahrain_grand_prix",
    ]

    assert manifest["columns"] == [str(column) for column in corpus.columns]

    assert manifest["dtypes"] == {
        str(column): str(dtype) for column, dtype in corpus.dtypes.items()
    }

    checksum = manifest["corpus_sha256"]

    assert isinstance(
        checksum,
        str,
    )

    assert len(checksum) == 64


def test_corpus_checksum_matches_unchanged_file(
    tmp_path: Path,
) -> None:
    """An unchanged corpus should match its manifest digest."""
    race_artifacts = save_sample_races(tmp_path / "races")

    corpus = build_race_corpus_from_artifacts(race_artifacts)

    artifacts = save_race_corpus_artifacts(
        corpus,
        output_directory=tmp_path / "corpus",
    )

    assert verify_race_corpus_checksum(artifacts) is True


def test_corpus_checksum_detects_modified_file(
    tmp_path: Path,
) -> None:
    """Changing corpus bytes should invalidate its SHA-256 digest."""
    race_artifacts = save_sample_races(tmp_path / "races")

    corpus = build_race_corpus_from_artifacts(race_artifacts)

    artifacts = save_race_corpus_artifacts(
        corpus,
        output_directory=tmp_path / "corpus",
    )

    with artifacts.corpus_path.open("ab") as file_handle:
        file_handle.write(b"corpus-modification")

    assert verify_race_corpus_checksum(artifacts) is False


def test_existing_corpus_is_protected_by_default(
    tmp_path: Path,
) -> None:
    """A repeated save should not overwrite corpus artifacts accidentally."""
    race_artifacts = save_sample_races(tmp_path / "races")

    corpus = build_race_corpus_from_artifacts(race_artifacts)

    output_directory = tmp_path / "corpus"

    save_race_corpus_artifacts(
        corpus,
        output_directory=output_directory,
    )

    with pytest.raises(
        RaceCorpusBuildError,
        match="already exist",
    ):
        save_race_corpus_artifacts(
            corpus,
            output_directory=output_directory,
        )


def test_overwrite_allows_rebuilding_corpus(
    tmp_path: Path,
) -> None:
    """Explicit overwrite should permit rebuilding corpus artifacts."""
    race_artifacts = save_sample_races(tmp_path / "races")

    corpus = build_race_corpus_from_artifacts(race_artifacts)

    output_directory = tmp_path / "corpus"

    first = save_race_corpus_artifacts(
        corpus,
        output_directory=output_directory,
    )

    second = save_race_corpus_artifacts(
        corpus,
        output_directory=output_directory,
        overwrite=True,
    )

    assert second.corpus_path == first.corpus_path
    assert second.manifest_path == first.manifest_path

    assert verify_race_corpus_checksum(second) is True


def test_save_corpus_rejects_non_dataframe(
    tmp_path: Path,
) -> None:
    """Corpus persistence requires a pandas DataFrame."""
    invalid_corpus: Any = []

    with pytest.raises(
        TypeError,
        match="corpus must be provided as a pandas DataFrame",
    ):
        save_race_corpus_artifacts(
            invalid_corpus,
            output_directory=tmp_path,
        )


def test_save_corpus_wraps_validation_failure(
    tmp_path: Path,
) -> None:
    """Invalid corpus structure should fail before persistence."""
    invalid_corpus = pd.DataFrame(
        {
            "RaceId": [
                "invalid",
            ]
        }
    )

    with pytest.raises(
        RaceCorpusBuildError,
        match="failed validation",
    ):
        save_race_corpus_artifacts(
            invalid_corpus,
            output_directory=tmp_path,
        )


def test_load_corpus_rejects_non_path() -> None:
    """Corpus loading requires a pathlib Path."""
    invalid_path: Any = "corpus.parquet"

    with pytest.raises(
        TypeError,
        match="corpus_path must be provided as a pathlib Path",
    ):
        load_race_corpus_artifact(invalid_path)


def test_load_corpus_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Missing corpus artifacts should produce clear errors."""
    with pytest.raises(
        RaceCorpusBuildError,
        match="does not exist",
    ):
        load_race_corpus_artifact(tmp_path / "missing.parquet")


def test_load_corpus_rejects_invalid_parquet(
    tmp_path: Path,
) -> None:
    """Unreadable Parquet content should be wrapped."""
    corpus_path = tmp_path / "invalid.parquet"

    corpus_path.write_text(
        "not parquet",
        encoding="utf-8",
    )

    with pytest.raises(
        RaceCorpusBuildError,
        match="Could not read race-corpus artifact",
    ):
        load_race_corpus_artifact(corpus_path)


def test_load_corpus_with_validation_disabled(
    tmp_path: Path,
) -> None:
    """Diagnostic loading may bypass corpus structure validation."""
    malformed = pd.DataFrame(
        {
            "Example": [
                1,
                2,
            ]
        }
    )

    corpus_path = tmp_path / "malformed.parquet"

    malformed.to_parquet(
        corpus_path,
        index=False,
    )

    loaded = load_race_corpus_artifact(
        corpus_path,
        validate=False,
    )

    pd.testing.assert_frame_equal(
        loaded,
        malformed,
    )


def test_build_and_save_corpus_runs_complete_pipeline(
    tmp_path: Path,
) -> None:
    """Top-level corpus building should discover, combine and persist."""
    input_directory = tmp_path / "races"

    output_directory = tmp_path / "corpus"

    save_sample_races(input_directory)

    artifacts = build_and_save_race_corpus(
        input_directory=input_directory,
        output_directory=output_directory,
    )

    assert artifacts.race_count == 3
    assert artifacts.season_count == 2
    assert artifacts.row_count == 27

    assert artifacts.corpus_path.is_file()
    assert artifacts.manifest_path.is_file()

    assert verify_race_corpus_checksum(artifacts) is True


def test_build_and_save_corpus_uses_default_directories(
    tmp_path: Path,
) -> None:
    """Project settings should determine default input and output paths."""
    settings = settings_with_temporary_processed_directory(tmp_path)

    race_directory = settings.paths.processed_data_dir / "race_datasets"

    save_sample_races(race_directory)

    artifacts = build_and_save_race_corpus(settings=settings)

    assert (
        artifacts.corpus_path.parent
        == (settings.paths.processed_data_dir / "corpora").resolve()
    )

    assert (
        artifacts.manifest_path.parent
        == (settings.paths.processed_data_dir / "corpora").resolve()
    )


def test_failed_corpus_write_leaves_no_partial_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed Parquet writes must not leave incomplete corpus artifacts."""
    race_artifacts = save_sample_races(tmp_path / "races")

    corpus = build_race_corpus_from_artifacts(race_artifacts)

    output_directory = tmp_path / "corpus"

    def fail_to_parquet(
        self: pd.DataFrame,
        path: Path,
        *,
        index: bool,
    ) -> None:
        del self

        raise OSError(f"Simulated failure: {path}; index={index}")

    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        fail_to_parquet,
    )

    with pytest.raises(
        RaceCorpusBuildError,
        match="Could not save race-corpus artifacts",
    ):
        save_race_corpus_artifacts(
            corpus,
            output_directory=output_directory,
        )

    assert output_directory.is_dir()

    assert list(output_directory.iterdir()) == []


def test_argument_parser_accepts_corpus_options(
    tmp_path: Path,
) -> None:
    """Corpus CLI should expose input, output and overwrite options."""
    parser = create_argument_parser()

    input_directory = tmp_path / "input"

    output_directory = tmp_path / "output"

    arguments = parser.parse_args(
        [
            "--input-directory",
            str(input_directory),
            "--output-directory",
            str(output_directory),
            "--overwrite",
        ]
    )

    assert arguments.input_directory == (input_directory)

    assert arguments.output_directory == (output_directory)

    assert arguments.overwrite is True


def test_argument_parser_uses_optional_defaults() -> None:
    """Corpus command options should remain optional."""
    parser = create_argument_parser()

    arguments = parser.parse_args([])

    assert arguments.input_directory is None
    assert arguments.output_directory is None
    assert arguments.overwrite is False


def test_print_corpus_build_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Terminal summary should expose corpus dimensions and paths."""
    artifacts = RaceCorpusArtifacts(
        corpus_path=(tmp_path / "corpus.parquet"),
        manifest_path=(tmp_path / "manifest.json"),
        race_count=3,
        season_count=2,
        row_count=27,
        snapshot_count=9,
        driver_count=3,
        first_race_id=("2023_01_bahrain_grand_prix"),
        last_race_id=("2024_01_bahrain_grand_prix"),
    )

    print_corpus_build_summary(artifacts)

    output = capsys.readouterr().out

    assert "TelemetryX race corpus" in output
    assert "Races: 3" in output
    assert "Seasons: 2" in output
    assert "Rows: 27" in output
    assert "Snapshots: 9" in output
    assert "Drivers: 3" in output

    assert "First race: 2023_01_bahrain_grand_prix" in output

    assert "Last race: 2024_01_bahrain_grand_prix" in output

    assert str(artifacts.corpus_path) in output

    assert str(artifacts.manifest_path) in output
