from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import telemetryx.data.build_dataset as build_dataset_module
from telemetryx.config import Settings, load_settings
from telemetryx.data.build_dataset import (
    RaceDatasetBuildError,
    build_and_save_race_dataset,
    create_argument_parser,
    print_build_summary,
)
from telemetryx.data.dataset_artifacts import (
    load_race_dataset_artifact,
    verify_race_dataset_checksum,
)


class FakeSession:
    """Small FastF1-like loaded race session for orchestration tests."""

    def __init__(
        self,
        *,
        laps: object | None = None,
        results: object | None = None,
        round_number: object = 1,
        event_name: object = "Bahrain Grand Prix",
        session_name: object = "Race",
    ) -> None:
        """Create a fake session with configurable race data."""
        self.laps = make_raw_laps() if laps is None else laps

        self.results = make_results() if results is None else results

        self.event: dict[str, object] = {
            "RoundNumber": round_number,
            "EventName": event_name,
        }

        self.name = session_name


def make_raw_laps() -> pd.DataFrame:
    """
    Return valid FastF1-like laps that still require cleaning.

    VER and NOR complete three laps while BOT completes two.
    """
    return pd.DataFrame(
        {
            "Driver": [
                " ver ",
                "VER",
                " ver ",
                "nor",
                " NOR ",
                "nor",
                " bot ",
                "BOT",
            ],
            "LapNumber": [
                1.0,
                2.0,
                3.0,
                1.0,
                2.0,
                3.0,
                1.0,
                2.0,
            ],
            "LapTime": [
                "00:01:30",
                "00:01:29",
                "00:01:28",
                "00:01:31",
                "00:01:30",
                "00:01:29",
                "00:01:32",
                "00:01:31",
            ],
            "Position": [
                1.0,
                1.0,
                1.0,
                2.0,
                2.0,
                2.0,
                3.0,
                3.0,
            ],
            "Stint": [
                1.0,
                1.0,
                2.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            "Compound": [
                " soft ",
                "SOFT",
                " medium ",
                "soft",
                " SOFT ",
                "soft",
                " soft ",
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
                " 1 ",
                "1",
                "1 ",
                "1",
                " 1",
                "1",
                "1 ",
                " 1 ",
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
                1.0,
                2.0,
                3.0,
            ],
            "Status": [
                "Finished",
                "Finished",
                "+1 Lap",
            ],
            "Points": [
                25.0,
                18.0,
                15.0,
            ],
        }
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


def install_fake_loader(
    monkeypatch: pytest.MonkeyPatch,
    session: FakeSession,
) -> dict[str, object]:
    """Replace FastF1 loading and capture requested race arguments."""
    captured_arguments: dict[str, object] = {}

    def fake_load_race_session(
        settings: Settings,
        *,
        season: int,
        event: str | int,
        session_type: str,
    ) -> FakeSession:
        captured_arguments.update(
            {
                "settings": settings,
                "season": season,
                "event": event,
                "session_type": session_type,
            }
        )

        return session

    monkeypatch.setattr(
        build_dataset_module,
        "load_race_session",
        fake_load_race_session,
    )

    return captured_arguments


def test_build_and_save_creates_processed_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The complete pipeline should create Parquet and manifest artifacts."""
    settings = settings_with_temporary_processed_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(),
    )

    artifacts = build_and_save_race_dataset(
        settings=settings,
    )

    expected_directory = (tmp_path / "processed" / "race_datasets").resolve()

    assert artifacts.race_id == ("2024_01_bahrain_grand_prix")

    assert artifacts.dataset_path == (
        expected_directory / "2024_01_bahrain_grand_prix_dataset.parquet"
    )

    assert artifacts.manifest_path == (
        expected_directory / "2024_01_bahrain_grand_prix_manifest.json"
    )

    assert artifacts.dataset_path.is_file()
    assert artifacts.manifest_path.is_file()

    assert artifacts.row_count == 9
    assert artifacts.snapshot_count == 3
    assert artifacts.driver_count == 3

    assert verify_race_dataset_checksum(artifacts) is True


def test_saved_dataset_contains_cleaned_replay_and_target_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The saved dataset should connect cleaning, replay and target stages."""
    settings = settings_with_temporary_processed_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(),
    )

    artifacts = build_and_save_race_dataset(
        settings=settings,
    )

    dataset = load_race_dataset_artifact(artifacts.dataset_path)

    assert set(dataset["Driver"]) == {
        "VER",
        "NOR",
        "BOT",
    }

    assert set(dataset["Compound"].dropna()) == {
        "SOFT",
        "MEDIUM",
    }

    assert dataset["SnapshotLap"].unique().tolist() == [
        1,
        2,
        3,
    ]

    assert "LastLapTimeSeconds" in dataset.columns
    assert "CumulativeLapTimeSeconds" in dataset.columns
    assert "WonRace" in dataset.columns

    winner_rows = dataset.loc[dataset["WonRace"]]

    assert set(winner_rows["Driver"]) == {"VER"}

    assert len(winner_rows) == 3


def test_saved_dataset_does_not_include_final_result_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-race result columns should not leak into processed features."""
    settings = settings_with_temporary_processed_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(),
    )

    artifacts = build_and_save_race_dataset(
        settings=settings,
    )

    dataset = load_race_dataset_artifact(artifacts.dataset_path)

    assert "Points" not in dataset.columns
    assert "Status" not in dataset.columns
    assert "FinalPosition" not in dataset.columns
    assert "ClassifiedPosition" not in dataset.columns


def test_manifest_describes_end_to_end_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The saved manifest should describe the resulting race dataset."""
    settings = settings_with_temporary_processed_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(),
    )

    artifacts = build_and_save_race_dataset(
        settings=settings,
    )

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))

    assert manifest["race_id"] == ("2024_01_bahrain_grand_prix")

    assert manifest["row_count"] == 9
    assert manifest["snapshot_count"] == 3
    assert manifest["driver_count"] == 3
    assert manifest["snapshot_lap_min"] == 1
    assert manifest["snapshot_lap_max"] == 3
    assert manifest["winner_row_count"] == 3
    assert manifest["target_column"] == "WonRace"


def test_explicit_race_arguments_are_forwarded_to_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Command overrides should be passed unchanged to FastF1 loading."""
    settings = settings_with_temporary_processed_directory(tmp_path)

    captured_arguments = install_fake_loader(
        monkeypatch,
        FakeSession(
            round_number=14,
            event_name="Italian Grand Prix",
        ),
    )

    artifacts = build_and_save_race_dataset(
        settings=settings,
        season=2023,
        event="Monza",
        session_type="R",
    )

    assert captured_arguments == {
        "settings": settings,
        "season": 2023,
        "event": "Monza",
        "session_type": "R",
    }

    assert artifacts.race_id == ("2023_14_italian_grand_prix")


def test_requested_snapshot_range_is_forwarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Start and end lap options should restrict the processed replay."""
    settings = settings_with_temporary_processed_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(),
    )

    artifacts = build_and_save_race_dataset(
        settings=settings,
        start_lap=2,
        end_lap=2,
    )

    dataset = load_race_dataset_artifact(artifacts.dataset_path)

    assert dataset["SnapshotLap"].unique().tolist() == [2]

    assert artifacts.snapshot_count == 1
    assert artifacts.row_count == 3
    assert int(dataset["WonRace"].sum()) == 1


def test_custom_output_directory_is_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit output directory should override project settings."""
    settings = settings_with_temporary_processed_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(),
    )

    custom_directory = tmp_path / "custom-datasets"

    artifacts = build_and_save_race_dataset(
        settings=settings,
        output_directory=custom_directory,
    )

    assert artifacts.dataset_path.parent == (custom_directory.resolve())

    assert artifacts.manifest_path.parent == (custom_directory.resolve())


def test_existing_artifacts_are_protected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated builds should not overwrite processed data by default."""
    settings = settings_with_temporary_processed_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(),
    )

    build_and_save_race_dataset(
        settings=settings,
    )

    with pytest.raises(
        RuntimeError,
        match="already exist",
    ):
        build_and_save_race_dataset(
            settings=settings,
        )


def test_overwrite_allows_rebuilding_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit overwrite mode should permit rebuilding one race."""
    settings = settings_with_temporary_processed_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(),
    )

    first = build_and_save_race_dataset(
        settings=settings,
    )

    second = build_and_save_race_dataset(
        settings=settings,
        overwrite=True,
    )

    assert second.dataset_path == first.dataset_path
    assert second.manifest_path == first.manifest_path

    assert verify_race_dataset_checksum(second) is True


def test_event_name_falls_back_to_requested_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing loaded event names should use the requested event identifier."""
    settings = settings_with_temporary_processed_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(
            event_name="   ",
        ),
    )

    artifacts = build_and_save_race_dataset(
        settings=settings,
        event="Monza",
    )

    assert artifacts.race_id == ("2024_01_monza")


def test_session_name_falls_back_to_requested_session_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing loaded session names should use the requested session type."""
    settings = settings_with_temporary_processed_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(
            session_name="   ",
        ),
    )

    artifacts = build_and_save_race_dataset(
        settings=settings,
        session_type="R",
    )

    dataset = load_race_dataset_artifact(artifacts.dataset_path)

    assert dataset["SessionName"].unique().tolist() == ["R"]


@pytest.mark.parametrize(
    "round_number",
    [
        None,
        pd.NA,
        0,
        -1,
        1.5,
        "not-a-round",
        "",
        "   ",
        True,
    ],
)
def test_invalid_round_number_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    round_number: object,
) -> None:
    """Loaded event metadata must contain a positive whole round number."""
    settings = settings_with_temporary_processed_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(
            round_number=round_number,
        ),
    )

    with pytest.raises(
        RaceDatasetBuildError,
        match="valid positive RoundNumber",
    ):
        build_and_save_race_dataset(
            settings=settings,
        )


@pytest.mark.parametrize(
    ("round_number", "expected_race_id"),
    [
        (
            1,
            "2024_01_bahrain_grand_prix",
        ),
        (
            1.0,
            "2024_01_bahrain_grand_prix",
        ),
        (
            "1",
            "2024_01_bahrain_grand_prix",
        ),
        (
            " 1 ",
            "2024_01_bahrain_grand_prix",
        ),
    ],
)
def test_supported_round_number_types_are_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    round_number: object,
    expected_race_id: str,
) -> None:
    """Common scalar representations of whole round numbers should work."""
    settings = settings_with_temporary_processed_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(
            round_number=round_number,
        ),
    )

    artifacts = build_and_save_race_dataset(
        settings=settings,
    )

    assert artifacts.race_id == expected_race_id


def test_missing_lap_dataframe_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestration layer requires loaded lap data."""
    settings = settings_with_temporary_processed_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(
            laps="not-a-dataframe",
        ),
    )

    with pytest.raises(
        RaceDatasetBuildError,
        match="does not contain a laps DataFrame",
    ):
        build_and_save_race_dataset(
            settings=settings,
        )


def test_empty_lap_dataframe_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty loaded lap table cannot produce a race dataset."""
    settings = settings_with_temporary_processed_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(
            laps=make_raw_laps().iloc[0:0],
        ),
    )

    with pytest.raises(
        RaceDatasetBuildError,
        match="laps table is empty",
    ):
        build_and_save_race_dataset(
            settings=settings,
        )


def test_missing_results_dataframe_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestration layer requires final results for winner labels."""
    settings = settings_with_temporary_processed_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(
            results="not-a-dataframe",
        ),
    )

    with pytest.raises(
        RaceDatasetBuildError,
        match="does not contain a results DataFrame",
    ):
        build_and_save_race_dataset(
            settings=settings,
        )


def test_empty_results_dataframe_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty final-results table cannot define winner targets."""
    settings = settings_with_temporary_processed_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(
            results=make_results().iloc[0:0],
        ),
    )

    with pytest.raises(
        RaceDatasetBuildError,
        match="results table is empty",
    ):
        build_and_save_race_dataset(
            settings=settings,
        )


def test_source_session_data_is_not_modified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Building should not mutate the DataFrames held by the loaded session."""
    settings = settings_with_temporary_processed_directory(tmp_path)

    session = FakeSession()

    assert isinstance(
        session.laps,
        pd.DataFrame,
    )

    assert isinstance(
        session.results,
        pd.DataFrame,
    )

    original_laps = session.laps.copy(deep=True)

    original_results = session.results.copy(deep=True)

    install_fake_loader(
        monkeypatch,
        session,
    )

    build_and_save_race_dataset(
        settings=settings,
    )

    pd.testing.assert_frame_equal(
        session.laps,
        original_laps,
    )

    pd.testing.assert_frame_equal(
        session.results,
        original_results,
    )


def test_argument_parser_accepts_dataset_build_options(
    tmp_path: Path,
) -> None:
    """The command parser should understand race and replay overrides."""
    parser = create_argument_parser()

    arguments = parser.parse_args(
        [
            "--season",
            "2023",
            "--event",
            "14",
            "--session-type",
            "R",
            "--start-lap",
            "5",
            "--end-lap",
            "40",
            "--output-directory",
            str(tmp_path),
            "--overwrite",
        ]
    )

    assert arguments.season == 2023
    assert arguments.event == 14
    assert arguments.session_type == "R"
    assert arguments.start_lap == 5
    assert arguments.end_lap == 40
    assert arguments.output_directory == tmp_path
    assert arguments.overwrite is True


def test_argument_parser_accepts_named_event() -> None:
    """The command parser should preserve textual event identifiers."""
    parser = create_argument_parser()

    arguments = parser.parse_args(
        [
            "--event",
            "Monza",
        ]
    )

    assert arguments.event == "Monza"


def test_print_build_summary_displays_artifact_information(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The terminal summary should show useful build information."""
    settings = settings_with_temporary_processed_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(),
    )

    artifacts = build_and_save_race_dataset(
        settings=settings,
    )

    print_build_summary(artifacts)

    output = capsys.readouterr().out

    assert "TelemetryX race dataset" in output
    assert "Race ID: 2024_01_bahrain_grand_prix" in output
    assert "Rows: 9" in output
    assert "Snapshots: 3" in output
    assert "Drivers: 3" in output
    assert str(artifacts.dataset_path) in output
    assert str(artifacts.manifest_path) in output
