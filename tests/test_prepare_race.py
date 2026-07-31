from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import telemetryx.data.prepare_race as prepare_race_module
from telemetryx.config import Settings, load_settings
from telemetryx.data.prepare_race import (
    RacePreparationError,
    create_argument_parser,
    prepare_race_data,
    print_preparation_summary,
)


class FakeSession:
    """Small FastF1-like session used by preparation tests."""

    def __init__(
        self,
        *,
        laps: object | None = None,
        event_name: object = "Bahrain Grand Prix",
        session_name: object = "Race",
    ) -> None:
        """Create a fake loaded race session."""
        self.laps = make_valid_laps() if laps is None else laps
        self.event: dict[str, object] = {
            "EventName": event_name,
        }
        self.name = session_name


def make_valid_laps() -> pd.DataFrame:
    """Return a valid FastF1-like lap table requiring cleaning."""
    return pd.DataFrame(
        {
            "Driver": [
                " ver ",
                "nor",
                " ver ",
                "nor",
            ],
            "LapNumber": [
                1.0,
                1.0,
                2.0,
                2.0,
            ],
            "LapTime": [
                "00:01:38.100",
                "00:01:38.500",
                "00:01:37.900",
                "00:01:38.200",
            ],
            "Position": [
                1.0,
                2.0,
                1.0,
                2.0,
            ],
            "Stint": [
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            "Compound": [
                " soft ",
                "soft",
                " medium ",
                "medium",
            ],
            "TyreLife": [
                1.0,
                1.0,
                2.0,
                2.0,
            ],
            "TrackStatus": [
                " 1 ",
                "1",
                " 1 ",
                "1",
            ],
        }
    )


def settings_with_temporary_interim_directory(
    tmp_path: Path,
) -> Settings:
    """Return project settings using a temporary interim directory."""
    settings = load_settings()

    temporary_paths = replace(
        settings.paths,
        interim_data_dir=tmp_path / "interim",
    )

    return replace(
        settings,
        paths=temporary_paths,
    )


def install_fake_loader(
    monkeypatch: pytest.MonkeyPatch,
    session: FakeSession,
) -> dict[str, object]:
    """Replace the real FastF1 loader and capture its arguments."""
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
        prepare_race_module,
        "load_race_session",
        fake_load_race_session,
    )

    return captured_arguments


def test_prepare_race_saves_cleaned_laps_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preparation should write cleaned Parquet and validation JSON."""
    settings = settings_with_temporary_interim_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(),
    )

    artifacts = prepare_race_data(
        settings=settings,
    )

    expected_directory = (tmp_path / "interim" / "cleaned_races").resolve()

    assert artifacts.cleaned_laps_path == (
        expected_directory / "2024_bahrain_grand_prix_race_laps.parquet"
    )

    assert artifacts.validation_report_path == (
        expected_directory / "2024_bahrain_grand_prix_race_validation.json"
    )

    assert artifacts.cleaned_laps_path.is_file()
    assert artifacts.validation_report_path.is_file()
    assert artifacts.row_count == 4
    assert artifacts.event_name == "Bahrain Grand Prix"


def test_saved_parquet_contains_cleaned_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Parquet artifact should contain standardized lap data."""
    settings = settings_with_temporary_interim_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(),
    )

    artifacts = prepare_race_data(
        settings=settings,
    )

    saved_laps = pd.read_parquet(artifacts.cleaned_laps_path)

    assert saved_laps["Driver"].tolist() == [
        "VER",
        "NOR",
        "VER",
        "NOR",
    ]

    assert saved_laps["Compound"].tolist() == [
        "SOFT",
        "SOFT",
        "MEDIUM",
        "MEDIUM",
    ]

    assert "LapTimeSeconds" in saved_laps.columns

    assert saved_laps["LapTimeSeconds"].tolist() == pytest.approx(
        [
            98.1,
            98.5,
            97.9,
            98.2,
        ]
    )


def test_validation_report_contains_race_and_cleaning_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The JSON artifact should describe validation and cleaned output."""
    settings = settings_with_temporary_interim_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(),
    )

    artifacts = prepare_race_data(
        settings=settings,
    )

    report = json.loads(artifacts.validation_report_path.read_text(encoding="utf-8"))

    assert report["race"] == {
        "season": 2024,
        "event_name": "Bahrain Grand Prix",
        "session_name": "Race",
    }

    assert report["cleaning"]["row_count"] == 4
    assert "LapTimeSeconds" in report["cleaning"]["columns"]

    assert report["input_validation"]["has_errors"] is False

    assert report["output_validation"]["has_errors"] is False


def test_prepare_race_passes_overrides_to_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit race arguments should be forwarded to the loader."""
    settings = settings_with_temporary_interim_directory(tmp_path)

    captured_arguments = install_fake_loader(
        monkeypatch,
        FakeSession(
            event_name="Italian Grand Prix",
        ),
    )

    output_directory = tmp_path / "custom-output"

    artifacts = prepare_race_data(
        settings=settings,
        season=2023,
        event="Monza",
        session_type="R",
        output_directory=output_directory,
    )

    assert captured_arguments == {
        "settings": settings,
        "season": 2023,
        "event": "Monza",
        "session_type": "R",
    }

    assert artifacts.cleaned_laps_path.parent == (output_directory.resolve())

    assert artifacts.cleaned_laps_path.name == (
        "2023_italian_grand_prix_race_laps.parquet"
    )


def test_existing_artifacts_are_protected_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second run should not replace artifacts without permission."""
    settings = settings_with_temporary_interim_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(),
    )

    prepare_race_data(
        settings=settings,
    )

    with pytest.raises(
        RacePreparationError,
        match="already exist",
    ):
        prepare_race_data(
            settings=settings,
        )


def test_overwrite_replaces_existing_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit overwrite mode should permit repeated preparation."""
    settings = settings_with_temporary_interim_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(),
    )

    first_result = prepare_race_data(
        settings=settings,
    )

    second_result = prepare_race_data(
        settings=settings,
        overwrite=True,
    )

    assert second_result.cleaned_laps_path == first_result.cleaned_laps_path

    assert second_result.validation_report_path == first_result.validation_report_path

    assert second_result.cleaned_laps_path.is_file()
    assert second_result.validation_report_path.is_file()


def test_non_dataframe_laps_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preparation should reject sessions without a lap DataFrame."""
    settings = settings_with_temporary_interim_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(
            laps=["not", "a", "dataframe"],
        ),
    )

    with pytest.raises(
        RacePreparationError,
        match="does not contain a lap DataFrame",
    ):
        prepare_race_data(
            settings=settings,
        )


def test_empty_lap_table_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preparation should reject an empty loaded lap table."""
    settings = settings_with_temporary_interim_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(
            laps=make_valid_laps().iloc[0:0],
        ),
    )

    with pytest.raises(
        RacePreparationError,
        match="lap table is empty",
    ):
        prepare_race_data(
            settings=settings,
        )


def test_missing_loaded_names_use_argument_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing FastF1 names should fall back to requested identifiers."""
    settings = settings_with_temporary_interim_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(
            event_name="   ",
            session_name="   ",
        ),
    )

    artifacts = prepare_race_data(
        settings=settings,
        season=2024,
        event=7,
        session_type="R",
    )

    assert artifacts.event_name == "7"

    assert artifacts.cleaned_laps_path.name == ("2024_7_r_laps.parquet")

    report = json.loads(artifacts.validation_report_path.read_text(encoding="utf-8"))

    assert report["race"]["event_name"] == "7"
    assert report["race"]["session_name"] == "R"


def test_failed_parquet_write_does_not_leave_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed write should not leave incomplete output files."""
    settings = settings_with_temporary_interim_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(),
    )

    def fail_to_parquet(
        self: pd.DataFrame,
        path: Path,
        *,
        index: bool,
    ) -> None:
        raise OSError(f"Simulated Parquet failure for {path}; index={index}")

    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        fail_to_parquet,
    )

    with pytest.raises(
        RacePreparationError,
        match="Could not save cleaned race artifacts",
    ):
        prepare_race_data(
            settings=settings,
        )

    output_directory = tmp_path / "interim" / "cleaned_races"

    if output_directory.exists():
        assert list(output_directory.iterdir()) == []


def test_argument_parser_accepts_race_overrides(
    tmp_path: Path,
) -> None:
    """The command-line parser should interpret preparation arguments."""
    parser = create_argument_parser()

    arguments = parser.parse_args(
        [
            "--season",
            "2023",
            "--event",
            "5",
            "--session-type",
            "R",
            "--output-directory",
            str(tmp_path),
            "--overwrite",
        ]
    )

    assert arguments.season == 2023
    assert arguments.event == 5
    assert arguments.session_type == "R"
    assert arguments.output_directory == tmp_path
    assert arguments.overwrite is True


def test_print_preparation_summary_displays_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The terminal summary should display generated artifact paths."""
    settings = settings_with_temporary_interim_directory(tmp_path)

    install_fake_loader(
        monkeypatch,
        FakeSession(),
    )

    artifacts = prepare_race_data(
        settings=settings,
    )

    print_preparation_summary(artifacts)

    captured_output = capsys.readouterr().out

    assert "TelemetryX race preparation" in captured_output
    assert "Race: Bahrain Grand Prix" in captured_output
    assert "Cleaned rows: 4" in captured_output
    assert str(artifacts.cleaned_laps_path) in captured_output
    assert str(artifacts.validation_report_path) in captured_output
