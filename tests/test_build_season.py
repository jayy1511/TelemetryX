from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import telemetryx.data.build_season as build_season_module
from telemetryx.config import Settings, load_settings
from telemetryx.data.build_season import (
    RaceBuildFailure,
    ScheduledRace,
    SeasonBuildResult,
    SeasonDatasetBuildError,
    build_season_datasets,
    create_argument_parser,
    load_season_schedule,
    parse_rounds_argument,
    print_season_build_summary,
    select_scheduled_races,
)
from telemetryx.data.dataset_artifacts import RaceDatasetArtifacts


def make_schedule() -> pd.DataFrame:
    """
    Return a small FastF1-like event schedule.

    Rows are intentionally not chronological so selection must sort them.
    Round zero represents a non-championship schedule row and should be
    ignored by race selection.
    """
    return pd.DataFrame(
        {
            "RoundNumber": [
                0,
                3,
                1,
                2,
            ],
            "EventName": [
                "Pre-Season Testing",
                "Australian Grand Prix",
                "Bahrain Grand Prix",
                "Saudi Arabian Grand Prix",
            ],
        }
    )


def make_artifacts(
    *,
    season: int,
    round_number: int,
    event_slug: str,
) -> RaceDatasetArtifacts:
    """Return fake saved-artifact metadata for one successful build."""
    race_id = f"{season}_{round_number:02d}_{event_slug}"

    return RaceDatasetArtifacts(
        dataset_path=Path(f"/tmp/{race_id}_dataset.parquet"),
        manifest_path=Path(f"/tmp/{race_id}_manifest.json"),
        race_id=race_id,
        row_count=100,
        snapshot_count=10,
        driver_count=20,
    )


def install_fake_schedule_loader(
    monkeypatch: pytest.MonkeyPatch,
    schedule: pd.DataFrame | None = None,
) -> list[int]:
    """Replace schedule loading and record requested seasons."""
    requested_seasons: list[int] = []

    selected_schedule = make_schedule() if schedule is None else schedule

    def fake_load_season_schedule(
        season: int,
    ) -> pd.DataFrame:
        requested_seasons.append(season)

        return selected_schedule.copy(deep=True)

    monkeypatch.setattr(
        build_season_module,
        "load_season_schedule",
        fake_load_season_schedule,
    )

    return requested_seasons


def install_successful_race_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, object]]:
    """Replace single-race building and capture every invocation."""
    calls: list[dict[str, object]] = []

    event_slugs = {
        1: "bahrain_grand_prix",
        2: "saudi_arabian_grand_prix",
        3: "australian_grand_prix",
    }

    def fake_build_and_save_race_dataset(
        settings: Settings,
        *,
        season: int,
        event: str | int,
        session_type: str,
        start_lap: int,
        end_lap: int | None,
        output_directory: Path | None,
        overwrite: bool,
    ) -> RaceDatasetArtifacts:
        calls.append(
            {
                "settings": settings,
                "season": season,
                "event": event,
                "session_type": session_type,
                "start_lap": start_lap,
                "end_lap": end_lap,
                "output_directory": output_directory,
                "overwrite": overwrite,
            }
        )

        if not isinstance(
            event,
            int,
        ):
            raise AssertionError("Season builds should use championship round numbers.")

        return make_artifacts(
            season=season,
            round_number=event,
            event_slug=event_slugs[event],
        )

    monkeypatch.setattr(
        build_season_module,
        "build_and_save_race_dataset",
        fake_build_and_save_race_dataset,
    )

    return calls


def test_select_scheduled_races_returns_chronological_races() -> None:
    """Schedule rows should become ordered championship race specifications."""
    races = select_scheduled_races(
        make_schedule(),
        season=2023,
    )

    assert races == (
        ScheduledRace(
            season=2023,
            round_number=1,
            event_name="Bahrain Grand Prix",
        ),
        ScheduledRace(
            season=2023,
            round_number=2,
            event_name="Saudi Arabian Grand Prix",
        ),
        ScheduledRace(
            season=2023,
            round_number=3,
            event_name="Australian Grand Prix",
        ),
    )


def test_select_scheduled_races_filters_requested_rounds() -> None:
    """Only explicitly requested championship rounds should be returned."""
    races = select_scheduled_races(
        make_schedule(),
        season=2023,
        rounds=(
            3,
            1,
        ),
    )

    assert [race.round_number for race in races] == [
        1,
        3,
    ]


def test_round_zero_schedule_rows_are_ignored() -> None:
    """Non-positive round numbers should not become race builds."""
    races = select_scheduled_races(
        make_schedule(),
        season=2023,
    )

    assert all(race.round_number > 0 for race in races)

    assert all(race.event_name != "Pre-Season Testing" for race in races)


def test_select_scheduled_races_rejects_non_dataframe() -> None:
    """Schedule selection requires a pandas DataFrame."""
    invalid_schedule: Any = []

    with pytest.raises(
        TypeError,
        match="schedule must be provided as a pandas DataFrame",
    ):
        select_scheduled_races(
            invalid_schedule,
            season=2023,
        )


def test_select_scheduled_races_requires_columns() -> None:
    """Round number and event name columns are mandatory."""
    schedule = pd.DataFrame(
        {
            "RoundNumber": [
                1,
            ],
        }
    )

    with pytest.raises(
        SeasonDatasetBuildError,
        match="missing required columns",
    ) as exception_info:
        select_scheduled_races(
            schedule,
            season=2023,
        )

    assert "EventName" in str(exception_info.value)


def test_invalid_event_name_is_rejected() -> None:
    """Championship race rows must contain usable event names."""
    schedule = pd.DataFrame(
        {
            "RoundNumber": [
                1,
            ],
            "EventName": [
                "   ",
            ],
        }
    )

    with pytest.raises(
        SeasonDatasetBuildError,
        match="invalid EventName",
    ):
        select_scheduled_races(
            schedule,
            season=2023,
        )


def test_duplicate_schedule_round_is_rejected() -> None:
    """A championship schedule cannot contain the same round twice."""
    schedule = pd.DataFrame(
        {
            "RoundNumber": [
                1,
                1,
            ],
            "EventName": [
                "Bahrain Grand Prix",
                "Different Grand Prix",
            ],
        }
    )

    with pytest.raises(
        SeasonDatasetBuildError,
        match="duplicate championship round 1",
    ):
        select_scheduled_races(
            schedule,
            season=2023,
        )


def test_schedule_without_race_rounds_is_rejected() -> None:
    """A schedule containing no positive championship rounds is unusable."""
    schedule = pd.DataFrame(
        {
            "RoundNumber": [
                0,
                -1,
            ],
            "EventName": [
                "Testing",
                "Other",
            ],
        }
    )

    with pytest.raises(
        SeasonDatasetBuildError,
        match="No race rounds were found",
    ):
        select_scheduled_races(
            schedule,
            season=2023,
        )


def test_unavailable_requested_round_is_rejected() -> None:
    """Requested rounds must exist in the discovered schedule."""
    with pytest.raises(
        SeasonDatasetBuildError,
        match="Requested rounds are unavailable",
    ) as exception_info:
        select_scheduled_races(
            make_schedule(),
            season=2023,
            rounds=(
                1,
                99,
            ),
        )

    assert "99" in str(exception_info.value)


def test_load_season_schedule_calls_fastf1_without_testing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schedule loading should explicitly exclude testing events."""
    original_schedule = make_schedule()

    captured_arguments: dict[
        str,
        object,
    ] = {}

    def fake_get_event_schedule(
        year: int,
        *,
        include_testing: bool,
    ) -> pd.DataFrame:
        captured_arguments.update(
            {
                "year": year,
                "include_testing": include_testing,
            }
        )

        return original_schedule

    monkeypatch.setattr(
        build_season_module.fastf1,
        "get_event_schedule",
        fake_get_event_schedule,
    )

    loaded = load_season_schedule(2023)

    assert captured_arguments == {
        "year": 2023,
        "include_testing": False,
    }

    pd.testing.assert_frame_equal(
        loaded,
        original_schedule,
    )

    assert loaded is not original_schedule


def test_load_season_schedule_returns_independent_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing the returned schedule must not mutate FastF1's object."""
    original_schedule = make_schedule()

    def fake_get_event_schedule(
        year: int,
        *,
        include_testing: bool,
    ) -> pd.DataFrame:
        del year
        del include_testing

        return original_schedule

    monkeypatch.setattr(
        build_season_module.fastf1,
        "get_event_schedule",
        fake_get_event_schedule,
    )

    loaded = load_season_schedule(2023)

    loaded.loc[
        1,
        "EventName",
    ] = "Modified Event"

    assert (
        original_schedule.loc[
            1,
            "EventName",
        ]
        == "Australian Grand Prix"
    )


def test_load_season_schedule_wraps_fastf1_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FastF1 schedule failures should become season-build errors."""

    def fail_get_event_schedule(
        year: int,
        *,
        include_testing: bool,
    ) -> pd.DataFrame:
        del year
        del include_testing

        raise RuntimeError("network unavailable")

    monkeypatch.setattr(
        build_season_module.fastf1,
        "get_event_schedule",
        fail_get_event_schedule,
    )

    with pytest.raises(
        SeasonDatasetBuildError,
        match="Could not load the FastF1 event schedule",
    ):
        load_season_schedule(2023)


def test_load_season_schedule_rejects_empty_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty FastF1 schedule cannot produce a season build."""
    monkeypatch.setattr(
        build_season_module.fastf1,
        "get_event_schedule",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    with pytest.raises(
        SeasonDatasetBuildError,
        match="empty event schedule",
    ):
        load_season_schedule(2023)


def test_load_season_schedule_rejects_missing_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FastF1 schedules must contain the required race identity fields."""
    schedule = pd.DataFrame(
        {
            "RoundNumber": [
                1,
            ],
        }
    )

    monkeypatch.setattr(
        build_season_module.fastf1,
        "get_event_schedule",
        lambda *args, **kwargs: schedule,
    )

    with pytest.raises(
        SeasonDatasetBuildError,
        match="missing required columns",
    ):
        load_season_schedule(2023)


def test_build_season_builds_every_selected_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every discovered championship race should reach the race builder."""
    settings = load_settings()

    requested_seasons = install_fake_schedule_loader(monkeypatch)

    build_calls = install_successful_race_builder(monkeypatch)

    result = build_season_datasets(
        season=2023,
        settings=settings,
    )

    assert requested_seasons == [2023]

    assert [call["event"] for call in build_calls] == [
        1,
        2,
        3,
    ]

    assert result.season == 2023
    assert result.requested_rounds == (
        1,
        2,
        3,
    )

    assert result.attempted_count == 3
    assert result.success_count == 3
    assert result.failure_count == 0
    assert result.succeeded is True

    assert len(result.successful_artifacts) == 3


def test_build_season_only_builds_requested_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit round selection should limit race builds."""
    settings = load_settings()

    install_fake_schedule_loader(monkeypatch)

    build_calls = install_successful_race_builder(monkeypatch)

    result = build_season_datasets(
        season=2023,
        settings=settings,
        rounds=[
            3,
            1,
        ],
    )

    assert [call["event"] for call in build_calls] == [
        1,
        3,
    ]

    assert result.requested_rounds == (
        1,
        3,
    )

    assert result.success_count == 2


def test_build_options_are_forwarded_to_every_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay and artifact options should reach the single-race builder."""
    settings = load_settings()

    install_fake_schedule_loader(monkeypatch)

    build_calls = install_successful_race_builder(monkeypatch)

    build_season_datasets(
        season=2023,
        settings=settings,
        rounds=[
            1,
            2,
        ],
        session_type=" R ",
        start_lap=5,
        end_lap=25,
        output_directory=tmp_path,
        overwrite=True,
    )

    assert len(build_calls) == 2

    for call in build_calls:
        assert call["settings"] is settings
        assert call["season"] == 2023
        assert call["session_type"] == "R"
        assert call["start_lap"] == 5
        assert call["end_lap"] == 25
        assert call["output_directory"] == tmp_path
        assert call["overwrite"] is True


def test_build_continues_after_failure_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One bad race should not prevent later races from being attempted."""
    settings = load_settings()

    install_fake_schedule_loader(monkeypatch)

    attempted_rounds: list[int] = []

    def fake_builder(
        settings: Settings,
        *,
        season: int,
        event: str | int,
        session_type: str,
        start_lap: int,
        end_lap: int | None,
        output_directory: Path | None,
        overwrite: bool,
    ) -> RaceDatasetArtifacts:
        del settings
        del session_type
        del start_lap
        del end_lap
        del output_directory
        del overwrite

        assert isinstance(
            event,
            int,
        )

        attempted_rounds.append(event)

        if event == 2:
            raise RuntimeError("simulated round failure")

        return make_artifacts(
            season=season,
            round_number=event,
            event_slug=f"round_{event}",
        )

    monkeypatch.setattr(
        build_season_module,
        "build_and_save_race_dataset",
        fake_builder,
    )

    result = build_season_datasets(
        season=2023,
        settings=settings,
    )

    assert attempted_rounds == [
        1,
        2,
        3,
    ]

    assert result.attempted_count == 3
    assert result.success_count == 2
    assert result.failure_count == 1
    assert result.succeeded is False

    failure = result.failures[0]

    assert failure.season == 2023
    assert failure.round_number == 2
    assert failure.event_name == ("Saudi Arabian Grand Prix")
    assert failure.error_type == ("RuntimeError")
    assert failure.message == ("simulated round failure")


def test_stop_on_error_prevents_later_races(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict mode should stop immediately when a race build fails."""
    settings = load_settings()

    install_fake_schedule_loader(monkeypatch)

    attempted_rounds: list[int] = []

    def fake_builder(
        settings: Settings,
        *,
        season: int,
        event: str | int,
        session_type: str,
        start_lap: int,
        end_lap: int | None,
        output_directory: Path | None,
        overwrite: bool,
    ) -> RaceDatasetArtifacts:
        del settings
        del session_type
        del start_lap
        del end_lap
        del output_directory
        del overwrite

        assert isinstance(
            event,
            int,
        )

        attempted_rounds.append(event)

        if event == 2:
            raise RuntimeError("round two failed")

        return make_artifacts(
            season=season,
            round_number=event,
            event_slug=f"round_{event}",
        )

    monkeypatch.setattr(
        build_season_module,
        "build_and_save_race_dataset",
        fake_builder,
    )

    with pytest.raises(
        SeasonDatasetBuildError,
        match="Season build stopped after a race failure",
    ) as exception_info:
        build_season_datasets(
            season=2023,
            settings=settings,
            continue_on_error=False,
        )

    assert attempted_rounds == [
        1,
        2,
    ]

    assert isinstance(
        exception_info.value.__cause__,
        RuntimeError,
    )


@pytest.mark.parametrize(
    "season",
    [
        0,
        -1,
        True,
    ],
)
def test_build_season_rejects_invalid_season(
    season: int,
) -> None:
    """Season identifiers must be positive integers."""
    with pytest.raises(
        SeasonDatasetBuildError,
        match="season must be a positive integer",
    ):
        build_season_datasets(
            season=season,
            settings=load_settings(),
        )


@pytest.mark.parametrize(
    "rounds",
    [
        [],
        [
            1,
            1,
        ],
        [
            0,
        ],
        [
            -1,
        ],
        [
            True,
        ],
    ],
)
def test_build_season_rejects_invalid_requested_rounds(
    rounds: list[int],
) -> None:
    """Explicit round collections must contain unique positive integers."""
    with pytest.raises(
        SeasonDatasetBuildError,
    ):
        build_season_datasets(
            season=2023,
            settings=load_settings(),
            rounds=rounds,
        )


def test_build_season_rejects_blank_session_type() -> None:
    """The requested FastF1 session identifier cannot be blank."""
    with pytest.raises(
        SeasonDatasetBuildError,
        match="session_type cannot be blank",
    ):
        build_season_datasets(
            season=2023,
            settings=load_settings(),
            session_type="   ",
        )


def test_parse_rounds_argument_parses_and_sorts_values() -> None:
    """CLI round values should become a sorted integer tuple."""
    assert parse_rounds_argument("3, 1,2") == (
        1,
        2,
        3,
    )


@pytest.mark.parametrize(
    "raw_value",
    [
        "",
        " ",
        "1,",
        ",1",
        "1,,2",
        "abc",
        "1,abc",
        "0",
        "-1",
        "1,1",
    ],
)
def test_parse_rounds_argument_rejects_invalid_values(
    raw_value: str,
) -> None:
    """Malformed CLI round lists should produce argparse errors."""
    with pytest.raises(
        argparse.ArgumentTypeError,
    ):
        parse_rounds_argument(raw_value)


def test_argument_parser_accepts_season_build_options(
    tmp_path: Path,
) -> None:
    """The CLI parser should expose the batch-build configuration."""
    parser = create_argument_parser()

    arguments = parser.parse_args(
        [
            "--season",
            "2023",
            "--rounds",
            "3,1,2",
            "--session-type",
            "R",
            "--start-lap",
            "5",
            "--end-lap",
            "40",
            "--output-directory",
            str(tmp_path),
            "--overwrite",
            "--stop-on-error",
        ]
    )

    assert arguments.season == 2023
    assert arguments.rounds == (
        1,
        2,
        3,
    )
    assert arguments.session_type == "R"
    assert arguments.start_lap == 5
    assert arguments.end_lap == 40
    assert arguments.output_directory == tmp_path
    assert arguments.overwrite is True
    assert arguments.stop_on_error is True


def test_season_build_result_properties() -> None:
    """Season result convenience properties should reflect its contents."""
    success = make_artifacts(
        season=2023,
        round_number=1,
        event_slug="bahrain",
    )

    failure = RaceBuildFailure(
        season=2023,
        round_number=2,
        event_name="Saudi Arabian Grand Prix",
        error_type="RuntimeError",
        message="failed",
    )

    result = SeasonBuildResult(
        season=2023,
        requested_rounds=(
            1,
            2,
        ),
        successful_artifacts=(success,),
        failures=(failure,),
    )

    assert result.attempted_count == 2
    assert result.success_count == 1
    assert result.failure_count == 1
    assert result.succeeded is False


def test_successful_season_result_reports_success() -> None:
    """A season result with no failures should report success."""
    result = SeasonBuildResult(
        season=2023,
        requested_rounds=(1,),
        successful_artifacts=(
            make_artifacts(
                season=2023,
                round_number=1,
                event_slug="bahrain",
            ),
        ),
        failures=(),
    )

    assert result.succeeded is True


def test_empty_season_result_does_not_report_success() -> None:
    """A result with no attempted races should not be considered success."""
    result = SeasonBuildResult(
        season=2023,
        requested_rounds=(),
        successful_artifacts=(),
        failures=(),
    )

    assert result.attempted_count == 0
    assert result.succeeded is False


def test_print_season_build_summary_reports_successes_and_failures(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The terminal summary should expose batch success and failure details."""
    artifact = make_artifacts(
        season=2023,
        round_number=1,
        event_slug="bahrain_grand_prix",
    )

    failure = RaceBuildFailure(
        season=2023,
        round_number=2,
        event_name="Saudi Arabian Grand Prix",
        error_type="RuntimeError",
        message="simulated failure",
    )

    result = SeasonBuildResult(
        season=2023,
        requested_rounds=(
            1,
            2,
        ),
        successful_artifacts=(artifact,),
        failures=(failure,),
    )

    print_season_build_summary(result)

    output = capsys.readouterr().out

    assert "TelemetryX season dataset build" in output
    assert "Season: 2023" in output
    assert "Attempted races: 2" in output
    assert "Successful races: 1" in output
    assert "Failed races: 1" in output

    assert artifact.race_id in output
    assert "Saudi Arabian Grand Prix" in output
    assert "RuntimeError" in output
    assert "simulated failure" in output
