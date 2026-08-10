from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import fastf1
import numpy as np
import pandas as pd

from telemetryx.config import Settings, load_settings
from telemetryx.data.build_dataset import build_and_save_race_dataset
from telemetryx.data.dataset_artifacts import RaceDatasetArtifacts

DEFAULT_SESSION_TYPE: Final[str] = "R"


class SeasonDatasetBuildError(RuntimeError):
    """Raised when a season-level dataset build cannot be completed."""


@dataclass(frozen=True, slots=True)
class RaceBuildFailure:
    """Description of one race that failed during a season build."""

    season: int
    round_number: int
    event_name: str
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class SeasonBuildResult:
    """Summary of all race builds attempted for one season."""

    season: int
    requested_rounds: tuple[int, ...]
    successful_artifacts: tuple[RaceDatasetArtifacts, ...]
    failures: tuple[RaceBuildFailure, ...]

    @property
    def attempted_count(self) -> int:
        """Return the total number of attempted races."""
        return len(self.successful_artifacts) + len(self.failures)

    @property
    def success_count(self) -> int:
        """Return the number of successfully built races."""
        return len(self.successful_artifacts)

    @property
    def failure_count(self) -> int:
        """Return the number of failed race builds."""
        return len(self.failures)

    @property
    def succeeded(self) -> bool:
        """Return whether every requested race built successfully."""
        return self.attempted_count > 0 and self.failure_count == 0


@dataclass(frozen=True, slots=True)
class ScheduledRace:
    """Minimal race identity extracted from a FastF1 event schedule."""

    season: int
    round_number: int
    event_name: str


def build_season_datasets(
    *,
    season: int,
    settings: Settings | None = None,
    rounds: Sequence[int] | None = None,
    session_type: str = DEFAULT_SESSION_TYPE,
    start_lap: int = 1,
    end_lap: int | None = None,
    output_directory: Path | None = None,
    overwrite: bool = False,
    continue_on_error: bool = True,
) -> SeasonBuildResult:
    """
    Build processed race datasets for selected rounds of one season.

    The FastF1 event schedule is used only to discover race rounds and event
    names. Each race is then passed through the existing single-race
    TelemetryX build pipeline.

    Parameters
    ----------
    season:
        Championship season to build.
    settings:
        Optional validated TelemetryX settings.
    rounds:
        Optional championship rounds to build. When omitted, every race round
        discovered in the FastF1 schedule is selected.
    session_type:
        FastF1 session identifier. The MVP uses ``R``.
    start_lap:
        First completed leader lap included in each race dataset.
    end_lap:
        Optional final completed leader lap included in each race dataset.
    output_directory:
        Optional directory for processed race artifacts.
    overwrite:
        Whether existing race artifacts may be replaced.
    continue_on_error:
        Whether later races should still be attempted after one race fails.

    Returns
    -------
    SeasonBuildResult
        Successful artifacts and structured failures for the season.

    Raises
    ------
    SeasonDatasetBuildError
        If season-level arguments or schedule data are invalid.
    """
    _validate_positive_integer(
        value=season,
        field_name="season",
    )

    normalized_session_type = _normalize_session_type(session_type)

    requested_rounds = _normalize_requested_rounds(rounds)

    active_settings = settings if settings is not None else load_settings()

    schedule = load_season_schedule(season)

    selected_races = select_scheduled_races(
        schedule,
        season=season,
        rounds=requested_rounds,
    )

    successful_artifacts: list[RaceDatasetArtifacts] = []

    failures: list[RaceBuildFailure] = []

    for race in selected_races:
        try:
            artifacts = build_and_save_race_dataset(
                settings=active_settings,
                season=race.season,
                event=race.round_number,
                session_type=normalized_session_type,
                start_lap=start_lap,
                end_lap=end_lap,
                output_directory=output_directory,
                overwrite=overwrite,
            )
        except Exception as exc:
            failure = RaceBuildFailure(
                season=race.season,
                round_number=race.round_number,
                event_name=race.event_name,
                error_type=type(exc).__name__,
                message=str(exc),
            )

            failures.append(failure)

            if not continue_on_error:
                raise SeasonDatasetBuildError(
                    "Season build stopped after a race failure: "
                    f"{race.season} round {race.round_number} "
                    f"({race.event_name})."
                ) from exc

            continue

        successful_artifacts.append(artifacts)

    return SeasonBuildResult(
        season=season,
        requested_rounds=tuple(race.round_number for race in selected_races),
        successful_artifacts=tuple(successful_artifacts),
        failures=tuple(failures),
    )


def load_season_schedule(
    season: int,
) -> pd.DataFrame:
    """
    Load the FastF1 event schedule for one championship season.

    Testing events are excluded because TelemetryX currently models Grand
    Prix race sessions only.

    Parameters
    ----------
    season:
        Positive championship season.

    Returns
    -------
    pd.DataFrame
        Copied FastF1 event schedule.

    Raises
    ------
    SeasonDatasetBuildError
        If the schedule cannot be loaded or is structurally invalid.
    """
    _validate_positive_integer(
        value=season,
        field_name="season",
    )

    try:
        schedule = fastf1.get_event_schedule(
            season,
            include_testing=False,
        )
    except Exception as exc:
        raise SeasonDatasetBuildError(
            f"Could not load the FastF1 event schedule for {season}."
        ) from exc

    if not isinstance(
        schedule,
        pd.DataFrame,
    ):
        raise SeasonDatasetBuildError(
            "FastF1 returned an invalid event schedule object."
        )

    if schedule.empty:
        raise SeasonDatasetBuildError(
            f"FastF1 returned an empty event schedule for {season}."
        )

    required_columns = (
        "RoundNumber",
        "EventName",
    )

    missing_columns = [
        column for column in required_columns if column not in schedule.columns
    ]

    if missing_columns:
        raise SeasonDatasetBuildError(
            "The FastF1 event schedule is missing required columns: "
            f"{', '.join(missing_columns)}."
        )

    return schedule.copy(deep=True)


def select_scheduled_races(
    schedule: pd.DataFrame,
    *,
    season: int,
    rounds: tuple[int, ...] | None = None,
) -> tuple[ScheduledRace, ...]:
    """
    Convert a FastF1 schedule into deterministic race build specifications.

    Parameters
    ----------
    schedule:
        FastF1 event schedule.
    season:
        Championship season represented by the schedule.
    rounds:
        Optional normalized championship rounds to select.

    Returns
    -------
    tuple[ScheduledRace, ...]
        Selected races ordered by championship round.

    Raises
    ------
    SeasonDatasetBuildError
        If schedule rows contain unusable race metadata or requested rounds
        are unavailable.
    """
    if not isinstance(
        schedule,
        pd.DataFrame,
    ):
        raise TypeError("schedule must be provided as a pandas DataFrame.")

    _validate_positive_integer(
        value=season,
        field_name="season",
    )

    required_columns = (
        "RoundNumber",
        "EventName",
    )

    missing_columns = [
        column for column in required_columns if column not in schedule.columns
    ]

    if missing_columns:
        raise SeasonDatasetBuildError(
            "The event schedule is missing required columns: "
            f"{', '.join(missing_columns)}."
        )

    scheduled_races: list[ScheduledRace] = []

    seen_rounds: set[int] = set()

    for index, row in schedule.iterrows():
        round_number = _coerce_positive_integer(row["RoundNumber"])

        if round_number is None:
            continue

        event_name = _normalize_event_name(row["EventName"])

        if event_name is None:
            raise SeasonDatasetBuildError(
                "The event schedule contains a race with an invalid "
                f"EventName at row {index}."
            )

        if round_number in seen_rounds:
            raise SeasonDatasetBuildError(
                "The event schedule contains duplicate championship "
                f"round {round_number}."
            )

        seen_rounds.add(round_number)

        scheduled_races.append(
            ScheduledRace(
                season=season,
                round_number=round_number,
                event_name=event_name,
            )
        )

    scheduled_races.sort(key=lambda race: race.round_number)

    if not scheduled_races:
        raise SeasonDatasetBuildError(
            f"No race rounds were found in the {season} event schedule."
        )

    if rounds is None:
        return tuple(scheduled_races)

    requested = set(rounds)

    available = {race.round_number for race in scheduled_races}

    unavailable_rounds = sorted(requested.difference(available))

    if unavailable_rounds:
        raise SeasonDatasetBuildError(
            "Requested rounds are unavailable in the event schedule: "
            f"{', '.join(str(value) for value in unavailable_rounds)}."
        )

    return tuple(race for race in scheduled_races if race.round_number in requested)


def parse_rounds_argument(
    raw_value: str,
) -> tuple[int, ...]:
    """
    Parse a comma-separated CLI championship-round list.

    Examples
    --------
    ``"1,2,3"`` becomes ``(1, 2, 3)``.
    """
    parts = [part.strip() for part in raw_value.split(",")]

    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "Rounds must be a comma-separated list of positive integers."
        )

    parsed_rounds: list[int] = []

    for part in parts:
        try:
            round_number = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Rounds must be a comma-separated list of positive integers."
            ) from exc

        if round_number <= 0:
            raise argparse.ArgumentTypeError(
                "Rounds must contain only positive integers."
            )

        parsed_rounds.append(round_number)

    if len(set(parsed_rounds)) != len(parsed_rounds):
        raise argparse.ArgumentTypeError("Rounds must not contain duplicate values.")

    return tuple(sorted(parsed_rounds))


def create_argument_parser() -> argparse.ArgumentParser:
    """Create the season dataset-build command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build processed TelemetryX race datasets for one Formula 1 season."
        )
    )

    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Championship season to build.",
    )

    parser.add_argument(
        "--rounds",
        type=parse_rounds_argument,
        default=None,
        help=(
            "Optional comma-separated rounds, for example 1,2,3. "
            "Defaults to every race in the season schedule."
        ),
    )

    parser.add_argument(
        "--session-type",
        type=str,
        default=DEFAULT_SESSION_TYPE,
        help="FastF1 session identifier. Defaults to R.",
    )

    parser.add_argument(
        "--start-lap",
        type=int,
        default=1,
        help="First completed leader lap included in every race.",
    )

    parser.add_argument(
        "--end-lap",
        type=int,
        default=None,
        help="Optional final completed leader lap.",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help=("Optional processed race-dataset destination."),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing race artifacts.",
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help=("Stop the season build immediately when one race fails."),
    )

    return parser


def print_season_build_summary(
    result: SeasonBuildResult,
) -> None:
    """Print a concise season-level build report."""
    print()
    print("TelemetryX season dataset build")
    print("=" * 60)
    print(f"Season: {result.season}")
    print(f"Attempted races: {result.attempted_count}")
    print(f"Successful races: {result.success_count}")
    print(f"Failed races: {result.failure_count}")

    if result.successful_artifacts:
        print()
        print("Successful datasets:")

        for artifacts in result.successful_artifacts:
            print(f"  - {artifacts.race_id}")

    if result.failures:
        print()
        print("Failures:")

        for failure in result.failures:
            print(
                "  - "
                f"Round {failure.round_number} "
                f"{failure.event_name}: "
                f"{failure.error_type}: "
                f"{failure.message}"
            )


def main() -> None:
    """Run the historical season dataset build command."""
    parser = create_argument_parser()
    arguments = parser.parse_args()

    result = build_season_datasets(
        season=arguments.season,
        rounds=arguments.rounds,
        session_type=arguments.session_type,
        start_lap=arguments.start_lap,
        end_lap=arguments.end_lap,
        output_directory=arguments.output_directory,
        overwrite=arguments.overwrite,
        continue_on_error=(not arguments.stop_on_error),
    )

    print_season_build_summary(result)


def _normalize_requested_rounds(
    rounds: Sequence[int] | None,
) -> tuple[int, ...] | None:
    """Validate and normalize optional requested championship rounds."""
    if rounds is None:
        return None

    normalized: list[int] = []

    for round_number in rounds:
        _validate_positive_integer(
            value=round_number,
            field_name="round number",
        )

        normalized.append(round_number)

    if not normalized:
        raise SeasonDatasetBuildError(
            "rounds cannot be empty when explicitly provided."
        )

    if len(set(normalized)) != len(normalized):
        raise SeasonDatasetBuildError("rounds cannot contain duplicate values.")

    return tuple(sorted(normalized))


def _normalize_session_type(
    session_type: str,
) -> str:
    """Return a non-empty normalized FastF1 session identifier."""
    if not isinstance(
        session_type,
        str,
    ):
        raise SeasonDatasetBuildError("session_type must be a string.")

    normalized = session_type.strip()

    if not normalized:
        raise SeasonDatasetBuildError("session_type cannot be blank.")

    return normalized


def _normalize_event_name(
    value: object,
) -> str | None:
    """Return a normalized event name when one is available."""
    if not isinstance(
        value,
        str,
    ):
        return None

    normalized = value.strip()

    return normalized or None


def _coerce_positive_integer(
    value: object,
) -> int | None:
    """Convert supported schedule scalar values to positive integers."""
    if value is None:
        return None

    if value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(
        value,
        (int, np.integer),
    ):
        try:
            integer_value = int(str(value))
        except ValueError:
            return None

        if integer_value <= 0:
            return None

        return integer_value

    if isinstance(
        value,
        (float, np.floating),
    ):
        try:
            numeric_value = float(str(value))
        except ValueError:
            return None

        if not math.isfinite(numeric_value):
            return None

        if not numeric_value.is_integer():
            return None

        integer_value = int(numeric_value)

        if integer_value <= 0:
            return None

        return integer_value

    if isinstance(value, str):
        normalized = value.strip()

        if not normalized:
            return None

        try:
            numeric_value = float(normalized)
        except ValueError:
            return None

        if not math.isfinite(numeric_value):
            return None

        if not numeric_value.is_integer():
            return None

        integer_value = int(numeric_value)

        if integer_value <= 0:
            return None

        return integer_value

    return None


def _validate_positive_integer(
    *,
    value: int,
    field_name: str,
) -> None:
    """Validate one positive integer argument."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SeasonDatasetBuildError(f"{field_name} must be a positive integer.")


if __name__ == "__main__":
    main()
