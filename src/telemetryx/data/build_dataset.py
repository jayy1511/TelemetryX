from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from fastf1.core import Session

from telemetryx.config import Settings, load_settings
from telemetryx.data.clean import clean_lap_data
from telemetryx.data.dataset import build_race_dataset
from telemetryx.data.dataset_artifacts import (
    RaceDatasetArtifacts,
    save_race_dataset_artifacts,
)
from telemetryx.data.inspect_race import parse_event_identifier
from telemetryx.data.load import (
    EventIdentifier,
    load_race_session,
)


class RaceDatasetBuildError(RuntimeError):
    """Raised when a historical race dataset cannot be built."""


def build_and_save_race_dataset(
    settings: Settings | None = None,
    *,
    season: int | None = None,
    event: EventIdentifier | None = None,
    session_type: str | None = None,
    start_lap: int = 1,
    end_lap: int | None = None,
    output_directory: Path | None = None,
    overwrite: bool = False,
) -> RaceDatasetArtifacts:
    """
    Build and persist one historical race dataset.

    Parameters
    ----------
    settings:
        Validated TelemetryX settings. Defaults are loaded when omitted.
    season:
        Optional season override.
    event:
        Optional event-name or round-number override.
    session_type:
        Optional FastF1 session identifier.
    start_lap:
        First completed leader lap included in the replay.
    end_lap:
        Optional final completed leader lap included in the replay.
    output_directory:
        Optional processed-artifact destination.
    overwrite:
        Whether existing processed artifacts may be replaced.

    Returns
    -------
    RaceDatasetArtifacts
        Paths and summary information for the saved processed dataset.

    Raises
    ------
    RaceDatasetBuildError
        If required session data or race metadata is unavailable.
    """
    active_settings = settings if settings is not None else load_settings()

    selected_season = active_settings.sample_race.season if season is None else season

    selected_event: EventIdentifier = (
        active_settings.sample_race.event if event is None else event
    )

    selected_session_type = (
        active_settings.sample_race.session_type
        if session_type is None
        else session_type
    )

    session = load_race_session(
        settings=active_settings,
        season=selected_season,
        event=selected_event,
        session_type=selected_session_type,
    )

    laps = _extract_dataframe(
        session=session,
        attribute_name="laps",
    )

    results = _extract_dataframe(
        session=session,
        attribute_name="results",
    )

    cleaning_result = clean_lap_data(laps)

    round_number = _extract_round_number(session)

    event_name = _extract_event_name(
        session=session,
        fallback=selected_event,
    )

    session_name = _extract_session_name(
        session=session,
        fallback=selected_session_type,
    )

    dataset = build_race_dataset(
        cleaning_result.laps,
        results,
        season=selected_season,
        round_number=round_number,
        event_name=event_name,
        session_name=session_name,
        start_lap=start_lap,
        end_lap=end_lap,
    )

    return save_race_dataset_artifacts(
        dataset,
        settings=active_settings,
        output_directory=output_directory,
        overwrite=overwrite,
    )


def _extract_dataframe(
    *,
    session: Session,
    attribute_name: str,
) -> pd.DataFrame:
    """Return a copied DataFrame from a loaded FastF1 session."""
    value = getattr(
        session,
        attribute_name,
        None,
    )

    if not isinstance(value, pd.DataFrame):
        raise RaceDatasetBuildError(
            f"The loaded FastF1 session does not contain a {attribute_name} DataFrame."
        )

    if value.empty:
        raise RaceDatasetBuildError(
            f"The loaded FastF1 {attribute_name} table is empty."
        )

    return value.copy(deep=True)


def _extract_round_number(
    session: Session,
) -> int:
    """Return the positive championship round number."""
    raw_value: object = session.event.get("RoundNumber")

    round_number = _coerce_positive_integer(raw_value)

    if round_number is None:
        raise RaceDatasetBuildError(
            "The loaded FastF1 event does not contain a valid positive RoundNumber."
        )

    return round_number


def _extract_event_name(
    *,
    session: Session,
    fallback: EventIdentifier,
) -> str:
    """Return the loaded event name or a deterministic fallback."""
    raw_event_name: object = session.event.get("EventName")

    if isinstance(raw_event_name, str):
        normalized = raw_event_name.strip()

        if normalized:
            return normalized

    fallback_name = str(fallback).strip()

    if fallback_name:
        return fallback_name

    raise RaceDatasetBuildError("The race event name is unavailable.")


def _extract_session_name(
    *,
    session: Session,
    fallback: str,
) -> str:
    """Return the loaded session name or requested session identifier."""
    raw_session_name: object = session.name

    if isinstance(raw_session_name, str):
        normalized = raw_session_name.strip()

        if normalized:
            return normalized

    fallback_name = fallback.strip()

    if fallback_name:
        return fallback_name

    raise RaceDatasetBuildError("The race session name is unavailable.")


def _coerce_positive_integer(
    value: object,
) -> int | None:
    """Convert supported scalar values into a positive integer."""
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

        return integer_value if integer_value > 0 else None

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

        return integer_value if integer_value > 0 else None

    if isinstance(value, str):
        normalized = value.strip()

        if not normalized:
            return None

        try:
            integer_value = int(normalized)
        except ValueError:
            return None

        return integer_value if integer_value > 0 else None

    return None


def create_argument_parser() -> argparse.ArgumentParser:
    """Create the historical race-dataset command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a processed TelemetryX dataset for one historical Formula 1 race."
        )
    )

    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help=("Season override. Defaults to the configured sample race."),
    )

    parser.add_argument(
        "--event",
        type=parse_event_identifier,
        default=None,
        help=("Event name or positive championship round number."),
    )

    parser.add_argument(
        "--session-type",
        type=str,
        default=None,
        help=("FastF1 session identifier. Normally R for the race."),
    )

    parser.add_argument(
        "--start-lap",
        type=int,
        default=1,
        help=("First completed leader lap included in the dataset."),
    )

    parser.add_argument(
        "--end-lap",
        type=int,
        default=None,
        help=("Optional final completed leader lap included in the dataset."),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help=(
            "Optional processed dataset directory. Defaults to "
            "data/processed/race_datasets."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=("Replace existing processed artifacts for this race."),
    )

    return parser


def print_build_summary(
    artifacts: RaceDatasetArtifacts,
) -> None:
    """Print a concise summary of a completed race-dataset build."""
    print()
    print("TelemetryX race dataset")
    print("=" * 60)
    print(f"Race ID: {artifacts.race_id}")
    print(f"Rows: {artifacts.row_count}")
    print(f"Snapshots: {artifacts.snapshot_count}")
    print(f"Drivers: {artifacts.driver_count}")
    print(f"Dataset: {artifacts.dataset_path}")
    print(f"Manifest: {artifacts.manifest_path}")


def main() -> None:
    """Run the historical race-dataset build command."""
    parser = create_argument_parser()
    arguments = parser.parse_args()

    artifacts = build_and_save_race_dataset(
        season=arguments.season,
        event=arguments.event,
        session_type=arguments.session_type,
        start_lap=arguments.start_lap,
        end_lap=arguments.end_lap,
        output_directory=arguments.output_directory,
        overwrite=arguments.overwrite,
    )

    print_build_summary(artifacts)


if __name__ == "__main__":
    main()
