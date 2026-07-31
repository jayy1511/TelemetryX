from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from fastf1.core import Session

from telemetryx.config import Settings, load_settings
from telemetryx.data.clean import CleaningResult, clean_lap_data
from telemetryx.data.inspect_race import (
    parse_event_identifier,
    slugify,
)
from telemetryx.data.load import (
    EventIdentifier,
    load_race_session,
)


class RacePreparationError(RuntimeError):
    """Raised when cleaned race artifacts cannot be prepared or saved."""


@dataclass(frozen=True, slots=True)
class PreparedRaceArtifacts:
    """Paths and metadata produced while preparing one race."""

    cleaned_laps_path: Path
    validation_report_path: Path
    row_count: int
    event_name: str


def prepare_race_data(
    settings: Settings | None = None,
    *,
    season: int | None = None,
    event: EventIdentifier | None = None,
    session_type: str | None = None,
    output_directory: Path | None = None,
    overwrite: bool = False,
) -> PreparedRaceArtifacts:
    """
    Load, clean, validate, and save one historical race.

    Parameters
    ----------
    settings:
        Validated TelemetryX settings. The default settings are loaded when
        this argument is omitted.
    season:
        Optional season override.
    event:
        Optional race name or round-number override.
    session_type:
        Optional FastF1 session-type override.
    output_directory:
        Optional destination directory. The default is
        ``data/interim/cleaned_races``.
    overwrite:
        Whether existing output files may be replaced.

    Returns
    -------
    PreparedRaceArtifacts
        Paths and metadata for the generated artifacts.

    Raises
    ------
    RacePreparationError
        If session laps are unavailable or files cannot be saved.
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

    laps = _extract_lap_dataframe(session)

    cleaning_result = clean_lap_data(laps)

    event_name = _extract_event_name(
        session=session,
        fallback=selected_event,
    )

    session_name = _extract_session_name(
        session=session,
        fallback=selected_session_type,
    )

    destination = (
        output_directory.resolve()
        if output_directory is not None
        else (active_settings.paths.interim_data_dir / "cleaned_races").resolve()
    )

    artifact_stem = f"{selected_season}_{slugify(event_name)}_{slugify(session_name)}"

    cleaned_laps_path = destination / f"{artifact_stem}_laps.parquet"

    validation_report_path = destination / f"{artifact_stem}_validation.json"

    _check_output_paths(
        paths=(
            cleaned_laps_path,
            validation_report_path,
        ),
        overwrite=overwrite,
    )

    report_payload = _build_validation_payload(
        season=selected_season,
        event_name=event_name,
        session_name=session_name,
        cleaning_result=cleaning_result,
    )

    _save_artifacts(
        laps=cleaning_result.laps,
        report_payload=report_payload,
        cleaned_laps_path=cleaned_laps_path,
        validation_report_path=validation_report_path,
    )

    return PreparedRaceArtifacts(
        cleaned_laps_path=cleaned_laps_path,
        validation_report_path=validation_report_path,
        row_count=len(cleaning_result.laps),
        event_name=event_name,
    )


def _extract_lap_dataframe(
    session: Session,
) -> pd.DataFrame:
    """Return a copied lap DataFrame from a loaded FastF1 session."""
    laps: object = session.laps

    if not isinstance(laps, pd.DataFrame):
        raise RacePreparationError(
            "The loaded FastF1 session does not contain a lap DataFrame."
        )

    if laps.empty:
        raise RacePreparationError("The loaded FastF1 lap table is empty.")

    return laps.copy(deep=True)


def _extract_event_name(
    *,
    session: Session,
    fallback: EventIdentifier,
) -> str:
    """Return the loaded event name or a safe fallback."""
    raw_event_name: object = session.event.get("EventName")

    if isinstance(raw_event_name, str):
        normalized = raw_event_name.strip()

        if normalized:
            return normalized

    return str(fallback).strip() or "unknown_event"


def _extract_session_name(
    *,
    session: Session,
    fallback: str,
) -> str:
    """Return the loaded session name or a safe fallback."""
    raw_session_name: object = session.name

    if isinstance(raw_session_name, str):
        normalized = raw_session_name.strip()

        if normalized:
            return normalized

    return fallback.strip().upper() or "session"


def _build_validation_payload(
    *,
    season: int,
    event_name: str,
    session_name: str,
    cleaning_result: CleaningResult,
) -> dict[str, object]:
    """Create a JSON-compatible cleaning and validation report."""
    return {
        "race": {
            "season": season,
            "event_name": event_name,
            "session_name": session_name,
        },
        "cleaning": {
            "row_count": len(cleaning_result.laps),
            "columns": [str(column) for column in cleaning_result.laps.columns],
            "dtypes": {
                str(column): str(dtype)
                for column, dtype in cleaning_result.laps.dtypes.items()
            },
        },
        "input_validation": (cleaning_result.input_validation.to_dict()),
        "output_validation": (cleaning_result.output_validation.to_dict()),
    }


def _check_output_paths(
    *,
    paths: tuple[Path, ...],
    overwrite: bool,
) -> None:
    """Reject accidental replacement of existing artifacts."""
    if overwrite:
        return

    existing_paths = [path for path in paths if path.exists()]

    if not existing_paths:
        return

    formatted_paths = ", ".join(str(path) for path in existing_paths)

    raise RacePreparationError(
        "One or more race artifacts already exist. "
        "Use overwrite=True or --overwrite to replace them: "
        f"{formatted_paths}"
    )


def _save_artifacts(
    *,
    laps: pd.DataFrame,
    report_payload: dict[str, object],
    cleaned_laps_path: Path,
    validation_report_path: Path,
) -> None:
    """Save cleaned laps and validation metadata."""
    temporary_laps_path = cleaned_laps_path.with_suffix(".temporary.parquet")

    temporary_report_path = validation_report_path.with_suffix(".temporary.json")

    try:
        cleaned_laps_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        laps.to_parquet(
            temporary_laps_path,
            index=False,
        )

        temporary_report_path.write_text(
            json.dumps(
                report_payload,
                indent=2,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        temporary_laps_path.replace(cleaned_laps_path)

        temporary_report_path.replace(validation_report_path)
    except Exception as exc:
        _remove_temporary_file(temporary_laps_path)
        _remove_temporary_file(temporary_report_path)

        raise RacePreparationError("Could not save cleaned race artifacts.") from exc


def _remove_temporary_file(
    path: Path,
) -> None:
    """Remove an incomplete temporary file when it exists."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def create_argument_parser() -> argparse.ArgumentParser:
    """Create the race-preparation command-line parser."""
    parser = argparse.ArgumentParser(
        description=("Load, validate, clean, and save one historical Formula 1 race.")
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
        help=("FastF1 session identifier. The MVP normally uses R."),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help=("Optional artifact directory. Defaults to data/interim/cleaned_races."),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing artifacts for the selected race.",
    )

    return parser


def print_preparation_summary(
    artifacts: PreparedRaceArtifacts,
) -> None:
    """Print a concise summary of generated race artifacts."""
    print()
    print("TelemetryX race preparation")
    print("=" * 60)
    print(f"Race: {artifacts.event_name}")
    print(f"Cleaned rows: {artifacts.row_count}")
    print(f"Lap data: {artifacts.cleaned_laps_path}")
    print(f"Validation report: {artifacts.validation_report_path}")


def main() -> None:
    """Run the race-preparation command."""
    parser = create_argument_parser()
    arguments = parser.parse_args()

    artifacts = prepare_race_data(
        season=arguments.season,
        event=arguments.event,
        session_type=arguments.session_type,
        output_directory=arguments.output_directory,
        overwrite=arguments.overwrite,
    )

    print_preparation_summary(artifacts)


if __name__ == "__main__":
    main()
