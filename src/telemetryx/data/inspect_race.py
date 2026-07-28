from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastf1.core import Session

from telemetryx.config import Settings, load_settings
from telemetryx.data.load import EventIdentifier, load_race_session

REQUIRED_LAP_COLUMNS = frozenset(
    {
        "Driver",
        "LapNumber",
        "LapTime",
        "Position",
        "Stint",
        "Compound",
        "TyreLife",
        "TrackStatus",
    }
)


class RaceInventoryError(RuntimeError):
    """Raised when a loaded race cannot produce a valid inventory."""


def parse_event_identifier(value: str) -> EventIdentifier:
    """
    Convert a command-line event value into a race name or round number.

    Numeric values such as ``"1"`` become the integer ``1``. Other values
    remain strings and are interpreted by FastF1 as event names.

    Parameters
    ----------
    value:
        Raw command-line value supplied after ``--event``.

    Returns
    -------
    str | int
        A normalized event name or positive round number.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value is empty or is a non-positive integer.
    """
    normalized = value.strip()

    if not normalized:
        raise argparse.ArgumentTypeError(
            "The event must be a race name or positive round number."
        )

    try:
        round_number = int(normalized)
    except ValueError:
        return normalized

    if round_number < 1:
        raise argparse.ArgumentTypeError(
            "An event round number must be greater than zero."
        )

    return round_number


def dataframe_inventory(
    frame: pd.DataFrame | None,
) -> dict[str, Any]:
    """
    Create a JSON-compatible structural summary of a DataFrame.

    Parameters
    ----------
    frame:
        DataFrame to inspect. ``None`` represents unavailable session data.

    Returns
    -------
    dict[str, Any]
        Row count, column names, data types, and missing-value counts.
    """
    if frame is None:
        return {
            "available": False,
            "rows": 0,
            "columns": [],
            "dtypes": {},
            "missing_values": {},
        }

    missing_values = frame.isna().sum()

    return {
        "available": True,
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values": {
            str(column): int(count) for column, count in missing_values.items()
        },
    }


def validate_lap_table(laps: pd.DataFrame) -> None:
    """
    Validate the minimum lap-table contract required by TelemetryX.

    Parameters
    ----------
    laps:
        Loaded FastF1 lap table.

    Raises
    ------
    RaceInventoryError
        If the table is empty or required columns are absent.
    """
    if laps.empty:
        raise RaceInventoryError("The loaded FastF1 lap table is empty.")

    available_columns = {str(column) for column in laps.columns}

    missing_columns = sorted(REQUIRED_LAP_COLUMNS.difference(available_columns))

    if missing_columns:
        formatted_columns = ", ".join(missing_columns)

        raise RaceInventoryError(
            f"The FastF1 lap table is missing required columns: {formatted_columns}."
        )


def build_race_inventory(
    session: Session,
) -> dict[str, Any]:
    """
    Build a structural inventory for a loaded FastF1 session.

    Parameters
    ----------
    session:
        Fully loaded FastF1 session.

    Returns
    -------
    dict[str, Any]
        JSON-compatible race metadata and table summaries.

    Raises
    ------
    RaceInventoryError
        If required lap data is unavailable.
    """
    laps = _require_dataframe(
        session.laps,
        table_name="laps",
    )

    results = _optional_dataframe(
        session.results,
        table_name="results",
    )

    weather = _optional_dataframe(
        session.weather_data,
        table_name="weather",
    )

    race_control_messages = _optional_dataframe(
        session.race_control_messages,
        table_name="race_control_messages",
    )

    validate_lap_table(laps)

    event = session.event

    event_name = _optional_string(event.get("EventName")) or "unknown_event"

    official_event_name = _optional_string(event.get("OfficialEventName"))

    event_date = _optional_timestamp_string(event.get("EventDate"))

    round_number = _optional_integer(event.get("RoundNumber"))

    season = _extract_season(
        event_date_value=event.get("EventDate"),
    )

    return {
        "race": {
            "season": season,
            "round": round_number,
            "event_name": event_name,
            "official_event_name": official_event_name,
            "event_date": event_date,
            "session_name": str(session.name),
            "scheduled_total_laps": _optional_integer(session.total_laps),
        },
        "tables": {
            "laps": dataframe_inventory(laps),
            "results": dataframe_inventory(results),
            "weather": dataframe_inventory(weather),
            "race_control_messages": dataframe_inventory(race_control_messages),
        },
    }


def save_race_inventory(
    inventory: dict[str, Any],
    output_path: Path,
) -> Path:
    """
    Save a race inventory as formatted JSON.

    Parameters
    ----------
    inventory:
        JSON-compatible race inventory.
    output_path:
        Destination JSON file.

    Returns
    -------
    Path
        Absolute path to the saved inventory.

    Raises
    ------
    RaceInventoryError
        If the destination directory or file cannot be created.
    """
    resolved_output = output_path.resolve()

    try:
        resolved_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        resolved_output.write_text(
            json.dumps(
                inventory,
                indent=2,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        raise RaceInventoryError(
            f"Could not save race inventory: {resolved_output}"
        ) from exc

    return resolved_output


def default_inventory_path(
    settings: Settings,
    inventory: dict[str, Any],
) -> Path:
    """
    Construct the default inventory filename for a loaded race.

    Parameters
    ----------
    settings:
        Validated TelemetryX project settings.
    inventory:
        Inventory containing race metadata.

    Returns
    -------
    Path
        Path under ``data/interim/inventories``.
    """
    race_metadata = inventory["race"]

    if not isinstance(race_metadata, dict):
        raise RaceInventoryError("Race inventory metadata has an invalid structure.")

    season = race_metadata.get("season")
    event_name = race_metadata.get(
        "event_name",
        "unknown_event",
    )
    session_name = race_metadata.get(
        "session_name",
        "session",
    )

    filename = (
        f"{season}_"
        f"{slugify(str(event_name))}_"
        f"{slugify(str(session_name))}_inventory.json"
    )

    return settings.paths.interim_data_dir / "inventories" / filename


def print_inventory_summary(
    inventory: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Print a concise human-readable inventory summary.

    Parameters
    ----------
    inventory:
        Generated race inventory.
    output_path:
        Path where the JSON inventory was saved.
    """
    race_metadata = inventory["race"]
    table_metadata = inventory["tables"]

    if not isinstance(race_metadata, dict):
        raise RaceInventoryError("Race inventory metadata has an invalid structure.")

    if not isinstance(table_metadata, dict):
        raise RaceInventoryError("Table inventory metadata has an invalid structure.")

    print()
    print("TelemetryX race inventory")
    print("=" * 60)
    print(f"Race: {race_metadata.get('season')} {race_metadata.get('event_name')}")
    print(f"Session: {race_metadata.get('session_name')}")
    print(f"Scheduled laps: {race_metadata.get('scheduled_total_laps')}")
    print()

    for table_name, table_information in table_metadata.items():
        if not isinstance(table_information, dict):
            continue

        available = table_information.get(
            "available",
            False,
        )

        rows = table_information.get(
            "rows",
            0,
        )

        columns = table_information.get(
            "columns",
            [],
        )

        column_count = len(columns) if isinstance(columns, list) else 0

        print(
            f"{table_name}: available={available}, rows={rows}, columns={column_count}"
        )

    print()
    print(f"Inventory saved to: {output_path}")


def slugify(value: str) -> str:
    """
    Convert text into a simple filename-safe identifier.

    Parameters
    ----------
    value:
        Text to normalize.

    Returns
    -------
    str
        Lowercase identifier containing letters, numbers, and underscores.
    """
    normalized_characters = [
        character.lower() if character.isalnum() else "_" for character in value
    ]

    normalized = "".join(normalized_characters)

    parts = [part for part in normalized.split("_") if part]

    return "_".join(parts) or "unknown"


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser for race inspection commands.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Load one historical Formula 1 session and save a "
            "structural data inventory."
        )
    )

    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help=(
            "Championship season. Defaults to sample_race.season "
            "from config/settings.yaml."
        ),
    )

    parser.add_argument(
        "--event",
        type=parse_event_identifier,
        default=None,
        help=(
            "Event name or positive championship round number. "
            "Defaults to sample_race.event from settings."
        ),
    )

    parser.add_argument(
        "--session-type",
        type=str,
        default=None,
        help=(
            "FastF1 session identifier, such as R or Q. "
            "Defaults to the configured sample session."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output JSON path. When omitted, the inventory "
            "is saved under data/interim/inventories."
        ),
    )

    return parser


def main() -> None:
    """Run the race-inventory command."""
    parser = create_argument_parser()
    arguments = parser.parse_args()

    settings = load_settings()

    session = load_race_session(
        settings=settings,
        season=arguments.season,
        event=arguments.event,
        session_type=arguments.session_type,
    )

    inventory = build_race_inventory(session)

    output_path = (
        arguments.output
        if arguments.output is not None
        else default_inventory_path(
            settings=settings,
            inventory=inventory,
        )
    )

    saved_path = save_race_inventory(
        inventory=inventory,
        output_path=output_path,
    )

    print_inventory_summary(
        inventory=inventory,
        output_path=saved_path,
    )


def _require_dataframe(
    value: object,
    table_name: str,
) -> pd.DataFrame:
    """Return a required DataFrame or raise an inventory error."""
    if not isinstance(value, pd.DataFrame):
        raise RaceInventoryError(f"The loaded {table_name} object is not a DataFrame.")

    return value


def _optional_dataframe(
    value: object,
    table_name: str,
) -> pd.DataFrame | None:
    """Return an optional DataFrame while rejecting unexpected types."""
    if value is None:
        return None

    if not isinstance(value, pd.DataFrame):
        raise RaceInventoryError(f"The loaded {table_name} object is not a DataFrame.")

    return value


def _optional_string(
    value: object,
) -> str | None:
    """Convert a scalar value to a non-empty string when possible."""
    if value is None or _is_missing_scalar(value):
        return None

    normalized = str(value).strip()

    return normalized or None


def _optional_integer(
    value: object,
) -> int | None:
    """Convert a supported scalar numeric value to an integer."""
    if _is_missing_scalar(value) or isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, np.integer):
        try:
            return int(str(value))
        except ValueError:
            return None

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

        return int(value)

    if isinstance(value, np.floating):
        try:
            numeric_value = float(str(value))
        except ValueError:
            return None

        if not math.isfinite(numeric_value):
            return None

        return int(numeric_value)

    if isinstance(value, str):
        normalized = value.strip()

        if not normalized:
            return None

        try:
            return int(normalized)
        except ValueError:
            return None

    return None


def _optional_timestamp_string(
    value: object,
) -> str | None:
    """Convert a supported date-like scalar into ISO-8601 text."""
    timestamp = _coerce_timestamp(value)

    if timestamp is None:
        return None

    return timestamp.isoformat()


def _extract_season(
    event_date_value: object,
) -> int | None:
    """Extract the season year from a supported event-date value."""
    timestamp = _coerce_timestamp(event_date_value)

    if timestamp is None:
        return None

    return timestamp.year


def _coerce_timestamp(
    value: object,
) -> pd.Timestamp | None:
    """
    Convert supported scalar date values into a pandas Timestamp.

    Unsupported objects and missing values return ``None`` instead of being
    silently coerced.
    """
    if _is_missing_scalar(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value

    if isinstance(value, np.datetime64):
        try:
            return pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            return None

    if isinstance(value, datetime):
        try:
            return pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            return None

    if isinstance(value, date):
        try:
            return pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            return None

    if isinstance(value, str):
        normalized = value.strip()

        if not normalized:
            return None

        try:
            return pd.Timestamp(normalized)
        except (TypeError, ValueError, OverflowError):
            return None

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        try:
            return pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            return None

    if isinstance(value, np.integer):
        try:
            integer_value = int(str(value))
            return pd.Timestamp(integer_value)
        except (TypeError, ValueError, OverflowError):
            return None

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

        try:
            return pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            return None

    if isinstance(value, np.floating):
        try:
            numeric_value = float(str(value))
        except ValueError:
            return None

        if not math.isfinite(numeric_value):
            return None

        try:
            return pd.Timestamp(numeric_value)
        except (TypeError, ValueError, OverflowError):
            return None

    return None


def _is_missing_scalar(
    value: object,
) -> bool:
    """Return whether a supported scalar represents missing data."""
    if value is None:
        return True

    if value is pd.NA or value is pd.NaT:
        return True

    if isinstance(value, float):
        return math.isnan(value)

    if isinstance(value, np.floating):
        try:
            numeric_value = float(str(value))
        except ValueError:
            return False

        return math.isnan(numeric_value)

    return False


if __name__ == "__main__":
    main()
