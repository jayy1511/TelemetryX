from __future__ import annotations

from typing import Any, Final

import pandas as pd

REQUIRED_REPLAY_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "Driver",
    "LapNumber",
    "Position",
    "Stint",
    "Compound",
    "TyreLife",
    "TrackStatus",
    "LapTimeSeconds",
)

REPLAY_COLUMNS: Final[tuple[str, ...]] = (
    "SnapshotLap",
    "Driver",
    "Position",
    "CompletedLaps",
    "LapsBehindLeader",
    "Stint",
    "Compound",
    "TyreLife",
    "TrackStatus",
    "LastLapTimeSeconds",
    "CumulativeLapTimeSeconds",
    "IsLeader",
    "DataAvailableThroughLap",
)


class RaceReplayError(ValueError):
    """Raised when temporally valid race snapshots cannot be constructed."""


def build_race_replay(
    laps: pd.DataFrame,
    *,
    start_lap: int = 1,
    end_lap: int | None = None,
) -> pd.DataFrame:
    """
    Build one race-state row per driver and completed leader lap.

    A snapshot for lap ``N`` may only use lap records whose lap number is less
    than or equal to ``N``. Rows from later laps are never included.

    Parameters
    ----------
    laps:
        Cleaned FastF1-like lap data containing ``LapTimeSeconds``.
    start_lap:
        First leader lap to include.
    end_lap:
        Optional final leader lap to include. By default, all available leader
        laps are used.

    Returns
    -------
    pd.DataFrame
        Long-format replay table containing one row per snapshot and driver.

    Raises
    ------
    TypeError
        If ``laps`` is not a DataFrame.
    RaceReplayError
        If the input structure or values cannot produce safe snapshots.
    """
    if not isinstance(laps, pd.DataFrame):
        raise TypeError("laps must be provided as a pandas DataFrame.")

    _validate_lap_range(
        start_lap=start_lap,
        end_lap=end_lap,
    )

    _require_columns(
        frame=laps,
        required_columns=REQUIRED_REPLAY_INPUT_COLUMNS,
        object_name="lap table",
    )

    if laps.empty:
        raise RaceReplayError("Cannot build a race replay from an empty lap table.")

    working = _prepare_working_laps(laps)

    snapshot_laps = _select_snapshot_laps(
        working,
        start_lap=start_lap,
        end_lap=end_lap,
    )

    snapshots = [
        _build_snapshot(
            working,
            snapshot_lap=snapshot_lap,
        )
        for snapshot_lap in snapshot_laps
    ]

    replay = pd.concat(
        snapshots,
        ignore_index=True,
    )

    replay = replay.sort_values(
        by=[
            "SnapshotLap",
            "Position",
            "Driver",
        ],
        ascending=True,
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    _validate_replay_output(replay)

    return replay


def select_replay_snapshot(
    replay: pd.DataFrame,
    *,
    snapshot_lap: int,
) -> pd.DataFrame:
    """
    Return a copied race-state table for one completed leader lap.

    Parameters
    ----------
    replay:
        Replay table returned by :func:`build_race_replay`.
    snapshot_lap:
        Positive completed leader lap to select.

    Returns
    -------
    pd.DataFrame
        One row per driver represented at the requested snapshot.

    Raises
    ------
    TypeError
        If ``replay`` is not a DataFrame.
    RaceReplayError
        If the requested snapshot is invalid or unavailable.
    """
    if not isinstance(replay, pd.DataFrame):
        raise TypeError("replay must be provided as a pandas DataFrame.")

    if isinstance(snapshot_lap, bool) or snapshot_lap <= 0:
        raise RaceReplayError("snapshot_lap must be a positive integer.")

    _require_columns(
        frame=replay,
        required_columns=REPLAY_COLUMNS,
        object_name="replay table",
    )

    snapshot = replay.loc[replay["SnapshotLap"].eq(snapshot_lap)].copy()

    if snapshot.empty:
        raise RaceReplayError(f"Replay snapshot lap {snapshot_lap} is unavailable.")

    return snapshot.reset_index(drop=True)


def _prepare_working_laps(
    laps: pd.DataFrame,
) -> pd.DataFrame:
    """Return a normalized private copy used for replay construction."""
    working = laps.copy(deep=True)

    working["Driver"] = _normalize_driver_column(working["Driver"])

    working["LapNumber"] = _coerce_whole_number_column(
        frame=working,
        column="LapNumber",
        allow_missing=False,
    )

    working["Position"] = _coerce_whole_number_column(
        frame=working,
        column="Position",
        allow_missing=True,
    )

    working["Stint"] = _coerce_whole_number_column(
        frame=working,
        column="Stint",
        allow_missing=True,
    )

    working["TyreLife"] = _coerce_numeric_column(
        frame=working,
        column="TyreLife",
        allow_missing=True,
        non_negative=True,
    ).astype("Float64")

    working["LapTimeSeconds"] = _coerce_numeric_column(
        frame=working,
        column="LapTimeSeconds",
        allow_missing=True,
        non_negative=False,
    ).astype("Float64")

    non_positive_lap_times = working["LapTimeSeconds"].notna() & working[
        "LapTimeSeconds"
    ].le(0)

    if bool(non_positive_lap_times.any()):
        raise RaceReplayError(
            "LapTimeSeconds must be greater than zero at rows: "
            f"{_format_affected_indices(working, non_positive_lap_times)}."
        )

    working["Compound"] = _normalize_optional_text_column(
        working["Compound"],
        uppercase=True,
    )

    working["TrackStatus"] = _normalize_optional_text_column(
        working["TrackStatus"],
        uppercase=False,
    )

    duplicate_driver_laps = working.duplicated(
        subset=[
            "Driver",
            "LapNumber",
        ],
        keep=False,
    )

    if bool(duplicate_driver_laps.any()):
        raise RaceReplayError(
            "The lap table contains duplicate Driver and LapNumber rows at "
            f"indices: "
            f"{_format_affected_indices(working, duplicate_driver_laps)}."
        )

    working["_SourceOrder"] = range(len(working))

    working = working.sort_values(
        by=[
            "Driver",
            "LapNumber",
            "_SourceOrder",
        ],
        kind="stable",
    ).reset_index(drop=True)

    missing_lap_time_count = (
        working["LapTimeSeconds"]
        .isna()
        .astype("int64")
        .groupby(
            working["Driver"],
            sort=False,
        )
        .cumsum()
    )

    cumulative_lap_time = (
        working["LapTimeSeconds"]
        .fillna(0.0)
        .groupby(
            working["Driver"],
            sort=False,
        )
        .cumsum()
    )

    working["_CumulativeLapTimeSeconds"] = cumulative_lap_time.mask(
        missing_lap_time_count.gt(0)
    ).astype("Float64")

    return working


def _select_snapshot_laps(
    working: pd.DataFrame,
    *,
    start_lap: int,
    end_lap: int | None,
) -> tuple[int, ...]:
    """Return sorted leader laps within the requested replay interval."""
    leader_lap_values = working.loc[
        working["Position"].eq(1),
        "LapNumber",
    ].dropna()

    available_laps = sorted({int(value) for value in leader_lap_values.tolist()})

    selected_laps = tuple(
        lap
        for lap in available_laps
        if lap >= start_lap and (end_lap is None or lap <= end_lap)
    )

    if not selected_laps:
        requested_range = (
            f"{start_lap} onward"
            if end_lap is None
            else f"{start_lap} through {end_lap}"
        )

        raise RaceReplayError(
            "No completed leader laps are available for the requested "
            f"range: {requested_range}."
        )

    return selected_laps


def _build_snapshot(
    working: pd.DataFrame,
    *,
    snapshot_lap: int,
) -> pd.DataFrame:
    """Build one temporally bounded driver-state snapshot."""
    observed = working.loc[working["LapNumber"].le(snapshot_lap)]

    latest_driver_rows = (
        observed.groupby(
            "Driver",
            sort=False,
            group_keys=False,
        )
        .tail(1)
        .copy()
    )

    latest_driver_rows["SnapshotLap"] = snapshot_lap

    latest_driver_rows["CompletedLaps"] = latest_driver_rows["LapNumber"].astype(
        "Int64"
    )

    latest_driver_rows["LapsBehindLeader"] = (
        snapshot_lap - latest_driver_rows["CompletedLaps"]
    ).astype("Int64")

    latest_driver_rows["LastLapTimeSeconds"] = latest_driver_rows[
        "LapTimeSeconds"
    ].astype("Float64")

    latest_driver_rows["CumulativeLapTimeSeconds"] = latest_driver_rows[
        "_CumulativeLapTimeSeconds"
    ].astype("Float64")

    latest_driver_rows["IsLeader"] = (
        latest_driver_rows["Position"].eq(1)
        & latest_driver_rows["LapsBehindLeader"].eq(0)
    ).astype("boolean")

    latest_driver_rows["DataAvailableThroughLap"] = snapshot_lap

    return latest_driver_rows.loc[
        :,
        list(REPLAY_COLUMNS),
    ].copy()


def _validate_replay_output(
    replay: pd.DataFrame,
) -> None:
    """Validate temporal and structural replay invariants."""
    if replay.empty:
        raise RaceReplayError("Race replay construction produced no rows.")

    duplicate_snapshot_drivers = replay.duplicated(
        subset=[
            "SnapshotLap",
            "Driver",
        ],
        keep=False,
    )

    if bool(duplicate_snapshot_drivers.any()):
        raise RaceReplayError("Replay output contains duplicate snapshot-driver rows.")

    future_completed_laps = (
        replay["CompletedLaps"].gt(replay["SnapshotLap"]).fillna(False)
    )

    if bool(future_completed_laps.any()):
        raise RaceReplayError(
            "Replay output contains driver laps beyond the snapshot cutoff."
        )

    negative_lap_deficits = replay["LapsBehindLeader"].lt(0).fillna(False)

    if bool(negative_lap_deficits.any()):
        raise RaceReplayError("Replay output contains negative lap deficits.")

    invalid_availability_cutoff = replay["DataAvailableThroughLap"].ne(
        replay["SnapshotLap"]
    )

    if bool(invalid_availability_cutoff.any()):
        raise RaceReplayError(
            "Replay availability metadata does not match snapshot laps."
        )

    snapshot_laps = {int(value) for value in replay["SnapshotLap"].tolist()}

    leader_counts = (
        replay.loc[replay["IsLeader"].fillna(False)]
        .groupby(
            "SnapshotLap",
            sort=False,
        )
        .size()
    )

    invalid_leader_laps = [
        snapshot_lap
        for snapshot_lap in sorted(snapshot_laps)
        if int(leader_counts.get(snapshot_lap, 0)) != 1
    ]

    if invalid_leader_laps:
        formatted_laps = ", ".join(str(lap) for lap in invalid_leader_laps)

        raise RaceReplayError(
            "Each replay snapshot must contain exactly one leader. "
            f"Invalid snapshot laps: {formatted_laps}."
        )


def _normalize_driver_column(
    series: pd.Series[Any],
) -> pd.Series[Any]:
    """Normalize driver identifiers and reject missing values."""
    normalized = series.astype("string").str.strip().str.upper()

    missing_drivers = normalized.isna() | normalized.eq("")

    if bool(missing_drivers.any()):
        affected_indices = ", ".join(
            str(index) for index in series.index[missing_drivers].tolist()[:5]
        )

        raise RaceReplayError(
            "Driver contains missing or blank values at rows: "
            f"{affected_indices or 'unknown'}."
        )

    return normalized


def _normalize_optional_text_column(
    series: pd.Series[Any],
    *,
    uppercase: bool,
) -> pd.Series[Any]:
    """Normalize optional text while preserving missing values."""
    normalized = series.astype("string").str.strip()

    normalized = normalized.mask(
        normalized.eq(""),
        pd.NA,
    )

    if uppercase:
        normalized = normalized.str.upper()

    return normalized


def _coerce_whole_number_column(
    *,
    frame: pd.DataFrame,
    column: str,
    allow_missing: bool,
) -> pd.Series[Any]:
    """Convert a column to positive nullable whole numbers."""
    raw_values = frame[column]

    numeric_values = pd.to_numeric(
        raw_values,
        errors="coerce",
    )

    failed_conversions = raw_values.notna() & numeric_values.isna()

    if bool(failed_conversions.any()):
        raise RaceReplayError(
            f"{column} contains non-numeric values at rows: "
            f"{_format_affected_indices(frame, failed_conversions)}."
        )

    if not allow_missing and bool(numeric_values.isna().any()):
        raise RaceReplayError(
            f"{column} contains missing values at rows: "
            f"{_format_affected_indices(frame, numeric_values.isna())}."
        )

    non_positive_values = numeric_values.notna() & numeric_values.le(0)

    if bool(non_positive_values.any()):
        raise RaceReplayError(
            f"{column} must contain positive values at rows: "
            f"{_format_affected_indices(frame, non_positive_values)}."
        )

    fractional_values = numeric_values.notna() & numeric_values.mod(1).ne(0)

    if bool(fractional_values.any()):
        raise RaceReplayError(
            f"{column} must contain whole numbers at rows: "
            f"{_format_affected_indices(frame, fractional_values)}."
        )

    return numeric_values.astype("Int64")


def _coerce_numeric_column(
    *,
    frame: pd.DataFrame,
    column: str,
    allow_missing: bool,
    non_negative: bool,
) -> pd.Series[Any]:
    """Convert a column to numeric values without hiding invalid text."""
    raw_values = frame[column]

    numeric_values = pd.to_numeric(
        raw_values,
        errors="coerce",
    )

    failed_conversions = raw_values.notna() & numeric_values.isna()

    if bool(failed_conversions.any()):
        raise RaceReplayError(
            f"{column} contains non-numeric values at rows: "
            f"{_format_affected_indices(frame, failed_conversions)}."
        )

    if not allow_missing and bool(numeric_values.isna().any()):
        raise RaceReplayError(
            f"{column} contains missing values at rows: "
            f"{_format_affected_indices(frame, numeric_values.isna())}."
        )

    if non_negative:
        negative_values = numeric_values.notna() & numeric_values.lt(0)

        if bool(negative_values.any()):
            raise RaceReplayError(
                f"{column} contains negative values at rows: "
                f"{_format_affected_indices(frame, negative_values)}."
            )

    return numeric_values


def _require_columns(
    *,
    frame: pd.DataFrame,
    required_columns: tuple[str, ...],
    object_name: str,
) -> None:
    """Raise when a DataFrame lacks required columns."""
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]

    if not missing_columns:
        return

    formatted_columns = ", ".join(missing_columns)

    raise RaceReplayError(
        f"The {object_name} is missing required columns: {formatted_columns}."
    )


def _validate_lap_range(
    *,
    start_lap: int,
    end_lap: int | None,
) -> None:
    """Validate requested replay boundaries."""
    if isinstance(start_lap, bool) or start_lap <= 0:
        raise RaceReplayError("start_lap must be a positive integer.")

    if end_lap is None:
        return

    if isinstance(end_lap, bool) or end_lap <= 0:
        raise RaceReplayError("end_lap must be a positive integer when provided.")

    if end_lap < start_lap:
        raise RaceReplayError("end_lap cannot be smaller than start_lap.")


def _format_affected_indices(
    frame: pd.DataFrame,
    mask: pd.Series[bool],
    *,
    limit: int = 5,
) -> str:
    """Return a compact list of affected source indices."""
    indices = frame.index[mask].tolist()[:limit]

    return ", ".join(str(index) for index in indices) or "unknown"
