from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd

TARGET_COLUMN: Final[str] = "WonRace"

WINNER_TARGET_COLUMNS: Final[tuple[str, ...]] = (
    "Driver",
    TARGET_COLUMN,
)

RESULT_DRIVER_COLUMN_CANDIDATES: Final[tuple[str, ...]] = (
    "Abbreviation",
    "Driver",
)

REQUIRED_REPLAY_COLUMNS: Final[tuple[str, ...]] = (
    "SnapshotLap",
    "Driver",
)


class RaceTargetError(ValueError):
    """Raised when race-winner targets cannot be constructed safely."""


def build_winner_targets(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one binary winner target per classified driver.

    The final result position is used only to construct the target. It is not
    returned in the target table and therefore cannot accidentally become a
    model feature.

    Parameters
    ----------
    results:
        FastF1-like final results table containing a driver identifier and
        final ``Position``.

    Returns
    -------
    pd.DataFrame
        Two-column table containing ``Driver`` and ``WonRace``.

    Raises
    ------
    TypeError
        If ``results`` is not a pandas DataFrame.
    RaceTargetError
        If the final result table is empty, malformed, or does not contain
        exactly one winner.
    """
    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be provided as a pandas DataFrame.")

    if results.empty:
        raise RaceTargetError(
            "Cannot build winner targets from an empty results table."
        )

    duplicate_columns = _duplicate_column_names(results)

    if duplicate_columns:
        formatted_columns = ", ".join(duplicate_columns)

        raise RaceTargetError(
            f"The results table contains duplicate column names: {formatted_columns}."
        )

    driver_column = _select_result_driver_column(results)

    _require_columns(
        frame=results,
        required_columns=("Position",),
        object_name="results table",
    )

    working = pd.DataFrame(
        {
            "Driver": _normalize_driver_series(
                results[driver_column],
                field_name=driver_column,
            ),
            "_FinalPosition": _coerce_final_positions(results["Position"]),
        }
    )

    duplicate_drivers = working.duplicated(
        subset=["Driver"],
        keep=False,
    )

    if bool(duplicate_drivers.any()):
        affected_drivers = sorted(
            {
                str(driver)
                for driver in working.loc[
                    duplicate_drivers,
                    "Driver",
                ].tolist()
            }
        )

        raise RaceTargetError(
            "The results table contains duplicate driver rows: "
            f"{', '.join(affected_drivers)}."
        )

    winner_mask = working["_FinalPosition"].eq(1)
    winner_count = int(winner_mask.sum())

    if winner_count != 1:
        raise RaceTargetError(
            "The final results must contain exactly one position-one "
            f"driver; found {winner_count}."
        )

    working[TARGET_COLUMN] = winner_mask.astype("boolean")

    targets = (
        working.sort_values(
            by=[
                "_FinalPosition",
                "Driver",
            ],
            kind="stable",
        )
        .loc[
            :,
            list(WINNER_TARGET_COLUMNS),
        ]
        .reset_index(drop=True)
    )

    return targets


def attach_winner_targets(
    replay: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach the binary race-winner target to every replay row.

    The replay table contains temporally available race-state features.
    ``WonRace`` is a future outcome and must be used only as the supervised
    learning target.

    Parameters
    ----------
    replay:
        Temporally safe replay table with one row per snapshot and driver.
    targets:
        Driver-level target table returned by
        :func:`build_winner_targets`.

    Returns
    -------
    pd.DataFrame
        Copy of the replay table with one additional ``WonRace`` column.

    Raises
    ------
    TypeError
        If either argument is not a pandas DataFrame.
    RaceTargetError
        If the tables are malformed, drivers cannot be matched, or a snapshot
        does not contain exactly one positive winner target.
    """
    if not isinstance(replay, pd.DataFrame):
        raise TypeError("replay must be provided as a pandas DataFrame.")

    if not isinstance(targets, pd.DataFrame):
        raise TypeError("targets must be provided as a pandas DataFrame.")

    if replay.empty:
        raise RaceTargetError("Cannot attach targets to an empty replay table.")

    if targets.empty:
        raise RaceTargetError("Cannot attach an empty winner-target table.")

    _reject_duplicate_columns(
        replay,
        object_name="replay table",
    )

    _reject_duplicate_columns(
        targets,
        object_name="target table",
    )

    _require_columns(
        frame=replay,
        required_columns=REQUIRED_REPLAY_COLUMNS,
        object_name="replay table",
    )

    _require_columns(
        frame=targets,
        required_columns=WINNER_TARGET_COLUMNS,
        object_name="target table",
    )

    if TARGET_COLUMN in replay.columns:
        raise RaceTargetError(f"The replay table already contains {TARGET_COLUMN}.")

    prepared_replay = replay.copy(deep=True)

    prepared_replay["Driver"] = _normalize_driver_series(
        prepared_replay["Driver"],
        field_name="replay Driver",
    )

    prepared_targets = targets.loc[
        :,
        list(WINNER_TARGET_COLUMNS),
    ].copy()

    prepared_targets["Driver"] = _normalize_driver_series(
        prepared_targets["Driver"],
        field_name="target Driver",
    )

    prepared_targets[TARGET_COLUMN] = _normalize_target_series(
        prepared_targets[TARGET_COLUMN]
    )

    duplicate_target_drivers = prepared_targets.duplicated(
        subset=["Driver"],
        keep=False,
    )

    if bool(duplicate_target_drivers.any()):
        affected_drivers = sorted(
            {
                str(driver)
                for driver in prepared_targets.loc[
                    duplicate_target_drivers,
                    "Driver",
                ].tolist()
            }
        )

        raise RaceTargetError(
            "The target table contains duplicate drivers: "
            f"{', '.join(affected_drivers)}."
        )

    target_winner_count = int(prepared_targets[TARGET_COLUMN].sum())

    if target_winner_count != 1:
        raise RaceTargetError(
            "The target table must contain exactly one winner; "
            f"found {target_winner_count}."
        )

    replay_drivers = {str(driver) for driver in prepared_replay["Driver"].tolist()}

    target_drivers = {str(driver) for driver in prepared_targets["Driver"].tolist()}

    missing_target_drivers = sorted(replay_drivers.difference(target_drivers))

    if missing_target_drivers:
        raise RaceTargetError(
            "Winner targets are missing for replay drivers: "
            f"{', '.join(missing_target_drivers)}."
        )

    training_rows = prepared_replay.merge(
        prepared_targets,
        on="Driver",
        how="left",
        sort=False,
        validate="many_to_one",
    )

    if bool(training_rows[TARGET_COLUMN].isna().any()):
        raise RaceTargetError(
            "One or more replay rows could not be matched to a winner target."
        )

    training_rows[TARGET_COLUMN] = training_rows[TARGET_COLUMN].astype("boolean")

    _validate_snapshot_targets(training_rows)

    return training_rows


def _select_result_driver_column(
    results: pd.DataFrame,
) -> str:
    """Return the supported driver-identifier column in a results table."""
    for column in RESULT_DRIVER_COLUMN_CANDIDATES:
        if column in results.columns:
            return column

    formatted_candidates = ", ".join(RESULT_DRIVER_COLUMN_CANDIDATES)

    raise RaceTargetError(
        "The results table has no supported driver column. "
        f"Expected one of: {formatted_candidates}."
    )


def _normalize_driver_series(
    series: pd.Series[Any],
    *,
    field_name: str,
) -> pd.Series[Any]:
    """Normalize driver identifiers and reject missing values."""
    normalized = series.astype("string").str.strip().str.upper()

    missing_values = normalized.isna() | normalized.eq("")

    if bool(missing_values.any()):
        affected_indices = series.index[missing_values].tolist()[:5]

        formatted_indices = (
            ", ".join(str(index) for index in affected_indices) or "unknown"
        )

        raise RaceTargetError(
            f"{field_name} contains missing or blank driver identifiers "
            f"at rows: {formatted_indices}."
        )

    return normalized


def _coerce_final_positions(
    series: pd.Series[Any],
) -> pd.Series[Any]:
    """Convert final positions to positive nullable integers."""
    numeric_positions = pd.to_numeric(
        series,
        errors="coerce",
    )

    failed_conversions = series.notna() & numeric_positions.isna()

    if bool(failed_conversions.any()):
        affected_indices = series.index[failed_conversions].tolist()[:5]

        raise RaceTargetError(
            "Position contains non-numeric values at rows: "
            f"{_format_indices(affected_indices)}."
        )

    missing_positions = numeric_positions.isna()

    if bool(missing_positions.any()):
        affected_indices = series.index[missing_positions].tolist()[:5]

        raise RaceTargetError(
            "Position contains missing values at rows: "
            f"{_format_indices(affected_indices)}."
        )

    non_positive_positions = numeric_positions.le(0)

    if bool(non_positive_positions.any()):
        affected_indices = series.index[non_positive_positions].tolist()[:5]

        raise RaceTargetError(
            "Position must contain positive values at rows: "
            f"{_format_indices(affected_indices)}."
        )

    fractional_positions = numeric_positions.mod(1).ne(0)

    if bool(fractional_positions.any()):
        affected_indices = series.index[fractional_positions].tolist()[:5]

        raise RaceTargetError(
            "Position must contain whole numbers at rows: "
            f"{_format_indices(affected_indices)}."
        )

    return numeric_positions.astype("Int64")


def _normalize_target_series(
    series: pd.Series[Any],
) -> pd.Series[Any]:
    """Validate and return a nullable Boolean winner-target Series."""
    if bool(series.isna().any()):
        affected_indices = series.index[series.isna()].tolist()[:5]

        raise RaceTargetError(
            f"{TARGET_COLUMN} contains missing values at rows: "
            f"{_format_indices(affected_indices)}."
        )

    boolean_values = series.map(
        lambda value: isinstance(
            value,
            (bool, np.bool_),
        )
    )

    if not bool(boolean_values.all()):
        invalid_indices = series.index[~boolean_values].tolist()[:5]

        raise RaceTargetError(
            f"{TARGET_COLUMN} must contain only boolean values at rows: "
            f"{_format_indices(invalid_indices)}."
        )

    return series.astype("boolean")


def _validate_snapshot_targets(
    training_rows: pd.DataFrame,
) -> None:
    """Ensure every replay snapshot contains exactly one positive target."""
    if bool(training_rows["SnapshotLap"].isna().any()):
        raise RaceTargetError(
            "SnapshotLap contains missing values after target attachment."
        )

    winner_counts = training_rows.groupby(
        "SnapshotLap",
        sort=False,
    )[TARGET_COLUMN].sum()

    invalid_snapshot_laps = [
        str(snapshot_lap)
        for snapshot_lap, winner_count in winner_counts.items()
        if int(winner_count) != 1
    ]

    if invalid_snapshot_laps:
        raise RaceTargetError(
            "Each replay snapshot must contain exactly one positive "
            "winner target. Invalid snapshot laps: "
            f"{', '.join(invalid_snapshot_laps)}."
        )


def _reject_duplicate_columns(
    frame: pd.DataFrame,
    *,
    object_name: str,
) -> None:
    """Reject ambiguous duplicate DataFrame column names."""
    duplicate_columns = _duplicate_column_names(frame)

    if not duplicate_columns:
        return

    raise RaceTargetError(
        f"The {object_name} contains duplicate column names: "
        f"{', '.join(duplicate_columns)}."
    )


def _duplicate_column_names(
    frame: pd.DataFrame,
) -> tuple[str, ...]:
    """Return duplicated DataFrame column names."""
    duplicated = frame.columns[frame.columns.duplicated()]

    return tuple(str(column) for column in duplicated)


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

    raise RaceTargetError(
        f"The {object_name} is missing required columns: {', '.join(missing_columns)}."
    )


def _format_indices(
    indices: list[Any],
) -> str:
    """Return a compact representation of affected row indices."""
    return ", ".join(str(index) for index in indices) or "unknown"
