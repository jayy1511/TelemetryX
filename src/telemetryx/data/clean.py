from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import pandas as pd

from telemetryx.data.validate import (
    ValidationReport,
    validate_lap_data,
)

STANDARDIZED_COLUMNS: Final[tuple[str, ...]] = (
    "Driver",
    "LapNumber",
    "LapTime",
    "Position",
    "Stint",
    "Compound",
    "TyreLife",
    "TrackStatus",
)

INTEGER_COLUMNS: Final[tuple[str, ...]] = (
    "LapNumber",
    "Position",
    "Stint",
)

TEXT_COLUMNS: Final[tuple[str, ...]] = (
    "Driver",
    "Compound",
    "TrackStatus",
)


class RaceDataCleaningError(RuntimeError):
    """Raised when lap data cannot be cleaned safely."""


@dataclass(frozen=True, slots=True)
class CleaningResult:
    """Cleaned lap data together with its validation reports."""

    laps: pd.DataFrame
    input_validation: ValidationReport
    output_validation: ValidationReport


def clean_lap_data(
    laps: pd.DataFrame,
    *,
    fail_on_input_errors: bool = True,
) -> CleaningResult:
    """
    Return a standardized copy of a FastF1 lap table.

    The input DataFrame is never modified. No rows are silently removed and
    missing values remain missing.

    Parameters
    ----------
    laps:
        FastF1-like lap table.
    fail_on_input_errors:
        Whether error-level validation findings in the input should stop
        cleaning immediately.

    Returns
    -------
    CleaningResult
        Cleaned lap data and validation reports from before and after cleaning.

    Raises
    ------
    TypeError
        If ``laps`` is not a pandas DataFrame.
    RaceDataValidationError
        If input validation contains errors and strict mode is enabled.
    RaceDataCleaningError
        If required columns disappear or cleaning produces invalid output.
    """
    if not isinstance(laps, pd.DataFrame):
        raise TypeError("laps must be provided as a pandas DataFrame.")

    input_validation = validate_lap_data(laps)

    if fail_on_input_errors:
        input_validation.raise_for_errors()

    cleaned = laps.copy(deep=True)

    _standardize_text_columns(cleaned)
    _standardize_integer_columns(cleaned)
    _standardize_tyre_life(cleaned)
    _standardize_lap_times(cleaned)

    output_validation = validate_lap_data(cleaned)

    if output_validation.has_errors:
        error_codes = ", ".join(
            issue.code
            for issue in output_validation.issues
            if issue.severity.value == "error"
        )

        raise RaceDataCleaningError(
            f"Cleaning produced invalid lap data. Validation errors: {error_codes}."
        )

    return CleaningResult(
        laps=cleaned,
        input_validation=input_validation,
        output_validation=output_validation,
    )


def select_standardized_lap_columns(
    laps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a copy containing only the minimum standardized lap columns.

    This function is intended for inspection and early pipeline development.
    It does not modify the original DataFrame.

    Parameters
    ----------
    laps:
        DataFrame containing the required standardized columns.

    Returns
    -------
    pd.DataFrame
        Copy containing columns in a deterministic order.

    Raises
    ------
    RaceDataCleaningError
        If one or more required columns are missing.
    """
    missing_columns = [
        column for column in STANDARDIZED_COLUMNS if column not in laps.columns
    ]

    if missing_columns:
        formatted = ", ".join(missing_columns)

        raise RaceDataCleaningError(
            f"Cannot select standardized lap columns. Missing columns: {formatted}."
        )

    return laps.loc[:, list(STANDARDIZED_COLUMNS)].copy()


def _standardize_text_columns(
    frame: pd.DataFrame,
) -> None:
    """Strip whitespace and normalize selected text columns."""
    for column in TEXT_COLUMNS:
        if column not in frame.columns:
            continue

        normalized = frame[column].astype("string").str.strip()

        normalized = normalized.mask(
            normalized.eq(""),
            pd.NA,
        )

        if column in {"Driver", "Compound"}:
            normalized = normalized.str.upper()

        frame[column] = normalized


def _standardize_integer_columns(
    frame: pd.DataFrame,
) -> None:
    """Convert selected numeric columns to nullable integer values."""
    for column in INTEGER_COLUMNS:
        if column not in frame.columns:
            continue

        numeric_values = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        fractional_mask = numeric_values.notna() & numeric_values.mod(1).ne(0)

        if bool(fractional_mask.any()):
            invalid_indices = _format_indices(frame.index[fractional_mask])

            raise RaceDataCleaningError(
                f"{column} contains fractional values at rows: {invalid_indices}."
            )

        try:
            frame[column] = numeric_values.astype("Int64")
        except (TypeError, ValueError) as exc:
            raise RaceDataCleaningError(
                f"Could not convert {column} to nullable integers."
            ) from exc


def _standardize_tyre_life(
    frame: pd.DataFrame,
) -> None:
    """Convert tyre life to nullable floating-point values."""
    if "TyreLife" not in frame.columns:
        return

    tyre_life = pd.to_numeric(
        frame["TyreLife"],
        errors="coerce",
    )

    try:
        frame["TyreLife"] = tyre_life.astype("Float64")
    except (TypeError, ValueError) as exc:
        raise RaceDataCleaningError(
            "Could not convert TyreLife to nullable floats."
        ) from exc


def _standardize_lap_times(
    frame: pd.DataFrame,
) -> None:
    """Convert lap times to pandas timedeltas and add seconds."""
    if "LapTime" not in frame.columns:
        return

    original_lap_times = frame["LapTime"]

    converted_lap_times = pd.to_timedelta(
        original_lap_times,
        errors="coerce",
    )

    invalid_conversion = original_lap_times.notna() & converted_lap_times.isna()

    if bool(invalid_conversion.any()):
        invalid_indices = _format_indices(frame.index[invalid_conversion])

        raise RaceDataCleaningError(
            "LapTime contains values that cannot be converted "
            f"at rows: {invalid_indices}."
        )

    frame["LapTime"] = converted_lap_times

    lap_time_seconds = converted_lap_times / pd.Timedelta(seconds=1)

    frame["LapTimeSeconds"] = lap_time_seconds.astype("Float64")


def _format_indices(
    indices: Any,
    *,
    limit: int = 5,
) -> str:
    """Return a compact representation of affected row indices."""
    values = [str(value) for value in list(indices)[:limit]]

    return ", ".join(values) or "unknown"
