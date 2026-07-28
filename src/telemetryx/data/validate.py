from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import pandas as pd

DRIVER_LAP_KEY_COLUMNS: Final[tuple[str, str]] = (
    "Driver",
    "LapNumber",
)

REQUIRED_LAP_COLUMNS: Final[frozenset[str]] = frozenset(
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


class ValidationSeverity(StrEnum):
    """Severity assigned to a data-validation issue."""

    ERROR = "error"
    WARNING = "warning"


class RaceDataValidationError(ValueError):
    """Raised when a validation report contains one or more errors."""


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One data-quality problem detected in a race table."""

    code: str
    severity: ValidationSeverity
    message: str
    affected_rows: int
    sample_indices: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation of the issue."""
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "affected_rows": self.affected_rows,
            "sample_indices": list(self.sample_indices),
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Complete data-validation result for one lap table."""

    row_count: int
    issues: tuple[ValidationIssue, ...]

    @property
    def has_errors(self) -> bool:
        """Return whether the report contains at least one error."""
        return any(issue.severity is ValidationSeverity.ERROR for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Return whether the report contains at least one warning."""
        return any(
            issue.severity is ValidationSeverity.WARNING for issue in self.issues
        )

    @property
    def error_count(self) -> int:
        """Return the number of error-level issues."""
        return sum(issue.severity is ValidationSeverity.ERROR for issue in self.issues)

    @property
    def warning_count(self) -> int:
        """Return the number of warning-level issues."""
        return sum(
            issue.severity is ValidationSeverity.WARNING for issue in self.issues
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation of the report."""
        return {
            "row_count": self.row_count,
            "has_errors": self.has_errors,
            "has_warnings": self.has_warnings,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def raise_for_errors(self) -> None:
        """
        Raise an exception when error-level validation issues exist.

        Warnings do not raise an exception because they may represent expected
        missing values that still require investigation.
        """
        if not self.has_errors:
            return

        error_messages = [
            f"{issue.code}: {issue.message}"
            for issue in self.issues
            if issue.severity is ValidationSeverity.ERROR
        ]

        formatted_errors = "; ".join(error_messages)

        raise RaceDataValidationError(f"Lap-data validation failed: {formatted_errors}")


def validate_lap_data(
    laps: pd.DataFrame,
) -> ValidationReport:
    """
    Validate the structural quality of a FastF1 lap table.

    The input DataFrame is inspected but never modified.

    Parameters
    ----------
    laps:
        FastF1-like lap table containing one row per recorded driver lap.

    Returns
    -------
    ValidationReport
        All detected errors and warnings.

    Raises
    ------
    TypeError
        If ``laps`` is not a pandas DataFrame.
    """
    if not isinstance(laps, pd.DataFrame):
        raise TypeError("laps must be provided as a pandas DataFrame.")

    issues: list[ValidationIssue] = []

    if laps.empty:
        issues.append(
            ValidationIssue(
                code="empty_lap_table",
                severity=ValidationSeverity.ERROR,
                message="The lap table contains no rows.",
                affected_rows=0,
            )
        )

        return ValidationReport(
            row_count=0,
            issues=tuple(issues),
        )

    duplicate_column_names = _duplicate_column_names(laps)

    if duplicate_column_names:
        formatted_columns = ", ".join(duplicate_column_names)

        issues.append(
            ValidationIssue(
                code="duplicate_column_names",
                severity=ValidationSeverity.ERROR,
                message=(
                    "The lap table contains duplicate column names: "
                    f"{formatted_columns}."
                ),
                affected_rows=len(laps),
            )
        )

        return ValidationReport(
            row_count=len(laps),
            issues=tuple(issues),
        )

    available_columns = {str(column) for column in laps.columns}

    missing_columns = sorted(REQUIRED_LAP_COLUMNS.difference(available_columns))

    if missing_columns:
        formatted_columns = ", ".join(missing_columns)

        issues.append(
            ValidationIssue(
                code="missing_required_columns",
                severity=ValidationSeverity.ERROR,
                message=(
                    f"The lap table is missing required columns: {formatted_columns}."
                ),
                affected_rows=len(laps),
            )
        )

    if _has_columns(
        laps,
        DRIVER_LAP_KEY_COLUMNS,
    ):
        duplicate_driver_laps = laps.duplicated(
            subset=list(DRIVER_LAP_KEY_COLUMNS),
            keep=False,
        )

        _append_mask_issue(
            issues=issues,
            frame=laps,
            mask=duplicate_driver_laps,
            code="duplicate_driver_lap_rows",
            severity=ValidationSeverity.ERROR,
            message=("Multiple rows use the same Driver and LapNumber."),
        )

    if "Driver" in laps.columns:
        driver_values = laps["Driver"]

        missing_drivers = _missing_text_mask(driver_values)

        _append_mask_issue(
            issues=issues,
            frame=laps,
            mask=missing_drivers,
            code="missing_driver",
            severity=ValidationSeverity.ERROR,
            message=("One or more lap rows have no driver identifier."),
        )

        non_missing_driver_text = driver_values.astype("string").fillna("")

        driver_whitespace = non_missing_driver_text.ne(
            non_missing_driver_text.str.strip()
        )

        _append_mask_issue(
            issues=issues,
            frame=laps,
            mask=driver_whitespace,
            code="driver_whitespace",
            severity=ValidationSeverity.WARNING,
            message=(
                "One or more driver identifiers contain leading or trailing whitespace."
            ),
        )

    if "LapNumber" in laps.columns:
        lap_numbers = _coerce_numeric(laps["LapNumber"])

        missing_lap_numbers = lap_numbers.isna()

        _append_mask_issue(
            issues=issues,
            frame=laps,
            mask=missing_lap_numbers,
            code="invalid_lap_number",
            severity=ValidationSeverity.ERROR,
            message=("One or more lap numbers are missing or non-numeric."),
        )

        non_positive_laps = lap_numbers.notna() & lap_numbers.le(0)

        _append_mask_issue(
            issues=issues,
            frame=laps,
            mask=non_positive_laps,
            code="non_positive_lap_number",
            severity=ValidationSeverity.ERROR,
            message=("Lap numbers must be greater than zero."),
        )

        fractional_laps = lap_numbers.notna() & lap_numbers.mod(1).ne(0)

        _append_mask_issue(
            issues=issues,
            frame=laps,
            mask=fractional_laps,
            code="fractional_lap_number",
            severity=ValidationSeverity.ERROR,
            message=("Lap numbers must be whole numbers."),
        )

    if "Position" in laps.columns:
        _validate_position_column(
            laps=laps,
            issues=issues,
        )

    if "Stint" in laps.columns:
        stint_numbers = _coerce_numeric(laps["Stint"])

        invalid_stints = stint_numbers.notna() & (
            stint_numbers.le(0) | stint_numbers.mod(1).ne(0)
        )

        _append_mask_issue(
            issues=issues,
            frame=laps,
            mask=invalid_stints,
            code="invalid_stint_number",
            severity=ValidationSeverity.ERROR,
            message=("Stint numbers must be positive whole numbers."),
        )

    if "TyreLife" in laps.columns:
        tyre_life = _coerce_numeric(laps["TyreLife"])

        negative_tyre_life = tyre_life.notna() & tyre_life.lt(0)

        _append_mask_issue(
            issues=issues,
            frame=laps,
            mask=negative_tyre_life,
            code="negative_tyre_life",
            severity=ValidationSeverity.ERROR,
            message=("Tyre life cannot be negative."),
        )

    if "LapTime" in laps.columns:
        _validate_lap_time_column(
            laps=laps,
            issues=issues,
        )

    if "Compound" in laps.columns:
        missing_compounds = _missing_text_mask(laps["Compound"])

        _append_mask_issue(
            issues=issues,
            frame=laps,
            mask=missing_compounds,
            code="missing_tyre_compound",
            severity=ValidationSeverity.WARNING,
            message=("One or more lap rows have no tyre compound."),
        )

    if "TrackStatus" in laps.columns:
        missing_track_status = _missing_text_mask(laps["TrackStatus"])

        _append_mask_issue(
            issues=issues,
            frame=laps,
            mask=missing_track_status,
            code="missing_track_status",
            severity=ValidationSeverity.WARNING,
            message=("One or more lap rows have no track-status value."),
        )

    return ValidationReport(
        row_count=len(laps),
        issues=tuple(issues),
    )


def _validate_position_column(
    laps: pd.DataFrame,
    issues: list[ValidationIssue],
) -> None:
    """Validate current-position values."""
    positions = _coerce_numeric(laps["Position"])

    missing_positions = positions.isna()

    _append_mask_issue(
        issues=issues,
        frame=laps,
        mask=missing_positions,
        code="missing_position",
        severity=ValidationSeverity.WARNING,
        message=("One or more lap rows have no current position."),
    )

    non_positive_positions = positions.notna() & positions.le(0)

    _append_mask_issue(
        issues=issues,
        frame=laps,
        mask=non_positive_positions,
        code="non_positive_position",
        severity=ValidationSeverity.ERROR,
        message=("Current positions must be greater than zero."),
    )

    fractional_positions = positions.notna() & positions.mod(1).ne(0)

    _append_mask_issue(
        issues=issues,
        frame=laps,
        mask=fractional_positions,
        code="fractional_position",
        severity=ValidationSeverity.ERROR,
        message=("Current positions must be whole numbers."),
    )

    if "Driver" not in laps.columns:
        return

    driver_count = int(laps["Driver"].dropna().nunique())

    if driver_count == 0:
        return

    positions_above_driver_count = positions.notna() & positions.gt(driver_count)

    _append_mask_issue(
        issues=issues,
        frame=laps,
        mask=positions_above_driver_count,
        code="position_above_driver_count",
        severity=ValidationSeverity.ERROR,
        message=(
            "A current position exceeds the number of drivers represented in the table."
        ),
    )


def _validate_lap_time_column(
    laps: pd.DataFrame,
    issues: list[ValidationIssue],
) -> None:
    """Validate lap-time values without modifying the source column."""
    raw_lap_times = laps["LapTime"]
    converted_lap_times = _coerce_timedelta(raw_lap_times)

    originally_missing = raw_lap_times.isna()

    invalid_lap_times = ~originally_missing & converted_lap_times.isna()

    _append_mask_issue(
        issues=issues,
        frame=laps,
        mask=invalid_lap_times,
        code="invalid_lap_time",
        severity=ValidationSeverity.ERROR,
        message=(
            "One or more non-missing lap times could not "
            "be interpreted as time durations."
        ),
    )

    missing_lap_times = originally_missing

    _append_mask_issue(
        issues=issues,
        frame=laps,
        mask=missing_lap_times,
        code="missing_lap_time",
        severity=ValidationSeverity.WARNING,
        message=("One or more lap rows have no recorded lap time."),
    )

    non_positive_lap_times = converted_lap_times.notna() & converted_lap_times.le(
        pd.Timedelta(0)
    )

    _append_mask_issue(
        issues=issues,
        frame=laps,
        mask=non_positive_lap_times,
        code="non_positive_lap_time",
        severity=ValidationSeverity.ERROR,
        message=("Recorded lap times must be greater than zero."),
    )


def _append_mask_issue(
    *,
    issues: list[ValidationIssue],
    frame: pd.DataFrame,
    mask: pd.Series[bool],
    code: str,
    severity: ValidationSeverity,
    message: str,
) -> None:
    """Append one issue when a boolean row mask matches data."""
    affected_rows = int(mask.sum())

    if affected_rows == 0:
        return

    issues.append(
        ValidationIssue(
            code=code,
            severity=severity,
            message=message,
            affected_rows=affected_rows,
            sample_indices=_sample_indices(
                frame=frame,
                mask=mask,
            ),
        )
    )


def _sample_indices(
    *,
    frame: pd.DataFrame,
    mask: pd.Series[bool],
    limit: int = 5,
) -> tuple[str, ...]:
    """Return a small sample of affected DataFrame index labels."""
    matched_rows = frame.loc[mask]
    labels = matched_rows.index[:limit].tolist()

    return tuple(str(label) for label in labels)


def _missing_text_mask(
    series: pd.Series[Any],
) -> pd.Series[bool]:
    """Return rows containing missing or blank text."""
    normalized = series.astype("string").str.strip()

    return series.isna() | normalized.eq("")


def _coerce_numeric(
    series: pd.Series[Any],
) -> pd.Series[Any]:
    """Convert a Series to numeric values, coercing invalid data to NaN."""
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def _coerce_timedelta(
    series: pd.Series[Any],
) -> pd.Series[Any]:
    """Convert a Series to timedeltas, coercing invalid data to NaT."""
    result = pd.to_timedelta(
        series,
        errors="coerce",
    )

    if not isinstance(result, pd.Series):
        raise TypeError(
            "Timedelta conversion unexpectedly returned a non-Series value."
        )

    return result


def _duplicate_column_names(
    frame: pd.DataFrame,
) -> tuple[str, ...]:
    """Return duplicated DataFrame column names."""
    duplicated = frame.columns[frame.columns.duplicated()]

    return tuple(str(column) for column in duplicated)


def _has_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> bool:
    """Return whether every requested column exists."""
    return all(column in frame.columns for column in columns)
