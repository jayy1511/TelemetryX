from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from telemetryx.data.validate import (
    RaceDataValidationError,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
    validate_lap_data,
)


def make_valid_laps() -> pd.DataFrame:
    """Return a small, structurally valid FastF1-like lap table."""
    return pd.DataFrame(
        {
            "Driver": [
                "VER",
                "NOR",
                "LEC",
                "VER",
                "NOR",
                "LEC",
            ],
            "LapNumber": [
                1,
                1,
                1,
                2,
                2,
                2,
            ],
            "LapTime": pd.to_timedelta(
                [
                    "00:01:38.100",
                    "00:01:38.500",
                    "00:01:39.000",
                    "00:01:37.900",
                    "00:01:38.200",
                    "00:01:38.700",
                ]
            ),
            "Position": [
                1,
                2,
                3,
                1,
                2,
                3,
            ],
            "Stint": [
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            "Compound": [
                "SOFT",
                "SOFT",
                "SOFT",
                "SOFT",
                "SOFT",
                "SOFT",
            ],
            "TyreLife": [
                1,
                1,
                1,
                2,
                2,
                2,
            ],
            "TrackStatus": [
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
            ],
        }
    )


def issue_codes(
    report: ValidationReport,
) -> set[str]:
    """Return the issue codes contained in a validation report."""
    return {issue.code for issue in report.issues}


def find_issue(
    report: ValidationReport,
    code: str,
) -> ValidationIssue:
    """Return one issue with the requested code."""
    for issue in report.issues:
        if issue.code == code:
            return issue

    raise AssertionError(f"Validation issue was not found: {code}")


def test_valid_lap_data_has_no_issues() -> None:
    """A structurally valid lap table should pass validation."""
    laps = make_valid_laps()

    report = validate_lap_data(laps)

    assert report.row_count == 6
    assert report.issues == ()
    assert report.has_errors is False
    assert report.has_warnings is False
    assert report.error_count == 0
    assert report.warning_count == 0


def test_validation_does_not_modify_input_dataframe() -> None:
    """Validation must never alter the source lap table."""
    laps = make_valid_laps()
    original = laps.copy(deep=True)

    validate_lap_data(laps)

    pd.testing.assert_frame_equal(
        laps,
        original,
    )


def test_non_dataframe_input_is_rejected() -> None:
    """The validator should reject objects that are not DataFrames."""
    invalid_input: Any = [
        {
            "Driver": "VER",
            "LapNumber": 1,
        }
    ]

    with pytest.raises(
        TypeError,
        match="laps must be provided as a pandas DataFrame",
    ):
        validate_lap_data(invalid_input)


def test_empty_lap_table_returns_error_report() -> None:
    """An empty lap table should create an error-level issue."""
    report = validate_lap_data(pd.DataFrame())

    assert report.row_count == 0
    assert report.has_errors is True
    assert report.error_count == 1
    assert report.warning_count == 0
    assert issue_codes(report) == {"empty_lap_table"}


def test_missing_required_columns_are_reported() -> None:
    """All required lap-table columns must be present."""
    laps = make_valid_laps().drop(
        columns=[
            "Compound",
            "TrackStatus",
        ]
    )

    report = validate_lap_data(laps)

    assert report.has_errors is True

    issue = find_issue(
        report,
        "missing_required_columns",
    )

    assert issue.severity is ValidationSeverity.ERROR
    assert issue.affected_rows == len(laps)
    assert "Compound" in issue.message
    assert "TrackStatus" in issue.message


def test_duplicate_column_names_are_rejected() -> None:
    """Ambiguous duplicate column names should stop further validation."""
    laps = make_valid_laps()

    renamed_columns = list(laps.columns)
    renamed_columns[-1] = "Compound"
    laps.columns = renamed_columns

    report = validate_lap_data(laps)

    assert report.has_errors is True
    assert report.error_count == 1
    assert issue_codes(report) == {"duplicate_column_names"}

    issue = find_issue(
        report,
        "duplicate_column_names",
    )

    assert "Compound" in issue.message


def test_duplicate_driver_lap_rows_are_reported() -> None:
    """A driver may have only one row for each recorded lap number."""
    laps = make_valid_laps()

    duplicate_row = laps.iloc[[0]].copy()

    laps_with_duplicate = pd.concat(
        [
            laps,
            duplicate_row,
        ],
        ignore_index=True,
    )

    report = validate_lap_data(laps_with_duplicate)

    issue = find_issue(
        report,
        "duplicate_driver_lap_rows",
    )

    assert issue.severity is ValidationSeverity.ERROR
    assert issue.affected_rows == 2
    assert len(issue.sample_indices) == 2


def test_missing_and_whitespace_driver_values_are_reported() -> None:
    """Missing drivers are errors and surrounding whitespace is a warning."""
    laps = make_valid_laps()
    laps["Driver"] = laps["Driver"].astype("object")

    laps.loc[0, "Driver"] = None
    laps.loc[1, "Driver"] = " NOR "

    report = validate_lap_data(laps)

    missing_issue = find_issue(
        report,
        "missing_driver",
    )

    whitespace_issue = find_issue(
        report,
        "driver_whitespace",
    )

    assert missing_issue.severity is ValidationSeverity.ERROR
    assert missing_issue.affected_rows == 1

    assert whitespace_issue.severity is ValidationSeverity.WARNING
    assert whitespace_issue.affected_rows == 1


@pytest.mark.parametrize(
    ("invalid_value", "expected_code"),
    [
        ("not-a-number", "invalid_lap_number"),
        (None, "invalid_lap_number"),
        (0, "non_positive_lap_number"),
        (-1, "non_positive_lap_number"),
        (1.5, "fractional_lap_number"),
    ],
)
def test_invalid_lap_numbers_are_reported(
    invalid_value: object,
    expected_code: str,
) -> None:
    """Lap numbers must be present, numeric, positive, and whole."""
    laps = make_valid_laps()
    laps["LapNumber"] = laps["LapNumber"].astype("object")

    laps.loc[0, "LapNumber"] = invalid_value

    report = validate_lap_data(laps)

    assert expected_code in issue_codes(report)

    issue = find_issue(
        report,
        expected_code,
    )

    assert issue.severity is ValidationSeverity.ERROR
    assert issue.affected_rows == 1


def test_invalid_positions_are_reported() -> None:
    """Positions must be positive whole values within the driver count."""
    laps = make_valid_laps()
    laps["Position"] = laps["Position"].astype("object")

    laps.loc[0, "Position"] = None
    laps.loc[1, "Position"] = 0
    laps.loc[2, "Position"] = 1.5
    laps.loc[3, "Position"] = 4

    report = validate_lap_data(laps)
    codes = issue_codes(report)

    assert "missing_position" in codes
    assert "non_positive_position" in codes
    assert "fractional_position" in codes
    assert "position_above_driver_count" in codes

    assert (
        find_issue(
            report,
            "missing_position",
        ).severity
        is ValidationSeverity.WARNING
    )

    assert (
        find_issue(
            report,
            "non_positive_position",
        ).severity
        is ValidationSeverity.ERROR
    )


def test_invalid_stints_and_negative_tyre_life_are_reported() -> None:
    """Stint numbers and tyre ages must satisfy their numeric rules."""
    laps = make_valid_laps()
    laps["Stint"] = laps["Stint"].astype("object")
    laps["TyreLife"] = laps["TyreLife"].astype("object")

    laps.loc[0, "Stint"] = 0
    laps.loc[1, "Stint"] = 1.5
    laps.loc[2, "TyreLife"] = -1

    report = validate_lap_data(laps)

    stint_issue = find_issue(
        report,
        "invalid_stint_number",
    )

    tyre_issue = find_issue(
        report,
        "negative_tyre_life",
    )

    assert stint_issue.affected_rows == 2
    assert tyre_issue.affected_rows == 1

    assert stint_issue.severity is ValidationSeverity.ERROR
    assert tyre_issue.severity is ValidationSeverity.ERROR


def test_invalid_missing_and_non_positive_lap_times_are_reported() -> None:
    """Lap-time validation should distinguish different failure types."""
    laps = make_valid_laps()
    laps["LapTime"] = laps["LapTime"].astype("object")

    laps.loc[0, "LapTime"] = None
    laps.loc[1, "LapTime"] = "not-a-time"
    laps.loc[2, "LapTime"] = pd.Timedelta(0)

    report = validate_lap_data(laps)

    missing_issue = find_issue(
        report,
        "missing_lap_time",
    )

    invalid_issue = find_issue(
        report,
        "invalid_lap_time",
    )

    non_positive_issue = find_issue(
        report,
        "non_positive_lap_time",
    )

    assert missing_issue.affected_rows == 1
    assert invalid_issue.affected_rows == 1
    assert non_positive_issue.affected_rows == 1

    assert missing_issue.severity is ValidationSeverity.WARNING
    assert invalid_issue.severity is ValidationSeverity.ERROR
    assert non_positive_issue.severity is ValidationSeverity.ERROR


def test_missing_compounds_and_track_status_are_warnings() -> None:
    """Missing categorical race values should be reported as warnings."""
    laps = make_valid_laps()
    laps["Compound"] = laps["Compound"].astype("object")
    laps["TrackStatus"] = laps["TrackStatus"].astype("object")

    laps.loc[0, "Compound"] = None
    laps.loc[1, "Compound"] = "   "

    laps.loc[2, "TrackStatus"] = None
    laps.loc[3, "TrackStatus"] = ""

    report = validate_lap_data(laps)

    compound_issue = find_issue(
        report,
        "missing_tyre_compound",
    )

    track_status_issue = find_issue(
        report,
        "missing_track_status",
    )

    assert compound_issue.severity is ValidationSeverity.WARNING
    assert compound_issue.affected_rows == 2

    assert track_status_issue.severity is ValidationSeverity.WARNING
    assert track_status_issue.affected_rows == 2


def test_report_to_dict_is_json_compatible() -> None:
    """A validation report should serialize into ordinary Python values."""
    laps = make_valid_laps()
    laps.loc[0, "TyreLife"] = -1
    laps.loc[1, "Compound"] = None

    report = validate_lap_data(laps)
    serialized = report.to_dict()

    assert serialized["row_count"] == 6
    assert serialized["has_errors"] is True
    assert serialized["has_warnings"] is True
    assert serialized["error_count"] == 1
    assert serialized["warning_count"] == 1

    issues = serialized["issues"]

    assert isinstance(issues, list)
    assert {issue["code"] for issue in issues if isinstance(issue, dict)} == {
        "negative_tyre_life",
        "missing_tyre_compound",
    }


def test_raise_for_errors_does_nothing_for_valid_report() -> None:
    """A report without errors should not raise an exception."""
    report = validate_lap_data(make_valid_laps())

    report.raise_for_errors()


def test_raise_for_errors_allows_warning_only_report() -> None:
    """Warnings alone should not stop the pipeline."""
    laps = make_valid_laps()
    laps.loc[0, "Compound"] = None

    report = validate_lap_data(laps)

    assert report.has_warnings is True
    assert report.has_errors is False

    report.raise_for_errors()


def test_raise_for_errors_raises_for_error_report() -> None:
    """Error-level findings should stop strict processing."""
    laps = make_valid_laps()
    laps.loc[0, "TyreLife"] = -1

    report = validate_lap_data(laps)

    with pytest.raises(
        RaceDataValidationError,
        match="negative_tyre_life",
    ):
        report.raise_for_errors()


def test_validation_issue_to_dict_converts_enum_and_tuple() -> None:
    """Individual issues should serialize into JSON-compatible values."""
    issue = ValidationIssue(
        code="example_issue",
        severity=ValidationSeverity.WARNING,
        message="Example validation warning.",
        affected_rows=2,
        sample_indices=(
            "4",
            "9",
        ),
    )

    assert issue.to_dict() == {
        "code": "example_issue",
        "severity": "warning",
        "message": "Example validation warning.",
        "affected_rows": 2,
        "sample_indices": [
            "4",
            "9",
        ],
    }
