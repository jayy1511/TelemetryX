from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from telemetryx.data.clean import (
    STANDARDIZED_COLUMNS,
    RaceDataCleaningError,
    clean_lap_data,
    select_standardized_lap_columns,
)
from telemetryx.data.validate import RaceDataValidationError


def make_valid_laps() -> pd.DataFrame:
    """Return a valid FastF1-like lap table requiring minor cleaning."""
    return pd.DataFrame(
        {
            "Driver": [
                " ver ",
                "nor",
                " VER ",
                " NOR ",
            ],
            "LapNumber": [
                1.0,
                1.0,
                2.0,
                2.0,
            ],
            "LapTime": [
                "00:01:38.100",
                "00:01:38.500",
                "00:01:37.900",
                "00:01:38.200",
            ],
            "Position": [
                1.0,
                2.0,
                1.0,
                2.0,
            ],
            "Stint": [
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            "Compound": [
                " soft ",
                "soft",
                " MEDIUM ",
                "medium",
            ],
            "TyreLife": [
                1,
                1,
                2,
                2,
            ],
            "TrackStatus": [
                " 1 ",
                "1",
                " 1",
                "1 ",
            ],
            "ExtraColumn": [
                "preserve",
                "this",
                "extra",
                "column",
            ],
        }
    )


def test_cleaning_does_not_modify_input_dataframe() -> None:
    """Cleaning must return a new DataFrame and preserve the raw input."""
    laps = make_valid_laps()
    original = laps.copy(deep=True)

    result = clean_lap_data(laps)

    pd.testing.assert_frame_equal(
        laps,
        original,
    )

    assert result.laps is not laps


def test_text_columns_are_trimmed_and_normalized() -> None:
    """Driver and compound values should be uppercase and whitespace-free."""
    result = clean_lap_data(make_valid_laps())

    cleaned = result.laps

    assert cleaned["Driver"].tolist() == [
        "VER",
        "NOR",
        "VER",
        "NOR",
    ]

    assert cleaned["Compound"].tolist() == [
        "SOFT",
        "SOFT",
        "MEDIUM",
        "MEDIUM",
    ]

    assert cleaned["TrackStatus"].tolist() == [
        "1",
        "1",
        "1",
        "1",
    ]


def test_numeric_columns_use_nullable_dtypes() -> None:
    """Integer-like race fields should use pandas nullable data types."""
    result = clean_lap_data(make_valid_laps())

    cleaned = result.laps

    assert str(cleaned["LapNumber"].dtype) == "Int64"
    assert str(cleaned["Position"].dtype) == "Int64"
    assert str(cleaned["Stint"].dtype) == "Int64"
    assert str(cleaned["TyreLife"].dtype) == "Float64"


def test_lap_times_are_converted_and_seconds_are_added() -> None:
    """Lap times should retain duration meaning and gain a numeric version."""
    result = clean_lap_data(make_valid_laps())

    cleaned = result.laps

    assert pd.api.types.is_timedelta64_dtype(cleaned["LapTime"].dtype)

    assert str(cleaned["LapTimeSeconds"].dtype) == "Float64"

    assert cleaned["LapTimeSeconds"].tolist() == pytest.approx(
        [
            98.1,
            98.5,
            97.9,
            98.2,
        ]
    )


def test_extra_columns_are_preserved() -> None:
    """Cleaning should not discard FastF1 columns it does not standardize."""
    result = clean_lap_data(make_valid_laps())

    assert "ExtraColumn" in result.laps.columns

    assert result.laps["ExtraColumn"].tolist() == [
        "preserve",
        "this",
        "extra",
        "column",
    ]


def test_missing_values_remain_missing() -> None:
    """Cleaning should not invent values for missing race information."""
    laps = make_valid_laps()

    laps["Compound"] = laps["Compound"].astype("object")
    laps["Position"] = laps["Position"].astype("object")
    laps["LapTime"] = laps["LapTime"].astype("object")

    laps.loc[0, "Compound"] = None
    laps.loc[1, "Position"] = None
    laps.loc[2, "LapTime"] = None

    result = clean_lap_data(laps)
    cleaned = result.laps

    assert pd.isna(cleaned.loc[0, "Compound"])
    assert pd.isna(cleaned.loc[1, "Position"])
    assert pd.isna(cleaned.loc[2, "LapTime"])
    assert pd.isna(cleaned.loc[2, "LapTimeSeconds"])


def test_blank_text_is_converted_to_missing_value() -> None:
    """Whitespace-only categorical values should become missing values."""
    laps = make_valid_laps()
    laps["Compound"] = laps["Compound"].astype("object")
    laps.loc[0, "Compound"] = "   "

    result = clean_lap_data(laps)

    assert pd.isna(result.laps.loc[0, "Compound"])

    assert result.output_validation.has_warnings is True


def test_warning_only_input_is_allowed() -> None:
    """Warnings should be recorded without preventing cleaning."""
    laps = make_valid_laps()
    laps["Compound"] = laps["Compound"].astype("object")
    laps.loc[0, "Compound"] = None

    result = clean_lap_data(laps)

    assert result.input_validation.has_warnings is True
    assert result.input_validation.has_errors is False

    assert result.output_validation.has_warnings is True
    assert result.output_validation.has_errors is False


def test_input_errors_stop_cleaning_in_strict_mode() -> None:
    """Error-level input findings should stop default cleaning."""
    laps = make_valid_laps()
    laps.loc[0, "TyreLife"] = -1

    with pytest.raises(
        RaceDataValidationError,
        match="negative_tyre_life",
    ):
        clean_lap_data(laps)


def test_fractional_integer_value_is_rejected_in_non_strict_mode() -> None:
    """Cleaning must not silently round fractional lap numbers."""
    laps = make_valid_laps()
    laps["LapNumber"] = laps["LapNumber"].astype("object")
    laps.loc[0, "LapNumber"] = 1.5

    with pytest.raises(
        RaceDataCleaningError,
        match="LapNumber contains fractional values",
    ):
        clean_lap_data(
            laps,
            fail_on_input_errors=False,
        )


def test_invalid_lap_time_is_rejected_in_non_strict_mode() -> None:
    """Unparseable non-missing lap times should not be silently removed."""
    laps = make_valid_laps()
    laps["LapTime"] = laps["LapTime"].astype("object")
    laps.loc[0, "LapTime"] = "not-a-lap-time"

    with pytest.raises(
        RaceDataCleaningError,
        match="LapTime contains values that cannot be converted",
    ):
        clean_lap_data(
            laps,
            fail_on_input_errors=False,
        )


def test_unresolved_input_error_fails_output_validation() -> None:
    """Non-strict cleaning must still reject invalid final data."""
    laps = make_valid_laps()
    laps.loc[0, "TyreLife"] = -1

    with pytest.raises(
        RaceDataCleaningError,
        match="Cleaning produced invalid lap data",
    ):
        clean_lap_data(
            laps,
            fail_on_input_errors=False,
        )


def test_select_standardized_columns_uses_deterministic_order() -> None:
    """Selected minimum columns should follow the documented schema order."""
    result = clean_lap_data(make_valid_laps())

    selected = select_standardized_lap_columns(result.laps)

    assert tuple(selected.columns) == STANDARDIZED_COLUMNS
    assert "ExtraColumn" not in selected.columns
    assert "LapTimeSeconds" not in selected.columns


def test_selected_standardized_table_is_a_copy() -> None:
    """Changing the selected table must not modify the cleaned source."""
    result = clean_lap_data(make_valid_laps())

    selected = select_standardized_lap_columns(result.laps)

    selected.loc[0, "Driver"] = "CHANGED"

    assert result.laps.loc[0, "Driver"] == "VER"


def test_missing_standardized_column_is_rejected() -> None:
    """Column selection should clearly identify missing requirements."""
    laps = make_valid_laps().drop(
        columns=[
            "TrackStatus",
            "TyreLife",
        ]
    )

    with pytest.raises(
        RaceDataCleaningError,
        match="Missing columns",
    ) as exception_info:
        select_standardized_lap_columns(laps)

    error_message = str(exception_info.value)

    assert "TrackStatus" in error_message
    assert "TyreLife" in error_message


def test_non_dataframe_input_is_rejected() -> None:
    """The cleaning API should reject non-DataFrame objects."""
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
        clean_lap_data(invalid_input)
