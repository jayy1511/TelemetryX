from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from telemetryx.config import Settings, load_settings
from telemetryx.data.inspect_race import (
    RaceInventoryError,
    build_race_inventory,
    dataframe_inventory,
    default_inventory_path,
    parse_event_identifier,
    print_inventory_summary,
    save_race_inventory,
    slugify,
    validate_lap_table,
)


class FakeSession:
    """Small test double containing FastF1-like session attributes."""

    def __init__(self) -> None:
        """Create a valid fake race session."""
        self.laps: object = make_valid_laps()

        self.results: object = pd.DataFrame(
            {
                "Abbreviation": ["VER", "NOR"],
                "Position": [1.0, 2.0],
                "Status": ["Finished", "Finished"],
            }
        )

        self.weather_data: object = pd.DataFrame(
            {
                "Time": pd.to_timedelta(
                    [
                        "00:00:00",
                        "00:01:00",
                    ]
                ),
                "Rainfall": [False, False],
                "TrackTemp": [31.0, 31.2],
            }
        )

        self.race_control_messages: object = pd.DataFrame(
            {
                "Time": pd.to_timedelta(
                    [
                        "00:00:10",
                    ]
                ),
                "Category": ["Flag"],
                "Message": ["GREEN LIGHT"],
            }
        )

        self.event: dict[str, object] = {
            "EventName": "Bahrain Grand Prix",
            "OfficialEventName": ("FORMULA 1 GULF AIR BAHRAIN GRAND PRIX 2024"),
            "EventDate": pd.Timestamp("2024-03-02"),
            "RoundNumber": 1,
        }

        self.name = "Race"
        self.total_laps = 57


def make_valid_laps() -> pd.DataFrame:
    """Return a minimal valid FastF1-like lap table."""
    return pd.DataFrame(
        {
            "Driver": ["VER", "NOR"],
            "LapNumber": [1.0, 1.0],
            "LapTime": pd.to_timedelta(
                [
                    "00:01:38.200",
                    "00:01:39.000",
                ]
            ),
            "Position": [1.0, 2.0],
            "Stint": [1.0, 1.0],
            "Compound": ["SOFT", "SOFT"],
            "TyreLife": [1.0, 1.0],
            "TrackStatus": ["1", "1"],
        }
    )


def settings_with_temporary_interim_directory(
    tmp_path: Path,
) -> Settings:
    """Return settings using a temporary interim-data directory."""
    settings = load_settings()

    temporary_paths = replace(
        settings.paths,
        interim_data_dir=tmp_path,
    )

    return replace(
        settings,
        paths=temporary_paths,
    )


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("Bahrain", "Bahrain"),
        ("  Monza  ", "Monza"),
        ("1", 1),
        ("  7  ", 7),
    ],
)
def test_parse_event_identifier_accepts_names_and_rounds(
    raw_value: str,
    expected: str | int,
) -> None:
    """Event arguments should support names and positive round numbers."""
    assert parse_event_identifier(raw_value) == expected


@pytest.mark.parametrize(
    "raw_value",
    [
        "",
        "   ",
        "0",
        "-1",
    ],
)
def test_parse_event_identifier_rejects_invalid_values(
    raw_value: str,
) -> None:
    """Empty names and non-positive rounds should be rejected."""
    with pytest.raises(argparse.ArgumentTypeError):
        parse_event_identifier(raw_value)


def test_dataframe_inventory_handles_unavailable_table() -> None:
    """A missing optional table should be represented explicitly."""
    inventory = dataframe_inventory(None)

    assert inventory == {
        "available": False,
        "rows": 0,
        "columns": [],
        "dtypes": {},
        "missing_values": {},
    }


def test_dataframe_inventory_describes_dataframe() -> None:
    """A DataFrame inventory should include its structure and missing data."""
    frame = pd.DataFrame(
        {
            "Driver": ["VER", None],
            "LapNumber": [1, 2],
        }
    )

    inventory = dataframe_inventory(frame)

    assert inventory["available"] is True
    assert inventory["rows"] == 2
    assert inventory["columns"] == [
        "Driver",
        "LapNumber",
    ]

    assert inventory["missing_values"] == {
        "Driver": 1,
        "LapNumber": 0,
    }

    assert "Driver" in inventory["dtypes"]
    assert "LapNumber" in inventory["dtypes"]


def test_validate_lap_table_accepts_required_columns() -> None:
    """A non-empty lap table with all required columns should be valid."""
    validate_lap_table(make_valid_laps())


def test_validate_lap_table_rejects_empty_dataframe() -> None:
    """An empty lap table cannot be used for race analysis."""
    empty_laps = make_valid_laps().iloc[0:0]

    with pytest.raises(
        RaceInventoryError,
        match="lap table is empty",
    ):
        validate_lap_table(empty_laps)


def test_validate_lap_table_reports_missing_columns() -> None:
    """The validation error should identify absent required columns."""
    laps = make_valid_laps().drop(
        columns=[
            "TyreLife",
            "TrackStatus",
        ]
    )

    with pytest.raises(
        RaceInventoryError,
        match="missing required columns",
    ) as exception_info:
        validate_lap_table(laps)

    error_message = str(exception_info.value)

    assert "TyreLife" in error_message
    assert "TrackStatus" in error_message


def test_build_race_inventory_extracts_metadata_and_tables() -> None:
    """A valid session should produce complete race metadata."""
    session = FakeSession()

    inventory = build_race_inventory(session)  # type: ignore[arg-type]

    race = inventory["race"]
    tables = inventory["tables"]

    assert race == {
        "season": 2024,
        "round": 1,
        "event_name": "Bahrain Grand Prix",
        "official_event_name": ("FORMULA 1 GULF AIR BAHRAIN GRAND PRIX 2024"),
        "event_date": "2024-03-02T00:00:00",
        "session_name": "Race",
        "scheduled_total_laps": 57,
    }

    assert tables["laps"]["available"] is True
    assert tables["laps"]["rows"] == 2

    assert tables["results"]["available"] is True
    assert tables["results"]["rows"] == 2

    assert tables["weather"]["available"] is True
    assert tables["weather"]["rows"] == 2

    assert tables["race_control_messages"]["available"] is True
    assert tables["race_control_messages"]["rows"] == 1


def test_build_race_inventory_allows_missing_optional_tables() -> None:
    """Results, weather, and race-control tables may be unavailable."""
    session = FakeSession()
    session.results = None
    session.weather_data = None
    session.race_control_messages = None

    inventory = build_race_inventory(session)  # type: ignore[arg-type]
    tables = inventory["tables"]

    assert tables["results"]["available"] is False
    assert tables["weather"]["available"] is False
    assert tables["race_control_messages"]["available"] is False


def test_build_race_inventory_rejects_invalid_table_type() -> None:
    """Unexpected FastF1 table types should not be silently accepted."""
    session = FakeSession()
    session.results = ["not", "a", "dataframe"]

    with pytest.raises(
        RaceInventoryError,
        match="loaded results object is not a DataFrame",
    ):
        build_race_inventory(session)  # type: ignore[arg-type]


def test_build_race_inventory_handles_missing_event_metadata() -> None:
    """Missing event metadata should remain missing instead of being invented."""
    session = FakeSession()

    session.event = {
        "EventName": pd.NA,
        "OfficialEventName": None,
        "EventDate": pd.NaT,
        "RoundNumber": pd.NA,
    }

    session.total_laps = None

    inventory = build_race_inventory(session)  # type: ignore[arg-type]
    race = inventory["race"]

    assert race["season"] is None
    assert race["round"] is None
    assert race["event_name"] == "unknown_event"
    assert race["official_event_name"] is None
    assert race["event_date"] is None
    assert race["scheduled_total_laps"] is None


def test_save_race_inventory_writes_formatted_json(
    tmp_path: Path,
) -> None:
    """Inventory output should be created as readable JSON."""
    inventory: dict[str, Any] = {
        "race": {
            "season": 2024,
            "event_name": "Bahrain Grand Prix",
        },
        "tables": {
            "laps": {
                "rows": 2,
            }
        },
    }

    output_path = tmp_path / "nested" / "race_inventory.json"

    saved_path = save_race_inventory(
        inventory=inventory,
        output_path=output_path,
    )

    assert saved_path == output_path.resolve()
    assert saved_path.is_file()

    loaded_inventory = json.loads(saved_path.read_text(encoding="utf-8"))

    assert loaded_inventory == inventory


def test_default_inventory_path_uses_safe_filename(
    tmp_path: Path,
) -> None:
    """Default inventory paths should be deterministic and filename-safe."""
    settings = settings_with_temporary_interim_directory(tmp_path)

    inventory: dict[str, Any] = {
        "race": {
            "season": 2024,
            "event_name": "Bahrain Grand Prix",
            "session_name": "Race",
        }
    }

    result = default_inventory_path(
        settings=settings,
        inventory=inventory,
    )

    assert result == (
        tmp_path / "inventories" / "2024_bahrain_grand_prix_race_inventory.json"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Bahrain Grand Prix", "bahrain_grand_prix"),
        ("  British GP  ", "british_gp"),
        ("São Paulo", "são_paulo"),
        ("Race / Session", "race_session"),
        ("!!!", "unknown"),
    ],
)
def test_slugify_creates_safe_identifiers(
    value: str,
    expected: str,
) -> None:
    """Slug creation should be predictable for inventory filenames."""
    assert slugify(value) == expected


def test_print_inventory_summary_displays_key_information(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The terminal summary should show the race and table dimensions."""
    session = FakeSession()
    inventory = build_race_inventory(session)  # type: ignore[arg-type]

    output_path = tmp_path / "inventory.json"

    print_inventory_summary(
        inventory=inventory,
        output_path=output_path,
    )

    captured_output = capsys.readouterr().out

    assert "TelemetryX race inventory" in captured_output
    assert "2024 Bahrain Grand Prix" in captured_output
    assert "Scheduled laps: 57" in captured_output
    assert "laps: available=True, rows=2" in captured_output
    assert str(output_path) in captured_output
