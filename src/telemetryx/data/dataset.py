from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

from telemetryx.data.replay import (
    REPLAY_COLUMNS,
    build_race_replay,
)
from telemetryx.data.targets import (
    TARGET_COLUMN,
    attach_winner_targets,
    build_winner_targets,
)

RACE_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "RaceId",
    "Season",
    "RoundNumber",
    "EventName",
    "SessionName",
)

RACE_DATASET_COLUMNS: Final[tuple[str, ...]] = (
    *RACE_METADATA_COLUMNS,
    *REPLAY_COLUMNS,
    TARGET_COLUMN,
)

RACE_DATASET_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "RaceId",
    "SnapshotLap",
    "Driver",
)

DISALLOWED_POST_RACE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "FinalPosition",
        "ClassifiedPosition",
        "Points",
        "Status",
        "GridPosition",
    }
)


class RaceDatasetError(ValueError):
    """Raised when a valid race-level dataset cannot be assembled."""


@dataclass(frozen=True, slots=True)
class RaceMetadata:
    """Identity information attached to every row of one race dataset."""

    season: int
    round_number: int
    event_name: str
    session_name: str = "Race"

    def __post_init__(self) -> None:
        """Validate and normalize race identity values."""
        _validate_positive_integer(
            value=self.season,
            field_name="season",
        )

        _validate_positive_integer(
            value=self.round_number,
            field_name="round_number",
        )

        normalized_event_name = _normalize_required_text(
            value=self.event_name,
            field_name="event_name",
        )

        normalized_session_name = _normalize_required_text(
            value=self.session_name,
            field_name="session_name",
        )

        object.__setattr__(
            self,
            "event_name",
            normalized_event_name,
        )

        object.__setattr__(
            self,
            "session_name",
            normalized_session_name,
        )

    @property
    def race_id(self) -> str:
        """Return a deterministic identifier for the race."""
        event_slug = _slugify_identifier(self.event_name)

        return f"{self.season}_{self.round_number:02d}_{event_slug}"


def build_race_dataset(
    laps: pd.DataFrame,
    results: pd.DataFrame,
    *,
    season: int,
    round_number: int,
    event_name: str,
    session_name: str = "Race",
    start_lap: int = 1,
    end_lap: int | None = None,
) -> pd.DataFrame:
    """
    Assemble one temporally safe supervised-learning race dataset.

    The returned table contains replay features, race identity metadata and
    the eventual binary race-winner target.

    Final result columns such as finishing position, status and points are
    deliberately excluded.

    Parameters
    ----------
    laps:
        Cleaned lap data containing the columns required by the replay
        builder.
    results:
        Final FastF1-like race results used only to derive ``WonRace``.
    season:
        Championship season.
    round_number:
        Positive championship round number.
    event_name:
        Human-readable race name.
    session_name:
        Session name, normally ``Race``.
    start_lap:
        First completed leader lap to include.
    end_lap:
        Optional final completed leader lap to include.

    Returns
    -------
    pd.DataFrame
        One row per race, snapshot lap and driver.

    Raises
    ------
    TypeError
        If laps or results is not a pandas DataFrame.
    RaceDatasetError
        If metadata or the assembled dataset violates its schema.
    """
    if not isinstance(laps, pd.DataFrame):
        raise TypeError("laps must be provided as a pandas DataFrame.")

    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be provided as a pandas DataFrame.")

    metadata = RaceMetadata(
        season=season,
        round_number=round_number,
        event_name=event_name,
        session_name=session_name,
    )

    replay = build_race_replay(
        laps,
        start_lap=start_lap,
        end_lap=end_lap,
    )

    targets = build_winner_targets(results)

    dataset = attach_winner_targets(
        replay,
        targets,
    )

    dataset.insert(
        0,
        "SessionName",
        metadata.session_name,
    )

    dataset.insert(
        0,
        "EventName",
        metadata.event_name,
    )

    dataset.insert(
        0,
        "RoundNumber",
        metadata.round_number,
    )

    dataset.insert(
        0,
        "Season",
        metadata.season,
    )

    dataset.insert(
        0,
        "RaceId",
        metadata.race_id,
    )

    dataset = dataset.loc[
        :,
        list(RACE_DATASET_COLUMNS),
    ].copy()

    validate_race_dataset(dataset)

    return dataset


def validate_race_dataset(
    dataset: pd.DataFrame,
) -> None:
    """
    Validate structural and temporal invariants of one race dataset.

    This function returns normally when the dataset is valid and raises
    ``RaceDatasetError`` when an invariant is violated.

    Parameters
    ----------
    dataset:
        Race-level dataset produced by :func:`build_race_dataset`.

    Raises
    ------
    TypeError
        If ``dataset`` is not a pandas DataFrame.
    RaceDatasetError
        If the dataset is empty, malformed or temporally inconsistent.
    """
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("dataset must be provided as a pandas DataFrame.")

    if dataset.empty:
        raise RaceDatasetError("The race dataset contains no rows.")

    duplicate_columns = _duplicate_column_names(dataset)

    if duplicate_columns:
        raise RaceDatasetError(
            "The race dataset contains duplicate column names: "
            f"{', '.join(duplicate_columns)}."
        )

    _require_columns(
        frame=dataset,
        required_columns=RACE_DATASET_COLUMNS,
    )

    disallowed_columns = sorted(
        DISALLOWED_POST_RACE_COLUMNS.intersection(
            {str(column) for column in dataset.columns}
        )
    )

    if disallowed_columns:
        raise RaceDatasetError(
            "The race dataset contains prohibited post-race columns: "
            f"{', '.join(disallowed_columns)}."
        )

    duplicate_keys = dataset.duplicated(
        subset=list(RACE_DATASET_KEY_COLUMNS),
        keep=False,
    )

    if bool(duplicate_keys.any()):
        affected_indices = _format_affected_indices(
            dataset,
            duplicate_keys,
        )

        raise RaceDatasetError(
            "The race dataset contains duplicate race-snapshot-driver "
            f"rows at indices: {affected_indices}."
        )

    _validate_single_race_metadata(dataset)
    _validate_temporal_columns(dataset)
    _validate_target_column(dataset)
    _validate_snapshot_winner_counts(dataset)


def _validate_single_race_metadata(
    dataset: pd.DataFrame,
) -> None:
    """Ensure metadata is complete and constant across one race dataset."""
    for column in RACE_METADATA_COLUMNS:
        values = dataset[column]

        if bool(values.isna().any()):
            raise RaceDatasetError(f"{column} contains missing race metadata.")

        unique_count = int(values.nunique(dropna=False))

        if unique_count != 1:
            raise RaceDatasetError(
                f"{column} must contain exactly one value within a race dataset."
            )

    event_name = str(dataset["EventName"].iloc[0]).strip()

    session_name = str(dataset["SessionName"].iloc[0]).strip()

    race_id = str(dataset["RaceId"].iloc[0]).strip()

    if not event_name:
        raise RaceDatasetError("EventName cannot be blank.")

    if not session_name:
        raise RaceDatasetError("SessionName cannot be blank.")

    if not race_id:
        raise RaceDatasetError("RaceId cannot be blank.")


def _validate_temporal_columns(
    dataset: pd.DataFrame,
) -> None:
    """Ensure replay rows respect their snapshot cutoffs."""
    future_completed_laps = (
        dataset["CompletedLaps"].gt(dataset["SnapshotLap"]).fillna(False)
    )

    if bool(future_completed_laps.any()):
        raise RaceDatasetError(
            "The race dataset contains completed laps beyond their snapshot cutoff."
        )

    invalid_availability = (
        dataset["DataAvailableThroughLap"].ne(dataset["SnapshotLap"]).fillna(True)
    )

    if bool(invalid_availability.any()):
        raise RaceDatasetError(
            "DataAvailableThroughLap must equal SnapshotLap for every dataset row."
        )

    expected_lap_deficit = dataset["SnapshotLap"] - dataset["CompletedLaps"]

    invalid_lap_deficit = (
        dataset["LapsBehindLeader"].ne(expected_lap_deficit).fillna(True)
    )

    if bool(invalid_lap_deficit.any()):
        raise RaceDatasetError(
            "LapsBehindLeader is inconsistent with SnapshotLap and CompletedLaps."
        )


def _validate_target_column(
    dataset: pd.DataFrame,
) -> None:
    """Ensure the supervised target is complete and Boolean."""
    target = dataset[TARGET_COLUMN]

    if bool(target.isna().any()):
        raise RaceDatasetError(f"{TARGET_COLUMN} contains missing values.")

    if not pd.api.types.is_bool_dtype(target.dtype):
        raise RaceDatasetError(f"{TARGET_COLUMN} must use a Boolean dtype.")


def _validate_snapshot_winner_counts(
    dataset: pd.DataFrame,
) -> None:
    """Ensure every race snapshot contains exactly one positive target."""
    winner_counts = dataset.groupby(
        [
            "RaceId",
            "SnapshotLap",
        ],
        sort=False,
        dropna=False,
    )[TARGET_COLUMN].sum()

    invalid_snapshots: list[str] = []

    for group_key, winner_count in winner_counts.items():
        if not isinstance(group_key, tuple) or len(group_key) != 2:
            raise RaceDatasetError("Unexpected race-snapshot grouping key.")

        race_id, snapshot_lap = group_key

        if int(winner_count) != 1:
            invalid_snapshots.append(f"{race_id}/lap {snapshot_lap}")

    if invalid_snapshots:
        raise RaceDatasetError(
            "Every race snapshot must contain exactly one winner target. "
            "Invalid snapshots: "
            f"{', '.join(invalid_snapshots)}."
        )


def _require_columns(
    *,
    frame: pd.DataFrame,
    required_columns: tuple[str, ...],
) -> None:
    """Raise when required race-dataset columns are missing."""
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]

    if not missing_columns:
        return

    raise RaceDatasetError(
        f"The race dataset is missing required columns: {', '.join(missing_columns)}."
    )


def _duplicate_column_names(
    frame: pd.DataFrame,
) -> tuple[str, ...]:
    """Return duplicated DataFrame column names."""
    duplicate_columns = frame.columns[frame.columns.duplicated()]

    return tuple(str(column) for column in duplicate_columns)


def _validate_positive_integer(
    *,
    value: int,
    field_name: str,
) -> None:
    """Validate a positive integer metadata field."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RaceDatasetError(f"{field_name} must be a positive integer.")


def _normalize_required_text(
    *,
    value: str,
    field_name: str,
) -> str:
    """Normalize a required non-empty metadata string."""
    if not isinstance(value, str):
        raise RaceDatasetError(f"{field_name} must be a string.")

    normalized = value.strip()

    if not normalized:
        raise RaceDatasetError(f"{field_name} cannot be blank.")

    return normalized


def _slugify_identifier(
    value: str,
) -> str:
    """Convert text into a deterministic race-identifier component."""
    normalized_characters = [
        character.lower() if character.isalnum() else "_" for character in value.strip()
    ]

    parts = [part for part in "".join(normalized_characters).split("_") if part]

    return "_".join(parts) or "unknown_event"


def _format_affected_indices(
    frame: pd.DataFrame,
    mask: pd.Series[bool],
    *,
    limit: int = 5,
) -> str:
    """Return a compact list of affected DataFrame indices."""
    indices = frame.index[mask].tolist()[:limit]

    return ", ".join(str(index) for index in indices) or "unknown"
