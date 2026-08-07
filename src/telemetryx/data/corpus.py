from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import pandas as pd

from telemetryx.data.dataset import (
    RACE_DATASET_COLUMNS,
    RACE_DATASET_KEY_COLUMNS,
    validate_race_dataset,
)
from telemetryx.data.targets import TARGET_COLUMN

CORPUS_KEY_COLUMNS: Final[tuple[str, ...]] = (*RACE_DATASET_KEY_COLUMNS,)

CORPUS_SORT_COLUMNS: Final[tuple[str, ...]] = (
    "Season",
    "RoundNumber",
    "SnapshotLap",
    "Position",
    "Driver",
)


class RaceCorpusError(ValueError):
    """Raised when multiple race datasets cannot form a valid corpus."""


@dataclass(frozen=True, slots=True)
class RaceCorpusSummary:
    """Summary information for a validated multi-race corpus."""

    race_count: int
    season_count: int
    row_count: int
    snapshot_count: int
    driver_count: int
    first_race_id: str
    last_race_id: str


def combine_race_datasets(
    datasets: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """
    Combine multiple validated race datasets into one chronological corpus.

    Every input dataset must use the same schema and pandas data types.
    Individual races are validated before concatenation, and the combined
    corpus is validated again afterward.

    Parameters
    ----------
    datasets:
        Ordered collection of race-level datasets.

    Returns
    -------
    pd.DataFrame
        Chronologically ordered multi-race corpus.

    Raises
    ------
    TypeError
        If an item is not a pandas DataFrame.
    RaceCorpusError
        If no datasets are supplied or their schemas conflict.
    """
    if len(datasets) == 0:
        raise RaceCorpusError(
            "At least one race dataset is required to build a corpus."
        )

    validated_datasets: list[pd.DataFrame] = []

    expected_dtypes: dict[str, str] | None = None
    seen_race_ids: set[str] = set()
    seen_season_rounds: dict[tuple[int, int], str] = {}

    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError(
                "Every corpus item must be provided as a pandas DataFrame. "
                f"Invalid item index: {index}."
            )

        try:
            validate_race_dataset(dataset)
        except (TypeError, ValueError) as exc:
            raise RaceCorpusError(
                f"Race dataset at index {index} failed validation."
            ) from exc

        _validate_exact_schema(
            dataset=dataset,
            dataset_index=index,
        )

        current_dtypes = _extract_dtype_signature(dataset)

        if expected_dtypes is None:
            expected_dtypes = current_dtypes
        else:
            _validate_dtype_signature(
                expected=expected_dtypes,
                actual=current_dtypes,
                dataset_index=index,
            )

        race_id = _extract_single_race_id(dataset)

        if race_id in seen_race_ids:
            raise RaceCorpusError(
                f"The corpus contains the same race more than once: {race_id}."
            )

        season, round_number = _extract_season_round(dataset)

        season_round_key = (
            season,
            round_number,
        )

        existing_race_id = seen_season_rounds.get(season_round_key)

        if existing_race_id is not None:
            raise RaceCorpusError(
                "Multiple race datasets use the same season and round: "
                f"{season} round {round_number} "
                f"({existing_race_id}, {race_id})."
            )

        seen_race_ids.add(race_id)

        seen_season_rounds[season_round_key] = race_id

        validated_datasets.append(dataset.copy(deep=True))

    corpus = pd.concat(
        validated_datasets,
        ignore_index=True,
    )

    corpus = corpus.sort_values(
        by=list(CORPUS_SORT_COLUMNS),
        ascending=True,
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    validate_race_corpus(corpus)

    return corpus


def validate_race_corpus(
    corpus: pd.DataFrame,
) -> None:
    """
    Validate structural invariants of a multi-race dataset corpus.

    Parameters
    ----------
    corpus:
        Combined race-level dataset table.

    Raises
    ------
    TypeError
        If ``corpus`` is not a pandas DataFrame.
    RaceCorpusError
        If corpus-level structural invariants are violated.
    """
    if not isinstance(corpus, pd.DataFrame):
        raise TypeError("corpus must be provided as a pandas DataFrame.")

    if corpus.empty:
        raise RaceCorpusError("The race corpus contains no rows.")

    duplicate_columns = _duplicate_column_names(corpus)

    if duplicate_columns:
        raise RaceCorpusError(
            "The race corpus contains duplicate column names: "
            f"{', '.join(duplicate_columns)}."
        )

    missing_columns = [
        column for column in RACE_DATASET_COLUMNS if column not in corpus.columns
    ]

    if missing_columns:
        raise RaceCorpusError(
            "The race corpus is missing required columns: "
            f"{', '.join(missing_columns)}."
        )

    duplicate_keys = corpus.duplicated(
        subset=list(CORPUS_KEY_COLUMNS),
        keep=False,
    )

    if bool(duplicate_keys.any()):
        raise RaceCorpusError(
            "The race corpus contains duplicate race-snapshot-driver rows."
        )

    race_ids = corpus["RaceId"].astype("string").dropna().unique().tolist()

    if not race_ids:
        raise RaceCorpusError("The race corpus contains no valid RaceId values.")

    _validate_races_individually(corpus)

    _validate_unique_season_rounds(corpus)

    _validate_global_snapshot_targets(corpus)


def summarize_race_corpus(
    corpus: pd.DataFrame,
) -> RaceCorpusSummary:
    """
    Return high-level dimensions of a validated race corpus.

    Parameters
    ----------
    corpus:
        Valid multi-race corpus.

    Returns
    -------
    RaceCorpusSummary
        Counts and chronological race boundaries.
    """
    validate_race_corpus(corpus)

    ordered_races = (
        corpus.loc[
            :,
            [
                "RaceId",
                "Season",
                "RoundNumber",
            ],
        ]
        .drop_duplicates()
        .sort_values(
            by=[
                "Season",
                "RoundNumber",
                "RaceId",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    race_ids = [str(value) for value in ordered_races["RaceId"].tolist()]

    snapshot_count = len(
        corpus.loc[
            :,
            [
                "RaceId",
                "SnapshotLap",
            ],
        ].drop_duplicates()
    )

    driver_count = int(corpus["Driver"].astype("string").nunique())

    season_count = int(corpus["Season"].nunique())

    return RaceCorpusSummary(
        race_count=len(race_ids),
        season_count=season_count,
        row_count=len(corpus),
        snapshot_count=snapshot_count,
        driver_count=driver_count,
        first_race_id=race_ids[0],
        last_race_id=race_ids[-1],
    )


def _validate_races_individually(
    corpus: pd.DataFrame,
) -> None:
    """Run the existing race validator independently for every race."""
    race_ids = corpus["RaceId"].astype("string").dropna().unique().tolist()

    for race_id_value in race_ids:
        race_id = str(race_id_value)

        race_dataset = corpus.loc[corpus["RaceId"].astype("string").eq(race_id)].copy()

        try:
            validate_race_dataset(race_dataset)
        except (TypeError, ValueError) as exc:
            raise RaceCorpusError(
                f"A race inside the corpus failed race-level validation: {race_id}."
            ) from exc


def _validate_unique_season_rounds(
    corpus: pd.DataFrame,
) -> None:
    """Ensure one race maps to each season-round combination."""
    race_metadata = corpus.loc[
        :,
        [
            "RaceId",
            "Season",
            "RoundNumber",
        ],
    ].drop_duplicates()

    duplicated_rounds = race_metadata.duplicated(
        subset=[
            "Season",
            "RoundNumber",
        ],
        keep=False,
    )

    if not bool(duplicated_rounds.any()):
        return

    affected = race_metadata.loc[duplicated_rounds]

    descriptions = [
        (f"{row.Season}/{row.RoundNumber}:{row.RaceId}")
        for row in affected.itertuples(index=False)
    ]

    raise RaceCorpusError(
        "The race corpus contains multiple races for the same "
        "season-round combination: "
        f"{', '.join(descriptions)}."
    )


def _validate_global_snapshot_targets(
    corpus: pd.DataFrame,
) -> None:
    """Ensure every race snapshot contains exactly one winner target."""
    winner_counts = corpus.groupby(
        [
            "RaceId",
            "SnapshotLap",
        ],
        sort=False,
        dropna=False,
    )[TARGET_COLUMN].sum()

    invalid_snapshots: list[str] = []

    for group_key, winner_count in winner_counts.items():
        if (
            not isinstance(
                group_key,
                tuple,
            )
            or len(group_key) != 2
        ):
            raise RaceCorpusError("Unexpected race-snapshot grouping key.")

        race_id, snapshot_lap = group_key

        if int(winner_count) != 1:
            invalid_snapshots.append(f"{race_id}/lap {snapshot_lap}")

    if invalid_snapshots:
        raise RaceCorpusError(
            "Every corpus snapshot must contain exactly one "
            "winner target. Invalid snapshots: "
            f"{', '.join(invalid_snapshots)}."
        )


def _validate_exact_schema(
    *,
    dataset: pd.DataFrame,
    dataset_index: int,
) -> None:
    """Ensure one race dataset exactly matches the current corpus schema."""
    actual_columns = tuple(str(column) for column in dataset.columns)

    if actual_columns == RACE_DATASET_COLUMNS:
        return

    raise RaceCorpusError(
        "Race dataset schema does not match the expected "
        f"race-dataset schema at index {dataset_index}."
    )


def _extract_dtype_signature(
    dataset: pd.DataFrame,
) -> dict[str, str]:
    """Return a stable column-to-dtype representation."""
    return {str(column): str(dtype) for column, dtype in dataset.dtypes.items()}


def _validate_dtype_signature(
    *,
    expected: dict[str, str],
    actual: dict[str, str],
    dataset_index: int,
) -> None:
    """Reject dtype changes that could cause unsafe concatenation."""
    mismatched_columns = [
        column
        for column, expected_dtype in expected.items()
        if actual.get(column) != expected_dtype
    ]

    if not mismatched_columns:
        return

    descriptions = [
        (
            f"{column}: expected "
            f"{expected[column]}, "
            f"found {actual.get(column, 'missing')}"
        )
        for column in mismatched_columns
    ]

    raise RaceCorpusError(
        "Race dataset dtypes do not match the corpus schema "
        f"at index {dataset_index}: "
        f"{'; '.join(descriptions)}."
    )


def _extract_single_race_id(
    dataset: pd.DataFrame,
) -> str:
    """Return the single RaceId represented by a race dataset."""
    values = dataset["RaceId"].astype("string").dropna().unique().tolist()

    if len(values) != 1:
        raise RaceCorpusError("Each race dataset must contain exactly one RaceId.")

    race_id = str(values[0]).strip()

    if not race_id:
        raise RaceCorpusError("RaceId cannot be blank.")

    return race_id


def _extract_season_round(
    dataset: pd.DataFrame,
) -> tuple[int, int]:
    """Return one race dataset's season and championship round."""
    season_raw = dataset["Season"].iloc[0]

    round_raw = dataset["RoundNumber"].iloc[0]

    try:
        season = int(str(season_raw))

        round_number = int(str(round_raw))
    except ValueError as exc:
        raise RaceCorpusError("Season and RoundNumber must be integer values.") from exc

    return (
        season,
        round_number,
    )


def _duplicate_column_names(
    frame: pd.DataFrame,
) -> tuple[str, ...]:
    """Return duplicate DataFrame column names."""
    duplicate_columns = frame.columns[frame.columns.duplicated()]

    return tuple(str(column) for column in duplicate_columns)
