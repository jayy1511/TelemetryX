"""Create leakage-safe chronological train, validation and test splits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np
import pandas as pd

from telemetryx.data.corpus import validate_race_corpus

SPLIT_COLUMN: Final[str] = "DatasetSplit"


class DatasetSplit(StrEnum):
    """Supported machine-learning dataset partitions."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class CorpusSplitError(ValueError):
    """Raised when a race corpus cannot be split chronologically."""


@dataclass(frozen=True, slots=True)
class RaceSplitAssignment:
    """Assignment of one complete race to one dataset split."""

    race_id: str
    season: int
    round_number: int
    split: DatasetSplit


@dataclass(frozen=True, slots=True)
class CorpusSplit:
    """Chronological train, validation and test corpus partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    assignments: tuple[RaceSplitAssignment, ...]

    @property
    def train_race_count(self) -> int:
        """Return the number of training races."""
        return _count_assignments(
            self.assignments,
            DatasetSplit.TRAIN,
        )

    @property
    def validation_race_count(self) -> int:
        """Return the number of validation races."""
        return _count_assignments(
            self.assignments,
            DatasetSplit.VALIDATION,
        )

    @property
    def test_race_count(self) -> int:
        """Return the number of test races."""
        return _count_assignments(
            self.assignments,
            DatasetSplit.TEST,
        )


def split_race_corpus(
    corpus: pd.DataFrame,
    *,
    validation_season: int,
    validation_last_races: int,
    test_seasons: Sequence[int],
) -> CorpusSplit:
    """
    Split a race corpus chronologically at complete-race boundaries.

    Training receives all races chronologically before the validation
    boundary. Validation receives the final ``validation_last_races`` races
    of ``validation_season``. Test receives every race in ``test_seasons``.

    Parameters
    ----------
    corpus:
        Valid TelemetryX multi-race corpus.
    validation_season:
        Season whose final races are reserved for validation.
    validation_last_races:
        Number of final races in ``validation_season`` reserved for
        validation.
    test_seasons:
        One or more complete seasons occurring after the validation season.

    Returns
    -------
    CorpusSplit
        Independent train, validation and test DataFrames plus race-level
        assignments.

    Raises
    ------
    TypeError
        If ``corpus`` is not a pandas DataFrame.
    CorpusSplitError
        If the requested split is invalid, incomplete or non-chronological.
    """
    if not isinstance(
        corpus,
        pd.DataFrame,
    ):
        raise TypeError("corpus must be provided as a pandas DataFrame.")

    try:
        validate_race_corpus(corpus)
    except (TypeError, ValueError) as exc:
        raise CorpusSplitError(
            "The race corpus failed validation before splitting."
        ) from exc

    _validate_positive_integer(
        value=validation_season,
        field_name="validation_season",
    )

    _validate_positive_integer(
        value=validation_last_races,
        field_name="validation_last_races",
    )

    normalized_test_seasons = _normalize_test_seasons(test_seasons)

    _validate_test_season_order(
        validation_season=validation_season,
        test_seasons=normalized_test_seasons,
    )

    race_metadata = _extract_race_metadata(corpus)

    available_seasons = {int(value) for value in race_metadata["Season"].tolist()}

    if validation_season not in available_seasons:
        raise CorpusSplitError(
            "The requested validation season is not present "
            f"in the corpus: {validation_season}."
        )

    missing_test_seasons = sorted(
        set(normalized_test_seasons).difference(available_seasons)
    )

    if missing_test_seasons:
        raise CorpusSplitError(
            "One or more requested test seasons are not present "
            "in the corpus: "
            f"{', '.join(str(value) for value in missing_test_seasons)}."
        )

    _validate_later_season_coverage(
        available_seasons=available_seasons,
        validation_season=validation_season,
        test_seasons=normalized_test_seasons,
    )

    validation_metadata = race_metadata.loc[
        race_metadata["Season"].eq(validation_season)
    ]

    if len(validation_metadata) < validation_last_races:
        raise CorpusSplitError(
            "validation_last_races exceeds the number of races "
            f"available in season {validation_season}: requested "
            f"{validation_last_races}, available "
            f"{len(validation_metadata)}."
        )

    validation_race_ids = set(
        str(value)
        for value in validation_metadata.tail(validation_last_races)["RaceId"].tolist()
    )

    test_race_ids = set(
        str(value)
        for value in race_metadata.loc[
            race_metadata["Season"].isin(normalized_test_seasons),
            "RaceId",
        ].tolist()
    )

    all_race_ids = set(str(value) for value in race_metadata["RaceId"].tolist())

    train_race_ids = all_race_ids - validation_race_ids - test_race_ids

    if not train_race_ids:
        raise CorpusSplitError("The requested split leaves no races for training.")

    if not validation_race_ids:
        raise CorpusSplitError("The requested split leaves no races for validation.")

    if not test_race_ids:
        raise CorpusSplitError("The requested split leaves no races for testing.")

    assignments = _build_assignments(
        race_metadata=race_metadata,
        train_race_ids=train_race_ids,
        validation_race_ids=validation_race_ids,
        test_race_ids=test_race_ids,
    )

    _validate_assignment_chronology(assignments)

    train = _select_races(
        corpus,
        train_race_ids,
    )

    validation = _select_races(
        corpus,
        validation_race_ids,
    )

    test = _select_races(
        corpus,
        test_race_ids,
    )

    _validate_split_frames(
        original=corpus,
        train=train,
        validation=validation,
        test=test,
    )

    return CorpusSplit(
        train=train,
        validation=validation,
        test=test,
        assignments=assignments,
    )


def add_split_labels(
    corpus: pd.DataFrame,
    split: CorpusSplit,
) -> pd.DataFrame:
    """
    Return a corpus copy containing a race-level split label column.

    This is useful for inspection and artifact export. The source corpus is
    never modified.

    Parameters
    ----------
    corpus:
        Original valid race corpus.
    split:
        Split assignment generated by ``split_race_corpus``.

    Returns
    -------
    pd.DataFrame
        Corpus copy with a ``DatasetSplit`` column.
    """
    if not isinstance(
        corpus,
        pd.DataFrame,
    ):
        raise TypeError("corpus must be provided as a pandas DataFrame.")

    if not isinstance(
        split,
        CorpusSplit,
    ):
        raise TypeError("split must be provided as a CorpusSplit.")

    try:
        validate_race_corpus(corpus)
    except (TypeError, ValueError) as exc:
        raise CorpusSplitError(
            "The race corpus failed validation before labeling."
        ) from exc

    if SPLIT_COLUMN in corpus.columns:
        raise CorpusSplitError(
            f"The corpus already contains the reserved {SPLIT_COLUMN} column."
        )

    assignment_map = {
        assignment.race_id: assignment.split.value for assignment in split.assignments
    }

    labeled = corpus.copy(deep=True)

    race_ids = labeled["RaceId"].astype("string")

    labels = race_ids.map(assignment_map)

    if bool(labels.isna().any()):
        missing_race_ids = sorted(
            {str(value) for value in race_ids.loc[labels.isna()].tolist()}
        )

        raise CorpusSplitError(
            "Split assignments are missing one or more corpus races: "
            f"{', '.join(missing_race_ids)}."
        )

    labeled[SPLIT_COLUMN] = labels.astype("string")

    return labeled


def _extract_race_metadata(
    corpus: pd.DataFrame,
) -> pd.DataFrame:
    """Return one chronologically ordered metadata row per race."""
    metadata = (
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

    return metadata


def _normalize_test_seasons(
    test_seasons: Sequence[int],
) -> tuple[int, ...]:
    """Validate and normalize requested test seasons."""
    if isinstance(
        test_seasons,
        (str, bytes),
    ):
        raise CorpusSplitError("test_seasons must be a sequence of positive integers.")

    normalized: list[int] = []

    for season in test_seasons:
        _validate_positive_integer(
            value=season,
            field_name="test season",
        )

        normalized.append(season)

    if not normalized:
        raise CorpusSplitError("At least one test season is required.")

    if len(set(normalized)) != len(normalized):
        raise CorpusSplitError("test_seasons cannot contain duplicate values.")

    return tuple(sorted(normalized))


def _validate_test_season_order(
    *,
    validation_season: int,
    test_seasons: tuple[int, ...],
) -> None:
    """Require every test season to occur after validation."""
    invalid_seasons = [season for season in test_seasons if season <= validation_season]

    if invalid_seasons:
        raise CorpusSplitError(
            "Every test season must occur after the validation season. "
            "Invalid test seasons: "
            f"{', '.join(str(value) for value in invalid_seasons)}."
        )


def _validate_later_season_coverage(
    *,
    available_seasons: set[int],
    validation_season: int,
    test_seasons: tuple[int, ...],
) -> None:
    """
    Prevent later seasons from accidentally entering the training split.

    Every season after validation must explicitly belong to the test set.
    """
    later_seasons = {
        season for season in available_seasons if season > validation_season
    }

    uncovered = sorted(later_seasons.difference(test_seasons))

    if uncovered:
        raise CorpusSplitError(
            "Every corpus season after the validation season must be "
            "explicitly assigned to test. Uncovered seasons: "
            f"{', '.join(str(value) for value in uncovered)}."
        )


def _build_assignments(
    *,
    race_metadata: pd.DataFrame,
    train_race_ids: set[str],
    validation_race_ids: set[str],
    test_race_ids: set[str],
) -> tuple[RaceSplitAssignment, ...]:
    """Build deterministic race-level split assignments."""
    assignments: list[RaceSplitAssignment] = []

    for row in race_metadata.itertuples(index=False):
        race_id = str(row.RaceId)

        season = _coerce_required_integer(
            row.Season,
            field_name="Season",
        )

        round_number = _coerce_required_integer(
            row.RoundNumber,
            field_name="RoundNumber",
        )

        if race_id in train_race_ids:
            split = DatasetSplit.TRAIN

        elif race_id in validation_race_ids:
            split = DatasetSplit.VALIDATION

        elif race_id in test_race_ids:
            split = DatasetSplit.TEST

        else:
            raise CorpusSplitError(
                f"A race was not assigned to any dataset split: {race_id}."
            )

        assignments.append(
            RaceSplitAssignment(
                race_id=race_id,
                season=season,
                round_number=round_number,
                split=split,
            )
        )

    return tuple(assignments)


def _validate_assignment_chronology(
    assignments: tuple[RaceSplitAssignment, ...],
) -> None:
    """Ensure train precedes validation and validation precedes test."""
    if not assignments:
        raise CorpusSplitError("Split assignments cannot be empty.")

    split_order = {
        DatasetSplit.TRAIN: 0,
        DatasetSplit.VALIDATION: 1,
        DatasetSplit.TEST: 2,
    }

    previous_order = -1

    for assignment in assignments:
        current_order = split_order[assignment.split]

        if current_order < previous_order:
            raise CorpusSplitError("Race split assignments are not chronological.")

        previous_order = current_order


def _select_races(
    corpus: pd.DataFrame,
    race_ids: set[str],
) -> pd.DataFrame:
    """Return an independent corpus subset containing complete races."""
    selected = corpus.loc[corpus["RaceId"].astype("string").isin(race_ids)].copy(
        deep=True
    )

    return selected.reset_index(drop=True)


def _validate_split_frames(
    *,
    original: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Validate split independence, completeness and race boundaries."""
    for name, frame in (
        (
            "train",
            train,
        ),
        (
            "validation",
            validation,
        ),
        (
            "test",
            test,
        ),
    ):
        if frame.empty:
            raise CorpusSplitError(f"The {name} split contains no rows.")

        try:
            validate_race_corpus(frame)
        except (TypeError, ValueError) as exc:
            raise CorpusSplitError(
                f"The {name} split failed corpus validation."
            ) from exc

    train_races = _race_id_set(train)

    validation_races = _race_id_set(validation)

    test_races = _race_id_set(test)

    if train_races.intersection(validation_races):
        raise CorpusSplitError("A race appears in both training and validation.")

    if train_races.intersection(test_races):
        raise CorpusSplitError("A race appears in both training and test.")

    if validation_races.intersection(test_races):
        raise CorpusSplitError("A race appears in both validation and test.")

    original_races = _race_id_set(original)

    assigned_races = train_races | validation_races | test_races

    if assigned_races != original_races:
        raise CorpusSplitError(
            "The split does not assign every corpus race exactly once."
        )

    total_rows = len(train) + len(validation) + len(test)

    if total_rows != len(original):
        raise CorpusSplitError(
            "The split does not preserve the complete corpus row count."
        )


def _race_id_set(
    frame: pd.DataFrame,
) -> set[str]:
    """Return normalized RaceId values represented by a DataFrame."""
    return {str(value) for value in frame["RaceId"].astype("string").dropna().tolist()}


def _count_assignments(
    assignments: tuple[RaceSplitAssignment, ...],
    split: DatasetSplit,
) -> int:
    """Count race assignments belonging to one split."""
    return sum(assignment.split == split for assignment in assignments)


def _coerce_required_integer(
    value: object,
    *,
    field_name: str,
) -> int:
    """Convert a validated pandas scalar into a required integer."""
    if value is None:
        raise CorpusSplitError(f"{field_name} cannot be missing.")

    if value is pd.NA or value is pd.NaT:
        raise CorpusSplitError(f"{field_name} cannot be missing.")

    if isinstance(value, bool):
        raise CorpusSplitError(f"{field_name} must be an integer.")

    if isinstance(
        value,
        (int, np.integer),
    ):
        return int(str(value))

    raise CorpusSplitError(f"{field_name} must be an integer.")


def _validate_positive_integer(
    *,
    value: int,
    field_name: str,
) -> None:
    """Require one strictly positive Python integer."""
    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            int,
        )
        or value <= 0
    ):
        raise CorpusSplitError(f"{field_name} must be a positive integer.")
