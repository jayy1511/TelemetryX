"""Tests for leakage-safe chronological race-corpus splitting."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from telemetryx.data.corpus import combine_race_datasets
from telemetryx.data.dataset import build_race_dataset
from telemetryx.data.split import (
    SPLIT_COLUMN,
    CorpusSplit,
    CorpusSplitError,
    DatasetSplit,
    add_split_labels,
    split_race_corpus,
)


def make_cleaned_laps() -> pd.DataFrame:
    """Return cleaned lap data for a small three-driver race."""
    return pd.DataFrame(
        {
            "Driver": [
                "VER",
                "VER",
                "VER",
                "NOR",
                "NOR",
                "NOR",
                "BOT",
                "BOT",
            ],
            "LapNumber": [
                1,
                2,
                3,
                1,
                2,
                3,
                1,
                2,
            ],
            "Position": [
                1,
                1,
                1,
                2,
                2,
                2,
                3,
                3,
            ],
            "Stint": [
                1,
                1,
                2,
                1,
                1,
                1,
                1,
                1,
            ],
            "Compound": [
                "SOFT",
                "SOFT",
                "MEDIUM",
                "SOFT",
                "SOFT",
                "SOFT",
                "SOFT",
                "SOFT",
            ],
            "TyreLife": [
                1.0,
                2.0,
                1.0,
                1.0,
                2.0,
                3.0,
                1.0,
                2.0,
            ],
            "TrackStatus": [
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
            ],
            "LapTimeSeconds": [
                90.0,
                89.0,
                88.0,
                91.0,
                90.0,
                89.0,
                92.0,
                91.0,
            ],
        }
    )


def make_results(
    *,
    winner: str = "VER",
) -> pd.DataFrame:
    """Return final results containing exactly one race winner."""
    drivers = [
        "VER",
        "NOR",
        "BOT",
    ]

    ordered_drivers = [
        winner,
        *[driver for driver in drivers if driver != winner],
    ]

    return pd.DataFrame(
        {
            "Abbreviation": ordered_drivers,
            "Position": [
                1,
                2,
                3,
            ],
            "Status": [
                "Finished",
                "Finished",
                "+1 Lap",
            ],
        }
    )


def make_race_dataset(
    *,
    season: int,
    round_number: int,
    event_name: str,
    winner: str = "VER",
) -> pd.DataFrame:
    """Return one valid race-level dataset."""
    return build_race_dataset(
        make_cleaned_laps(),
        make_results(
            winner=winner,
        ),
        season=season,
        round_number=round_number,
        event_name=event_name,
        session_name="Race",
    )


def make_split_corpus() -> pd.DataFrame:
    """
    Return a corpus designed for chronological split testing.

    Structure:

    2022 rounds 1-2 -> training
    2023 rounds 1-2 -> training
    2023 rounds 3-4 -> validation
    2024 rounds 1-2 -> test
    """
    races = [
        make_race_dataset(
            season=2024,
            round_number=2,
            event_name="Saudi Arabian Grand Prix",
        ),
        make_race_dataset(
            season=2023,
            round_number=4,
            event_name="Azerbaijan Grand Prix",
            winner="NOR",
        ),
        make_race_dataset(
            season=2022,
            round_number=1,
            event_name="Bahrain Grand Prix",
        ),
        make_race_dataset(
            season=2023,
            round_number=1,
            event_name="Bahrain Grand Prix",
        ),
        make_race_dataset(
            season=2024,
            round_number=1,
            event_name="Bahrain Grand Prix",
            winner="NOR",
        ),
        make_race_dataset(
            season=2022,
            round_number=2,
            event_name="Saudi Arabian Grand Prix",
        ),
        make_race_dataset(
            season=2023,
            round_number=3,
            event_name="Australian Grand Prix",
        ),
        make_race_dataset(
            season=2023,
            round_number=2,
            event_name="Saudi Arabian Grand Prix",
            winner="NOR",
        ),
    ]

    return combine_race_datasets(races)


def make_standard_split(
    corpus: pd.DataFrame | None = None,
) -> CorpusSplit:
    """Return the standard chronological split used by these tests."""
    active_corpus = make_split_corpus() if corpus is None else corpus

    return split_race_corpus(
        active_corpus,
        validation_season=2023,
        validation_last_races=2,
        test_seasons=(2024,),
    )


def race_ids(
    frame: pd.DataFrame,
) -> set[str]:
    """Return unique race identifiers represented by a DataFrame."""
    return {str(value) for value in frame["RaceId"].unique().tolist()}


def test_split_assigns_expected_race_counts() -> None:
    """The configured split should allocate complete races as expected."""
    split = make_standard_split()

    assert split.train_race_count == 4
    assert split.validation_race_count == 2
    assert split.test_race_count == 2

    assert len(split.assignments) == 8


def test_training_contains_only_pre_validation_boundary_races() -> None:
    """Training should contain earlier seasons and early validation races."""
    split = make_standard_split()

    assert race_ids(split.train) == {
        "2022_01_bahrain_grand_prix",
        "2022_02_saudi_arabian_grand_prix",
        "2023_01_bahrain_grand_prix",
        "2023_02_saudi_arabian_grand_prix",
    }


def test_validation_uses_final_races_of_validation_season() -> None:
    """Validation must use exactly the final N races of its season."""
    split = make_standard_split()

    assert race_ids(split.validation) == {
        "2023_03_australian_grand_prix",
        "2023_04_azerbaijan_grand_prix",
    }


def test_test_split_contains_complete_future_season() -> None:
    """Every requested future-season race should belong to test."""
    split = make_standard_split()

    assert race_ids(split.test) == {
        "2024_01_bahrain_grand_prix",
        "2024_02_saudi_arabian_grand_prix",
    }

    assert set(split.test["Season"].unique()) == {
        2024,
    }


def test_no_race_appears_in_multiple_splits() -> None:
    """RaceId membership must be mutually exclusive across partitions."""
    split = make_standard_split()

    train_races = race_ids(split.train)
    validation_races = race_ids(split.validation)
    test_races = race_ids(split.test)

    assert train_races.isdisjoint(validation_races)

    assert train_races.isdisjoint(test_races)

    assert validation_races.isdisjoint(test_races)


def test_split_preserves_every_corpus_row() -> None:
    """Splitting must neither remove nor duplicate observations."""
    corpus = make_split_corpus()

    split = make_standard_split(corpus)

    assert len(corpus) == 72

    assert len(split.train) == 36
    assert len(split.validation) == 18
    assert len(split.test) == 18

    assert len(split.train) + len(split.validation) + len(split.test) == len(corpus)


def test_every_race_is_assigned_exactly_once() -> None:
    """Assignments should cover every corpus race exactly once."""
    corpus = make_split_corpus()

    split = make_standard_split(corpus)

    corpus_races = race_ids(corpus)

    assigned_races = {assignment.race_id for assignment in split.assignments}

    assert assigned_races == corpus_races

    assert len(assigned_races) == len(split.assignments)


def test_assignments_are_chronological() -> None:
    """Race assignments should preserve chronological split ordering."""
    split = make_standard_split()

    assert [assignment.split for assignment in split.assignments] == [
        DatasetSplit.TRAIN,
        DatasetSplit.TRAIN,
        DatasetSplit.TRAIN,
        DatasetSplit.TRAIN,
        DatasetSplit.VALIDATION,
        DatasetSplit.VALIDATION,
        DatasetSplit.TEST,
        DatasetSplit.TEST,
    ]

    assert [
        (
            assignment.season,
            assignment.round_number,
        )
        for assignment in split.assignments
    ] == [
        (2022, 1),
        (2022, 2),
        (2023, 1),
        (2023, 2),
        (2023, 3),
        (2023, 4),
        (2024, 1),
        (2024, 2),
    ]


def test_split_does_not_modify_source_corpus() -> None:
    """Chronological splitting must leave the source corpus unchanged."""
    corpus = make_split_corpus()

    original = corpus.copy(deep=True)

    make_standard_split(corpus)

    pd.testing.assert_frame_equal(
        corpus,
        original,
    )


def test_split_frames_are_independent_copies() -> None:
    """Changing a returned split should not mutate the original corpus."""
    corpus = make_split_corpus()

    split = make_standard_split(corpus)

    original_value = corpus.loc[
        corpus["RaceId"].eq("2022_01_bahrain_grand_prix"),
        "Driver",
    ].iloc[0]

    split.train.loc[
        0,
        "Driver",
    ] = "XXX"

    current_value = corpus.loc[
        corpus["RaceId"].eq("2022_01_bahrain_grand_prix"),
        "Driver",
    ].iloc[0]

    assert current_value == original_value


def test_split_rejects_non_dataframe() -> None:
    """The split API requires a pandas DataFrame."""
    invalid_corpus: Any = []

    with pytest.raises(
        TypeError,
        match="corpus must be provided as a pandas DataFrame",
    ):
        split_race_corpus(
            invalid_corpus,
            validation_season=2023,
            validation_last_races=2,
            test_seasons=(2024,),
        )


def test_split_wraps_invalid_corpus() -> None:
    """Malformed corpora should fail before split membership is calculated."""
    corpus = make_split_corpus().drop(
        columns=[
            "SnapshotLap",
        ]
    )

    with pytest.raises(
        CorpusSplitError,
        match="failed validation before splitting",
    ):
        split_race_corpus(
            corpus,
            validation_season=2023,
            validation_last_races=2,
            test_seasons=(2024,),
        )


@pytest.mark.parametrize(
    "validation_season",
    [
        0,
        -1,
        True,
    ],
)
def test_invalid_validation_season_is_rejected(
    validation_season: int,
) -> None:
    """Validation season must be a positive Python integer."""
    with pytest.raises(
        CorpusSplitError,
        match="validation_season must be a positive integer",
    ):
        split_race_corpus(
            make_split_corpus(),
            validation_season=validation_season,
            validation_last_races=2,
            test_seasons=(2024,),
        )


@pytest.mark.parametrize(
    "validation_last_races",
    [
        0,
        -1,
        True,
    ],
)
def test_invalid_validation_race_count_is_rejected(
    validation_last_races: int,
) -> None:
    """The number of validation races must be positive."""
    with pytest.raises(
        CorpusSplitError,
        match="validation_last_races must be a positive integer",
    ):
        split_race_corpus(
            make_split_corpus(),
            validation_season=2023,
            validation_last_races=validation_last_races,
            test_seasons=(2024,),
        )


def test_missing_validation_season_is_rejected() -> None:
    """The validation season must actually exist in the corpus."""
    with pytest.raises(
        CorpusSplitError,
        match="validation season is not present",
    ):
        split_race_corpus(
            make_split_corpus(),
            validation_season=2021,
            validation_last_races=1,
            test_seasons=(2024,),
        )


def test_validation_count_cannot_exceed_available_races() -> None:
    """Validation cannot reserve more races than its season contains."""
    with pytest.raises(
        CorpusSplitError,
        match="exceeds the number of races",
    ):
        split_race_corpus(
            make_split_corpus(),
            validation_season=2023,
            validation_last_races=5,
            test_seasons=(2024,),
        )


def test_missing_test_season_is_rejected() -> None:
    """Every requested test season must exist in the corpus."""
    with pytest.raises(
        CorpusSplitError,
        match="test seasons are not present",
    ):
        split_race_corpus(
            make_split_corpus(),
            validation_season=2023,
            validation_last_races=2,
            test_seasons=(2025,),
        )


def test_empty_test_seasons_are_rejected() -> None:
    """At least one explicit future test season is required."""
    with pytest.raises(
        CorpusSplitError,
        match="At least one test season",
    ):
        split_race_corpus(
            make_split_corpus(),
            validation_season=2023,
            validation_last_races=2,
            test_seasons=(),
        )


def test_duplicate_test_seasons_are_rejected() -> None:
    """The same test season cannot be declared twice."""
    with pytest.raises(
        CorpusSplitError,
        match="duplicate values",
    ):
        split_race_corpus(
            make_split_corpus(),
            validation_season=2023,
            validation_last_races=2,
            test_seasons=(
                2024,
                2024,
            ),
        )


def test_string_test_seasons_are_rejected() -> None:
    """A string must not be interpreted as a sequence of seasons."""
    invalid_test_seasons: Any = "2024"

    with pytest.raises(
        CorpusSplitError,
        match="sequence of positive integers",
    ):
        split_race_corpus(
            make_split_corpus(),
            validation_season=2023,
            validation_last_races=2,
            test_seasons=invalid_test_seasons,
        )


@pytest.mark.parametrize(
    "test_season",
    [
        2023,
        2022,
    ],
)
def test_test_seasons_must_follow_validation(
    test_season: int,
) -> None:
    """Testing must occur strictly after the validation season."""
    with pytest.raises(
        CorpusSplitError,
        match="must occur after the validation season",
    ):
        split_race_corpus(
            make_split_corpus(),
            validation_season=2023,
            validation_last_races=2,
            test_seasons=(test_season,),
        )


def test_unassigned_future_season_is_rejected() -> None:
    """Future corpus seasons cannot accidentally fall into training."""
    races = [
        make_race_dataset(
            season=2022,
            round_number=1,
            event_name="Bahrain Grand Prix",
        ),
        make_race_dataset(
            season=2023,
            round_number=1,
            event_name="Bahrain Grand Prix",
        ),
        make_race_dataset(
            season=2023,
            round_number=2,
            event_name="Saudi Arabian Grand Prix",
        ),
        make_race_dataset(
            season=2024,
            round_number=1,
            event_name="Bahrain Grand Prix",
        ),
        make_race_dataset(
            season=2025,
            round_number=1,
            event_name="Australian Grand Prix",
        ),
    ]

    corpus = combine_race_datasets(races)

    with pytest.raises(
        CorpusSplitError,
        match="explicitly assigned to test",
    ) as exception_info:
        split_race_corpus(
            corpus,
            validation_season=2023,
            validation_last_races=1,
            test_seasons=(2024,),
        )

    assert "2025" in str(exception_info.value)


def test_split_rejects_configuration_without_training_races() -> None:
    """Validation and test reservations must leave training observations."""
    corpus = combine_race_datasets(
        [
            make_race_dataset(
                season=2023,
                round_number=1,
                event_name="Bahrain Grand Prix",
            ),
            make_race_dataset(
                season=2023,
                round_number=2,
                event_name="Saudi Arabian Grand Prix",
            ),
            make_race_dataset(
                season=2024,
                round_number=1,
                event_name="Bahrain Grand Prix",
            ),
        ]
    )

    with pytest.raises(
        CorpusSplitError,
        match="leaves no races for training",
    ):
        split_race_corpus(
            corpus,
            validation_season=2023,
            validation_last_races=2,
            test_seasons=(2024,),
        )


def test_add_split_labels_marks_every_row() -> None:
    """The labeled corpus should expose race-level partition membership."""
    corpus = make_split_corpus()

    split = make_standard_split(corpus)

    labeled = add_split_labels(
        corpus,
        split,
    )

    assert SPLIT_COLUMN in labeled.columns

    assert set(labeled[SPLIT_COLUMN].unique()) == {
        "train",
        "validation",
        "test",
    }

    train_labels = labeled.loc[
        labeled["RaceId"].eq("2023_01_bahrain_grand_prix"),
        SPLIT_COLUMN,
    ]

    assert set(train_labels) == {
        "train",
    }

    validation_labels = labeled.loc[
        labeled["RaceId"].eq("2023_04_azerbaijan_grand_prix"),
        SPLIT_COLUMN,
    ]

    assert set(validation_labels) == {
        "validation",
    }

    test_labels = labeled.loc[
        labeled["RaceId"].eq("2024_01_bahrain_grand_prix"),
        SPLIT_COLUMN,
    ]

    assert set(test_labels) == {
        "test",
    }


def test_add_split_labels_does_not_modify_source() -> None:
    """Adding labels must return a new DataFrame."""
    corpus = make_split_corpus()

    original = corpus.copy(deep=True)

    split = make_standard_split(corpus)

    labeled = add_split_labels(
        corpus,
        split,
    )

    pd.testing.assert_frame_equal(
        corpus,
        original,
    )

    assert SPLIT_COLUMN not in corpus.columns
    assert SPLIT_COLUMN in labeled.columns


def test_add_split_labels_rejects_missing_assignment() -> None:
    """Every race in the corpus must have a split assignment."""
    corpus = make_split_corpus()

    split = make_standard_split(corpus)

    incomplete_split = CorpusSplit(
        train=split.train,
        validation=split.validation,
        test=split.test,
        assignments=split.assignments[:-1],
    )

    with pytest.raises(
        CorpusSplitError,
        match="missing one or more corpus races",
    ):
        add_split_labels(
            corpus,
            incomplete_split,
        )


def test_add_split_labels_rejects_existing_reserved_column() -> None:
    """DatasetSplit is reserved for generated split labels."""
    corpus = make_split_corpus()

    split = make_standard_split(corpus)

    corpus[SPLIT_COLUMN] = "existing"

    with pytest.raises(
        CorpusSplitError,
        match="already contains the reserved",
    ):
        add_split_labels(
            corpus,
            split,
        )


def test_add_split_labels_rejects_non_dataframe() -> None:
    """Labeling requires a pandas DataFrame."""
    invalid_corpus: Any = []

    split = make_standard_split()

    with pytest.raises(
        TypeError,
        match="corpus must be provided as a pandas DataFrame",
    ):
        add_split_labels(
            invalid_corpus,
            split,
        )


def test_add_split_labels_rejects_non_split() -> None:
    """Labeling requires an actual CorpusSplit instance."""
    invalid_split: Any = {}

    with pytest.raises(
        TypeError,
        match="split must be provided as a CorpusSplit",
    ):
        add_split_labels(
            make_split_corpus(),
            invalid_split,
        )
