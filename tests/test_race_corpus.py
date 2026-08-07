from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from telemetryx.data.corpus import (
    CORPUS_KEY_COLUMNS,
    RaceCorpusError,
    combine_race_datasets,
    summarize_race_corpus,
    validate_race_corpus,
)
from telemetryx.data.dataset import (
    RACE_DATASET_COLUMNS,
    build_race_dataset,
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
    """Return final results with the requested driver as winner."""
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


def make_three_race_corpus_inputs() -> list[pd.DataFrame]:
    """Return datasets spanning two seasons in non-chronological order."""
    race_2024_01 = make_race_dataset(
        season=2024,
        round_number=1,
        event_name="Bahrain Grand Prix",
        winner="VER",
    )

    race_2023_02 = make_race_dataset(
        season=2023,
        round_number=2,
        event_name="Saudi Arabian Grand Prix",
        winner="NOR",
    )

    race_2023_01 = make_race_dataset(
        season=2023,
        round_number=1,
        event_name="Bahrain Grand Prix",
        winner="VER",
    )

    return [
        race_2024_01,
        race_2023_02,
        race_2023_01,
    ]


def test_combine_race_datasets_creates_multi_race_corpus() -> None:
    """Valid race datasets should combine into one corpus."""
    datasets = make_three_race_corpus_inputs()

    corpus = combine_race_datasets(datasets)

    assert tuple(corpus.columns) == RACE_DATASET_COLUMNS

    assert len(corpus) == 27

    assert corpus["RaceId"].nunique() == 3

    assert set(corpus["Season"].unique()) == {
        2023,
        2024,
    }


def test_corpus_is_sorted_chronologically() -> None:
    """Input order should not determine final corpus race order."""
    corpus = combine_race_datasets(make_three_race_corpus_inputs())

    race_order = (
        corpus.loc[
            :,
            [
                "RaceId",
                "Season",
                "RoundNumber",
            ],
        ]
        .drop_duplicates()["RaceId"]
        .tolist()
    )

    assert race_order == [
        "2023_01_bahrain_grand_prix",
        "2023_02_saudi_arabian_grand_prix",
        "2024_01_bahrain_grand_prix",
    ]


def test_rows_are_sorted_within_each_race() -> None:
    """Snapshots should remain deterministically ordered inside each race."""
    corpus = combine_race_datasets(make_three_race_corpus_inputs())

    first_race = corpus.loc[corpus["RaceId"].eq("2023_01_bahrain_grand_prix")]

    assert first_race["SnapshotLap"].tolist() == [
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        3,
    ]

    lap_one = first_race.loc[first_race["SnapshotLap"].eq(1)]

    assert lap_one["Position"].tolist() == [
        1,
        2,
        3,
    ]


def test_combine_does_not_modify_source_datasets() -> None:
    """Corpus construction must preserve every source race DataFrame."""
    datasets = make_three_race_corpus_inputs()

    originals = [dataset.copy(deep=True) for dataset in datasets]

    combine_race_datasets(datasets)

    for dataset, original in zip(
        datasets,
        originals,
        strict=True,
    ):
        pd.testing.assert_frame_equal(
            dataset,
            original,
        )


def test_single_race_can_form_valid_corpus() -> None:
    """The corpus abstraction should also support one valid race."""
    race = make_race_dataset(
        season=2024,
        round_number=1,
        event_name="Bahrain Grand Prix",
    )

    corpus = combine_race_datasets([race])

    assert len(corpus) == len(race)
    assert corpus["RaceId"].nunique() == 1

    validate_race_corpus(corpus)


def test_empty_dataset_sequence_is_rejected() -> None:
    """A training corpus requires at least one race dataset."""
    with pytest.raises(
        RaceCorpusError,
        match="At least one race dataset",
    ):
        combine_race_datasets([])


def test_non_dataframe_corpus_item_is_rejected() -> None:
    """Every input item must be a pandas DataFrame."""
    valid_race = make_race_dataset(
        season=2024,
        round_number=1,
        event_name="Bahrain Grand Prix",
    )

    invalid_item: Any = {
        "RaceId": "not-a-dataframe",
    }

    with pytest.raises(
        TypeError,
        match="Invalid item index: 1",
    ):
        combine_race_datasets(
            [
                valid_race,
                invalid_item,
            ]
        )


def test_duplicate_race_is_rejected() -> None:
    """The same race must not be added to the training corpus twice."""
    race = make_race_dataset(
        season=2024,
        round_number=1,
        event_name="Bahrain Grand Prix",
    )

    with pytest.raises(
        RaceCorpusError,
        match="same race more than once",
    ):
        combine_race_datasets(
            [
                race,
                race.copy(deep=True),
            ]
        )


def test_season_round_collision_is_rejected() -> None:
    """Two different races cannot claim the same championship round."""
    first_race = make_race_dataset(
        season=2024,
        round_number=1,
        event_name="Bahrain Grand Prix",
    )

    conflicting_race = make_race_dataset(
        season=2024,
        round_number=1,
        event_name="Different Grand Prix",
    )

    with pytest.raises(
        RaceCorpusError,
        match="same season and round",
    ):
        combine_race_datasets(
            [
                first_race,
                conflicting_race,
            ]
        )


def test_extra_schema_column_is_rejected() -> None:
    """All race datasets must use exactly the canonical race schema."""
    first_race = make_race_dataset(
        season=2024,
        round_number=1,
        event_name="Bahrain Grand Prix",
    )

    second_race = make_race_dataset(
        season=2024,
        round_number=2,
        event_name="Saudi Arabian Grand Prix",
    )

    second_race["UnexpectedFeature"] = 1

    with pytest.raises(
        RaceCorpusError,
        match="schema does not match",
    ):
        combine_race_datasets(
            [
                first_race,
                second_race,
            ]
        )


def test_dtype_mismatch_is_rejected() -> None:
    """Pandas dtype changes between races should not be silently promoted."""
    first_race = make_race_dataset(
        season=2024,
        round_number=1,
        event_name="Bahrain Grand Prix",
    )

    second_race = make_race_dataset(
        season=2024,
        round_number=2,
        event_name="Saudi Arabian Grand Prix",
    )

    second_race["Position"] = second_race["Position"].astype("Float64")

    with pytest.raises(
        RaceCorpusError,
        match="dtypes do not match",
    ) as exception_info:
        combine_race_datasets(
            [
                first_race,
                second_race,
            ]
        )

    assert "Position" in str(exception_info.value)


def test_invalid_individual_race_is_rejected() -> None:
    """Every race must still satisfy the race-level validator."""
    valid_race = make_race_dataset(
        season=2024,
        round_number=1,
        event_name="Bahrain Grand Prix",
    )

    invalid_race = make_race_dataset(
        season=2024,
        round_number=2,
        event_name="Saudi Arabian Grand Prix",
    )

    invalid_race.loc[
        0,
        "DataAvailableThroughLap",
    ] = 99

    with pytest.raises(
        RaceCorpusError,
        match="failed validation",
    ):
        combine_race_datasets(
            [
                valid_race,
                invalid_race,
            ]
        )


def test_validate_race_corpus_accepts_valid_corpus() -> None:
    """The corpus validator should accept correctly combined races."""
    corpus = combine_race_datasets(make_three_race_corpus_inputs())

    validate_race_corpus(corpus)


def test_validate_race_corpus_rejects_non_dataframe() -> None:
    """Corpus validation requires a pandas DataFrame."""
    invalid_corpus: Any = []

    with pytest.raises(
        TypeError,
        match="corpus must be provided as a pandas DataFrame",
    ):
        validate_race_corpus(invalid_corpus)


def test_validate_race_corpus_rejects_empty_table() -> None:
    """An empty corpus cannot be used for model development."""
    corpus = combine_race_datasets(make_three_race_corpus_inputs())

    empty_corpus = corpus.iloc[0:0]

    with pytest.raises(
        RaceCorpusError,
        match="contains no rows",
    ):
        validate_race_corpus(empty_corpus)


def test_validate_race_corpus_reports_missing_columns() -> None:
    """Required corpus schema columns must remain present."""
    corpus = combine_race_datasets(make_three_race_corpus_inputs())

    corpus = corpus.drop(
        columns=[
            "TrackStatus",
            "CumulativeLapTimeSeconds",
        ]
    )

    with pytest.raises(
        RaceCorpusError,
        match="missing required columns",
    ) as exception_info:
        validate_race_corpus(corpus)

    message = str(exception_info.value)

    assert "TrackStatus" in message
    assert "CumulativeLapTimeSeconds" in message


def test_validate_race_corpus_rejects_duplicate_columns() -> None:
    """Duplicate DataFrame column names make corpus semantics ambiguous."""
    corpus = combine_race_datasets(make_three_race_corpus_inputs())

    corpus = pd.concat(
        [
            corpus,
            corpus[["RaceId"]],
        ],
        axis=1,
    )

    with pytest.raises(
        RaceCorpusError,
        match="duplicate column names",
    ):
        validate_race_corpus(corpus)


def test_validate_race_corpus_rejects_duplicate_keys() -> None:
    """Race, snapshot and driver must uniquely identify every corpus row."""
    corpus = combine_race_datasets(make_three_race_corpus_inputs())

    duplicated_row = corpus.iloc[[0]].copy()

    corpus = pd.concat(
        [
            corpus,
            duplicated_row,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        RaceCorpusError,
        match="duplicate race-snapshot-driver rows",
    ):
        validate_race_corpus(corpus)

    assert CORPUS_KEY_COLUMNS == (
        "RaceId",
        "SnapshotLap",
        "Driver",
    )


def test_validate_corpus_rejects_season_round_collision() -> None:
    """Corpus validation should independently detect duplicate rounds."""
    first_race = make_race_dataset(
        season=2024,
        round_number=1,
        event_name="Bahrain Grand Prix",
    )

    second_race = make_race_dataset(
        season=2024,
        round_number=1,
        event_name="Different Grand Prix",
    )

    corpus = pd.concat(
        [
            first_race,
            second_race,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        RaceCorpusError,
        match="multiple races for the same season-round",
    ):
        validate_race_corpus(corpus)


def test_corpus_summary_reports_expected_dimensions() -> None:
    """Corpus summaries should describe races, seasons and snapshots."""
    corpus = combine_race_datasets(make_three_race_corpus_inputs())

    summary = summarize_race_corpus(corpus)

    assert summary.race_count == 3
    assert summary.season_count == 2
    assert summary.row_count == 27
    assert summary.snapshot_count == 9
    assert summary.driver_count == 3

    assert summary.first_race_id == ("2023_01_bahrain_grand_prix")

    assert summary.last_race_id == ("2024_01_bahrain_grand_prix")


def test_every_corpus_snapshot_has_one_winner() -> None:
    """Each race snapshot must contain exactly one positive target."""
    corpus = combine_race_datasets(make_three_race_corpus_inputs())

    winner_counts = corpus.groupby(
        [
            "RaceId",
            "SnapshotLap",
        ]
    )["WonRace"].sum()

    assert len(winner_counts) == 9

    assert all(int(count) == 1 for count in winner_counts.tolist())
