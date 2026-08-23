from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from telemetryx.data.dataset import build_race_dataset
from telemetryx.features.engineering import (
    MODEL_FEATURE_COLUMNS,
    engineer_race_features,
)
from telemetryx.modeling.preprocessing import (
    FittedPreprocessor,
    PreprocessingError,
    build_preprocessor,
    fit_preprocessor,
    fit_transform_training_features,
    transform_features,
)


def make_cleaned_laps(
    *,
    drivers: tuple[str, str, str] = (
        "VER",
        "NOR",
        "BOT",
    ),
    lap_time_offset: float = 0.0,
) -> pd.DataFrame:
    """Return a small three-driver cleaned race."""
    rows: list[dict[str, object]] = []

    for position, driver in enumerate(
        drivers,
        start=1,
    ):
        base_lap_time = 89.0 + position + lap_time_offset

        for lap_number in (
            1,
            2,
            3,
        ):
            rows.append(
                {
                    "Driver": driver,
                    "LapNumber": lap_number,
                    "Position": position,
                    "Stint": (1 if lap_number < 3 else 2),
                    "Compound": ("SOFT" if lap_number < 3 else "MEDIUM"),
                    "TyreLife": (float(lap_number) if lap_number < 3 else 1.0),
                    "TrackStatus": "1",
                    "LapTimeSeconds": (base_lap_time - (lap_number - 1)),
                }
            )

    return pd.DataFrame(rows)


def make_results(
    *,
    drivers: tuple[str, str, str] = (
        "VER",
        "NOR",
        "BOT",
    ),
) -> pd.DataFrame:
    """Return final results with the first supplied driver winning."""
    return pd.DataFrame(
        {
            "Abbreviation": list(drivers),
            "Position": [
                1,
                2,
                3,
            ],
            "Status": [
                "Finished",
                "Finished",
                "Finished",
            ],
        }
    )


def make_features(
    *,
    season: int = 2023,
    round_number: int = 1,
    event_name: str = "Bahrain Grand Prix",
    drivers: tuple[str, str, str] = (
        "VER",
        "NOR",
        "BOT",
    ),
    lap_time_offset: float = 0.0,
) -> pd.DataFrame:
    """Return one valid engineered race feature frame."""
    race_dataset = build_race_dataset(
        make_cleaned_laps(
            drivers=drivers,
            lap_time_offset=lap_time_offset,
        ),
        make_results(
            drivers=drivers,
        ),
        season=season,
        round_number=round_number,
        event_name=event_name,
        session_name="Race",
    )

    return engineer_race_features(race_dataset)


def get_numeric_scaler(
    fitted: FittedPreprocessor,
) -> Any:
    """Return the fitted numeric StandardScaler."""
    numeric_pipeline = fitted.transformer.named_transformers_["numeric"]

    return numeric_pipeline.named_steps["scaler"]


def get_categorical_encoder(
    fitted: FittedPreprocessor,
) -> Any:
    """Return the fitted categorical OneHotEncoder."""
    categorical_pipeline = fitted.transformer.named_transformers_["categorical"]

    return categorical_pipeline.named_steps["one_hot_encoder"]


def test_build_preprocessor_returns_unfitted_column_transformer() -> None:
    """Construction should not inspect or fit any race observations."""
    transformer = build_preprocessor()

    assert isinstance(
        transformer,
        ColumnTransformer,
    )

    assert not hasattr(
        transformer,
        "transformers_",
    )


def test_build_preprocessor_covers_all_default_model_features() -> None:
    """Every declared model feature should have a preprocessing strategy."""
    transformer = build_preprocessor()

    configured_columns: list[str] = []

    for (
        _name,
        _pipeline,
        columns,
    ) in transformer.transformers:
        configured_columns.extend(columns)

    assert set(configured_columns) == set(MODEL_FEATURE_COLUMNS)

    assert len(configured_columns) == len(MODEL_FEATURE_COLUMNS)


def test_build_preprocessor_accepts_safe_feature_subset() -> None:
    """Experiments may preprocess a smaller audited feature set."""
    transformer = build_preprocessor(
        model_columns=(
            "Driver",
            "Position",
            "TyreLife",
        )
    )

    configured_columns: list[str] = []

    for (
        _name,
        _pipeline,
        columns,
    ) in transformer.transformers:
        configured_columns.extend(columns)

    assert set(configured_columns) == {
        "Driver",
        "Position",
        "TyreLife",
    }


def test_build_preprocessor_rejects_target_column() -> None:
    """The target must not cross into preprocessing as an input feature."""
    with pytest.raises(
        PreprocessingError,
        match=("Model columns failed validation before preprocessor construction"),
    ):
        build_preprocessor(
            model_columns=(
                "Position",
                "WonRace",
            )
        )


def test_build_preprocessor_rejects_post_race_column() -> None:
    """Post-race information must not enter preprocessing."""
    with pytest.raises(
        PreprocessingError,
        match=("Model columns failed validation before preprocessor construction"),
    ):
        build_preprocessor(
            model_columns=(
                "Position",
                "Points",
            )
        )


def test_build_preprocessor_rejects_identifier_column() -> None:
    """Race identifiers remain outside the predictive model matrix."""
    with pytest.raises(
        PreprocessingError,
        match=("Model columns failed validation before preprocessor construction"),
    ):
        build_preprocessor(
            model_columns=(
                "Position",
                "RaceId",
            )
        )


def test_build_preprocessor_rejects_empty_feature_set() -> None:
    """At least one predictive input is required."""
    with pytest.raises(
        PreprocessingError,
        match=("Model columns failed validation before preprocessor construction"),
    ):
        build_preprocessor(model_columns=())


def test_fit_preprocessor_returns_fitted_state() -> None:
    """Training features should produce reusable preprocessing state."""
    features = make_features()

    fitted = fit_preprocessor(features)

    assert isinstance(
        fitted,
        FittedPreprocessor,
    )

    assert isinstance(
        fitted.transformer,
        ColumnTransformer,
    )

    assert fitted.model_columns == (MODEL_FEATURE_COLUMNS)

    assert fitted.input_feature_count == len(MODEL_FEATURE_COLUMNS)

    assert fitted.output_feature_count > 0

    assert fitted.output_feature_count == len(fitted.output_feature_names)


def test_fitted_output_feature_names_are_unique() -> None:
    """One-hot expansion must produce an unambiguous output schema."""
    fitted = fit_preprocessor(make_features())

    assert len(fitted.output_feature_names) == len(set(fitted.output_feature_names))


def test_fit_preprocessor_accepts_safe_subset() -> None:
    """A fitted transformer should preserve the requested input contract."""
    fitted = fit_preprocessor(
        make_features(),
        model_columns=(
            "Driver",
            "Position",
            "TyreLife",
        ),
    )

    assert fitted.model_columns == (
        "Driver",
        "Position",
        "TyreLife",
    )

    assert fitted.input_feature_count == 3


def test_fit_preprocessor_rejects_non_dataframe() -> None:
    """Training preprocessing requires a pandas DataFrame."""
    invalid_features: Any = []

    with pytest.raises(
        TypeError,
        match=("training_features must be provided as a pandas DataFrame"),
    ):
        fit_preprocessor(invalid_features)


def test_fit_preprocessor_wraps_invalid_feature_frame() -> None:
    """Malformed engineered data must fail before transformer fitting."""
    features = make_features().drop(
        columns=[
            "PositionFraction",
        ]
    )

    with pytest.raises(
        PreprocessingError,
        match=("Training features failed validation before preprocessing"),
    ):
        fit_preprocessor(features)


def test_transform_returns_numeric_dataframe() -> None:
    """Transformation should produce a dense numeric model matrix."""
    features = make_features()

    fitted = fit_preprocessor(features)

    transformed = transform_features(
        fitted,
        features,
    )

    assert isinstance(
        transformed,
        pd.DataFrame,
    )

    assert len(transformed) == len(features)

    assert tuple(transformed.columns) == fitted.output_feature_names

    assert all(str(dtype) == "float64" for dtype in transformed.dtypes)


def test_transform_output_contains_only_finite_values() -> None:
    """The final model matrix must contain no NaN or infinity."""
    features = make_features()

    fitted = fit_preprocessor(features)

    transformed = transform_features(
        fitted,
        features,
    )

    values = transformed.to_numpy(dtype=float)

    assert bool(np.isfinite(values).all())


def test_transform_preserves_dataframe_index() -> None:
    """Transformed rows must retain alignment with their source rows."""
    features = make_features()

    features.index = pd.Index(
        range(
            100,
            100 + len(features),
        )
    )

    fitted = fit_preprocessor(features)

    transformed = transform_features(
        fitted,
        features,
    )

    pd.testing.assert_index_equal(
        transformed.index,
        features.index,
    )


def test_transform_rejects_non_fitted_preprocessor() -> None:
    """Transformation requires explicit fitted preprocessing state."""
    invalid_fitted: Any = build_preprocessor()

    with pytest.raises(
        TypeError,
        match=("fitted must be provided as a FittedPreprocessor"),
    ):
        transform_features(
            invalid_fitted,
            make_features(),
        )


def test_transform_rejects_non_dataframe() -> None:
    """Features supplied for transformation must be a DataFrame."""
    fitted = fit_preprocessor(make_features())

    invalid_features: Any = []

    with pytest.raises(
        TypeError,
        match=("features must be provided as a pandas DataFrame"),
    ):
        transform_features(
            fitted,
            invalid_features,
        )


def test_transform_wraps_invalid_feature_frame() -> None:
    """Transformation cannot bypass the feature contract."""
    training_features = make_features()

    fitted = fit_preprocessor(training_features)

    malformed = make_features().drop(
        columns=[
            "CompletionFraction",
        ]
    )

    with pytest.raises(
        PreprocessingError,
        match=("Features failed validation before transformation"),
    ):
        transform_features(
            fitted,
            malformed,
        )


def test_unknown_validation_driver_does_not_change_output_schema() -> None:
    """Unseen categorical values should be ignored rather than refitted."""
    training_features = make_features()

    validation_features = make_features(
        season=2024,
        round_number=1,
        event_name="Australian Grand Prix",
        drivers=(
            "PIA",
            "NOR",
            "BOT",
        ),
    )

    fitted = fit_preprocessor(
        training_features,
        model_columns=("Driver",),
    )

    training_matrix = transform_features(
        fitted,
        training_features,
    )

    validation_matrix = transform_features(
        fitted,
        validation_features,
    )

    assert tuple(training_matrix.columns) == tuple(validation_matrix.columns)

    assert tuple(validation_matrix.columns) == fitted.output_feature_names

    assert all("PIA" not in column for column in validation_matrix.columns)


def test_unknown_category_is_encoded_without_learning_new_column() -> None:
    """An unseen driver should map to the existing one-hot schema."""
    training_features = make_features()

    validation_features = make_features(
        season=2024,
        round_number=1,
        event_name="Australian Grand Prix",
        drivers=(
            "PIA",
            "NOR",
            "BOT",
        ),
    )

    fitted = fit_preprocessor(
        training_features,
        model_columns=("Driver",),
    )

    transformed = transform_features(
        fitted,
        validation_features,
    )

    pia_index = validation_features.index[validation_features["Driver"].eq("PIA")][0]

    pia_values = transformed.loc[pia_index].to_numpy(dtype=float)

    assert float(pia_values.sum()) == pytest.approx(0.0)


def test_transform_does_not_refit_categorical_encoder() -> None:
    """Validation categories must not alter training-learned categories."""
    training_features = make_features()

    validation_features = make_features(
        season=2024,
        round_number=1,
        event_name="Australian Grand Prix",
        drivers=(
            "PIA",
            "NOR",
            "BOT",
        ),
    )

    fitted = fit_preprocessor(
        training_features,
        model_columns=("Driver",),
    )

    encoder = get_categorical_encoder(fitted)

    categories_before = tuple(
        tuple(str(value) for value in category) for category in encoder.categories_
    )

    transform_features(
        fitted,
        validation_features,
    )

    categories_after = tuple(
        tuple(str(value) for value in category) for category in encoder.categories_
    )

    assert categories_after == (categories_before)

    assert "PIA" not in {value for category in categories_after for value in category}


def test_numeric_scaler_learns_training_statistics_only() -> None:
    """Scaler statistics must come exclusively from training observations."""
    training_features = make_features(lap_time_offset=0.0)

    validation_features = make_features(
        season=2024,
        round_number=1,
        event_name="Australian Grand Prix",
        lap_time_offset=100.0,
    )

    fitted = fit_preprocessor(
        training_features,
        model_columns=("LastLapTimeSeconds",),
    )

    scaler = get_numeric_scaler(fitted)

    training_mean = float(training_features["LastLapTimeSeconds"].mean())

    validation_mean = float(validation_features["LastLapTimeSeconds"].mean())

    assert training_mean == pytest.approx(90.0)

    assert validation_mean == pytest.approx(190.0)

    assert float(scaler.mean_[0]) == pytest.approx(training_mean)

    assert float(scaler.mean_[0]) != pytest.approx(validation_mean)


def test_transform_does_not_refit_numeric_scaler() -> None:
    """Transforming future data must not modify learned scaler state."""
    training_features = make_features(lap_time_offset=0.0)

    validation_features = make_features(
        season=2024,
        round_number=1,
        event_name="Australian Grand Prix",
        lap_time_offset=100.0,
    )

    fitted = fit_preprocessor(
        training_features,
        model_columns=("LastLapTimeSeconds",),
    )

    scaler = get_numeric_scaler(fitted)

    mean_before = scaler.mean_.copy()
    scale_before = scaler.scale_.copy()

    transform_features(
        fitted,
        validation_features,
    )

    np.testing.assert_array_equal(
        scaler.mean_,
        mean_before,
    )

    np.testing.assert_array_equal(
        scaler.scale_,
        scale_before,
    )


def test_validation_values_use_training_scaler_statistics() -> None:
    """Future observations should be transformed by the training scaler."""
    training_features = make_features(lap_time_offset=0.0)

    validation_features = make_features(
        season=2024,
        round_number=1,
        event_name="Australian Grand Prix",
        lap_time_offset=100.0,
    )

    fitted = fit_preprocessor(
        training_features,
        model_columns=("LastLapTimeSeconds",),
    )

    transformed = transform_features(
        fitted,
        validation_features,
    )

    assert float(transformed.mean().iloc[0]) > 10.0


def test_missing_numeric_training_value_is_imputed() -> None:
    """Training missingness should produce finite imputed model values."""
    training_features = make_features()

    training_features.loc[
        1,
        "LastLapTimeSeconds",
    ] = pd.NA

    fitted = fit_preprocessor(
        training_features,
        model_columns=("LastLapTimeSeconds",),
    )

    transformed = transform_features(
        fitted,
        training_features,
    )

    assert bool(np.isfinite(transformed.to_numpy(dtype=float)).all())

    assert fitted.output_feature_count == 2


def test_missing_numeric_value_adds_missingness_indicator() -> None:
    """Observed training missingness should be represented explicitly."""
    training_features = make_features()

    training_features.loc[
        1,
        "LastLapTimeSeconds",
    ] = pd.NA

    fitted = fit_preprocessor(
        training_features,
        model_columns=("LastLapTimeSeconds",),
    )

    assert any(
        "missingindicator" in feature_name.lower()
        for feature_name in fitted.output_feature_names
    )


def test_train_and_validation_use_identical_output_columns() -> None:
    """Every split must use the exact schema learned from training."""
    training_features = make_features()

    validation_features = make_features(
        season=2024,
        round_number=1,
        event_name="Australian Grand Prix",
        drivers=(
            "PIA",
            "NOR",
            "BOT",
        ),
        lap_time_offset=25.0,
    )

    fitted = fit_preprocessor(training_features)

    training_matrix = transform_features(
        fitted,
        training_features,
    )

    validation_matrix = transform_features(
        fitted,
        validation_features,
    )

    assert tuple(training_matrix.columns) == tuple(validation_matrix.columns)

    assert tuple(training_matrix.columns) == fitted.output_feature_names


def test_transformed_matrix_excludes_target_and_metadata() -> None:
    """Model preprocessing must not reintroduce forbidden information."""
    features = make_features()

    fitted = fit_preprocessor(features)

    transformed = transform_features(
        fitted,
        features,
    )

    prohibited_names = {
        "WonRace",
        "RaceId",
        "Season",
        "RoundNumber",
        "EventName",
        "SessionName",
        "Points",
        "Status",
    }

    for feature_name in transformed.columns:
        assert feature_name not in (prohibited_names)


def test_fit_transform_training_features_returns_both_outputs() -> None:
    """The training convenience API should return fitted state and X."""
    training_features = make_features()

    fitted, transformed = fit_transform_training_features(training_features)

    assert isinstance(
        fitted,
        FittedPreprocessor,
    )

    assert isinstance(
        transformed,
        pd.DataFrame,
    )

    assert len(transformed) == len(training_features)

    assert tuple(transformed.columns) == fitted.output_feature_names


def test_fit_transform_matches_explicit_fit_then_transform() -> None:
    """The convenience API should behave like the explicit two-step path."""
    training_features = make_features()

    explicit_fitted = fit_preprocessor(training_features)

    explicit_matrix = transform_features(
        explicit_fitted,
        training_features,
    )

    (
        convenience_fitted,
        convenience_matrix,
    ) = fit_transform_training_features(training_features)

    assert convenience_fitted.model_columns == explicit_fitted.model_columns

    assert (
        convenience_fitted.output_feature_names == explicit_fitted.output_feature_names
    )

    pd.testing.assert_frame_equal(
        convenience_matrix,
        explicit_matrix,
    )


def test_transform_is_deterministic() -> None:
    """Repeated transforms with fixed fitted state should be identical."""
    features = make_features()

    fitted = fit_preprocessor(features)

    first = transform_features(
        fitted,
        features,
    )

    second = transform_features(
        fitted,
        features,
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )
