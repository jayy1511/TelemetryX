from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from telemetryx.data.targets import TARGET_COLUMN
from telemetryx.features.engineering import (
    FEATURE_KEY_COLUMNS,
    MODEL_FEATURE_COLUMNS,
)
from telemetryx.features.leakage import (
    FeatureLeakageError,
    select_model_target,
)
from telemetryx.modeling.preprocessing import (
    FittedPreprocessor,
    PreprocessingError,
    fit_transform_training_features,
    transform_features,
)

WINNER_PROBABILITY_COLUMN: Final[str] = "WinnerProbability"

PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    *FEATURE_KEY_COLUMNS,
    WINNER_PROBABILITY_COLUMN,
)


class BaselineModelError(ValueError):
    """Raised when the baseline winner model cannot be trained or applied."""


@dataclass(frozen=True, slots=True)
class BaselineWinnerModel:
    """Fitted preprocessing state and logistic-regression winner classifier."""

    preprocessor: FittedPreprocessor
    estimator: Any
    positive_class_index: int

    @property
    def model_columns(self) -> tuple[str, ...]:
        """Return the raw model features used during training."""
        return self.preprocessor.model_columns

    @property
    def transformed_feature_names(self) -> tuple[str, ...]:
        """Return the numeric feature schema seen by the classifier."""
        return self.preprocessor.output_feature_names

    @property
    def transformed_feature_count(self) -> int:
        """Return the number of numeric classifier inputs."""
        return self.preprocessor.output_feature_count


def train_baseline_winner_model(
    training_features: pd.DataFrame,
    *,
    model_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
    max_iterations: int = 2000,
) -> BaselineWinnerModel:
    """
    Fit the TelemetryX baseline winner classifier.

    Preprocessing is fitted using the supplied training feature frame only.
    The supervised target is then fitted by a binary logistic-regression
    classifier.

    Parameters
    ----------
    training_features:
        Engineered feature frame belonging exclusively to training races.
    model_columns:
        Audited model-input columns.
    max_iterations:
        Maximum number of logistic-regression optimization iterations.

    Returns
    -------
    BaselineWinnerModel
        Fitted preprocessing state and classifier.

    Raises
    ------
    TypeError
        If ``training_features`` is not a pandas DataFrame.
    BaselineModelError
        If preprocessing, target extraction or model fitting fails.
    """
    if not isinstance(
        training_features,
        pd.DataFrame,
    ):
        raise TypeError("training_features must be provided as a pandas DataFrame.")

    _validate_max_iterations(max_iterations)

    try:
        (
            fitted_preprocessor,
            training_matrix,
        ) = fit_transform_training_features(
            training_features,
            model_columns=model_columns,
        )

        target = select_model_target(training_features)
    except (
        TypeError,
        FeatureLeakageError,
        PreprocessingError,
    ) as exc:
        raise BaselineModelError(
            "Training data failed validation before baseline model fitting."
        ) from exc

    _validate_binary_training_target(target)

    target_array = target.astype("int64").to_numpy(
        dtype=np.int64,
    )

    estimator = LogisticRegression(
        solver="lbfgs",
        max_iter=max_iterations,
    )

    try:
        estimator.fit(
            training_matrix,
            target_array,
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise BaselineModelError(
            "Failed to fit the baseline logistic-regression model."
        ) from exc

    positive_class_index = _find_positive_class_index(estimator)

    return BaselineWinnerModel(
        preprocessor=fitted_preprocessor,
        estimator=estimator,
        positive_class_index=positive_class_index,
    )


def predict_winner_probabilities(
    model: BaselineWinnerModel,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict coherent winner probabilities for every race snapshot.

    Logistic regression initially predicts an independent binary winner
    probability for each driver row. TelemetryX then normalizes those values
    within each RaceId/SnapshotLap group so the represented drivers form one
    probability distribution whose probabilities sum to one.

    Parameters
    ----------
    model:
        Fitted TelemetryX baseline model.
    features:
        Engineered feature frame to score.

    Returns
    -------
    pd.DataFrame
        RaceId, SnapshotLap, Driver and normalized WinnerProbability.

    Raises
    ------
    TypeError
        If either argument has an incorrect type.
    BaselineModelError
        If transformation or probability generation fails.
    """
    if not isinstance(
        model,
        BaselineWinnerModel,
    ):
        raise TypeError("model must be provided as a BaselineWinnerModel.")

    if not isinstance(
        features,
        pd.DataFrame,
    ):
        raise TypeError("features must be provided as a pandas DataFrame.")

    if features.empty:
        raise BaselineModelError(
            "Features cannot be empty when predicting winner probabilities."
        )

    try:
        transformed = transform_features(
            model.preprocessor,
            features,
        )
    except (
        TypeError,
        PreprocessingError,
    ) as exc:
        raise BaselineModelError(
            "Features failed preprocessing before baseline prediction."
        ) from exc

    try:
        probability_matrix = np.asarray(
            model.estimator.predict_proba(transformed),
            dtype=np.float64,
        )
    except (
        AttributeError,
        TypeError,
        ValueError,
    ) as exc:
        raise BaselineModelError(
            "Failed to generate baseline winner probabilities."
        ) from exc

    raw_probabilities = _extract_positive_probabilities(
        probability_matrix=probability_matrix,
        positive_class_index=model.positive_class_index,
        expected_rows=len(features),
    )

    normalized_probabilities = _normalize_snapshot_probabilities(
        features=features,
        probabilities=raw_probabilities,
    )

    predictions = features.loc[
        :,
        list(FEATURE_KEY_COLUMNS),
    ].copy(deep=True)

    predictions[WINNER_PROBABILITY_COLUMN] = normalized_probabilities

    predictions = predictions.loc[
        :,
        list(PREDICTION_COLUMNS),
    ].reset_index(drop=True)

    validate_winner_probability_frame(predictions)

    return predictions


def validate_winner_probability_frame(
    predictions: pd.DataFrame,
) -> None:
    """
    Validate a TelemetryX race-snapshot winner probability table.

    Each RaceId/SnapshotLap group must contain finite probabilities inside
    [0, 1] that sum to one.

    Parameters
    ----------
    predictions:
        Prediction frame produced by ``predict_winner_probabilities``.

    Raises
    ------
    TypeError
        If ``predictions`` is not a pandas DataFrame.
    BaselineModelError
        If any probability-table invariant is violated.
    """
    if not isinstance(
        predictions,
        pd.DataFrame,
    ):
        raise TypeError("predictions must be provided as a pandas DataFrame.")

    if predictions.empty:
        raise BaselineModelError("The winner probability frame contains no rows.")

    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise BaselineModelError(
            "The winner probability frame does not match the "
            "required prediction schema."
        )

    if bool(predictions[list(FEATURE_KEY_COLUMNS)].isna().any().any()):
        raise BaselineModelError(
            "Prediction key columns cannot contain missing values."
        )

    duplicate_keys = predictions.duplicated(
        subset=list(FEATURE_KEY_COLUMNS),
        keep=False,
    )

    if bool(duplicate_keys.any()):
        raise BaselineModelError(
            "The winner probability frame contains duplicate race-snapshot-driver rows."
        )

    numeric_probabilities = pd.to_numeric(
        predictions[WINNER_PROBABILITY_COLUMN],
        errors="coerce",
    )

    if bool(numeric_probabilities.isna().any()):
        raise BaselineModelError("Winner probabilities cannot contain missing values.")

    probability_array = numeric_probabilities.to_numpy(
        dtype=np.float64,
    )

    if not bool(np.isfinite(probability_array).all()):
        raise BaselineModelError(
            "Winner probabilities must contain only finite values."
        )

    if bool((probability_array < 0.0).any()) or bool((probability_array > 1.0).any()):
        raise BaselineModelError("Winner probabilities must remain between 0 and 1.")

    probability_sums = (
        predictions.assign(_Probability=numeric_probabilities)
        .groupby(
            [
                "RaceId",
                "SnapshotLap",
            ],
            sort=False,
            dropna=False,
        )["_Probability"]
        .sum()
    )

    sums_array = probability_sums.to_numpy(
        dtype=np.float64,
    )

    if not bool(
        np.isclose(
            sums_array,
            1.0,
            rtol=1e-9,
            atol=1e-9,
        ).all()
    ):
        raise BaselineModelError(
            "Winner probabilities must sum to one within every race snapshot."
        )


def _validate_binary_training_target(
    target: pd.Series,
) -> None:
    """Require both winner and non-winner observations during training."""
    if target.empty:
        raise BaselineModelError("The training target cannot be empty.")

    if bool(target.isna().any()):
        raise BaselineModelError(
            f"{TARGET_COLUMN} cannot contain missing training values."
        )

    positive_count = int(target.sum())

    negative_count = len(target) - positive_count

    if positive_count <= 0:
        raise BaselineModelError("The training target contains no winner observations.")

    if negative_count <= 0:
        raise BaselineModelError(
            "The training target contains no non-winner observations."
        )


def _validate_max_iterations(
    max_iterations: int,
) -> None:
    """Require a positive Python integer optimization limit."""
    if (
        isinstance(
            max_iterations,
            bool,
        )
        or not isinstance(
            max_iterations,
            int,
        )
        or max_iterations <= 0
    ):
        raise BaselineModelError("max_iterations must be a positive integer.")


def _find_positive_class_index(
    estimator: Any,
) -> int:
    """Return the predict_proba column corresponding to winner class 1."""
    try:
        classes = np.asarray(estimator.classes_)
    except AttributeError as exc:
        raise BaselineModelError(
            "The fitted classifier does not expose class labels."
        ) from exc

    matches = np.flatnonzero(classes == 1)

    if len(matches) != 1:
        raise BaselineModelError(
            "The fitted classifier must contain exactly one positive winner class."
        )

    return int(matches[0])


def _extract_positive_probabilities(
    *,
    probability_matrix: np.ndarray,
    positive_class_index: int,
    expected_rows: int,
) -> np.ndarray:
    """Extract and validate the classifier's positive-class probabilities."""
    if probability_matrix.ndim != 2:
        raise BaselineModelError(
            "Classifier probabilities must form a two-dimensional matrix."
        )

    if probability_matrix.shape[0] != expected_rows:
        raise BaselineModelError(
            "Classifier probability rows do not match the feature frame."
        )

    if positive_class_index < 0 or positive_class_index >= probability_matrix.shape[1]:
        raise BaselineModelError(
            "The positive winner class index is outside the "
            "classifier probability matrix."
        )

    probabilities = probability_matrix[
        :,
        positive_class_index,
    ]

    if not bool(np.isfinite(probabilities).all()):
        raise BaselineModelError("Classifier probabilities contain non-finite values.")

    if bool((probabilities < 0.0).any()) or bool((probabilities > 1.0).any()):
        raise BaselineModelError(
            "Classifier probabilities must remain between 0 and 1."
        )

    return probabilities.astype(
        np.float64,
        copy=True,
    )


def _normalize_snapshot_probabilities(
    *,
    features: pd.DataFrame,
    probabilities: np.ndarray,
) -> pd.Series:
    """
    Normalize binary probabilities into one distribution per snapshot.

    This makes the baseline output interpretable as mutually exclusive winner
    probabilities among the drivers represented in the snapshot.
    """
    if len(probabilities) != len(features):
        raise BaselineModelError(
            "Probability count does not match the number of feature rows."
        )

    raw_probability = pd.Series(
        probabilities,
        index=features.index,
        dtype="float64",
    )

    group_totals = raw_probability.groupby(
        [
            features["RaceId"],
            features["SnapshotLap"],
        ],
        sort=False,
        dropna=False,
    ).transform("sum")

    invalid_total = group_totals.isna() | group_totals.le(0.0)

    if bool(invalid_total.any()):
        raise BaselineModelError(
            "A race snapshot has no positive probability mass "
            "available for normalization."
        )

    normalized = raw_probability.div(group_totals)

    return normalized.astype("float64")
