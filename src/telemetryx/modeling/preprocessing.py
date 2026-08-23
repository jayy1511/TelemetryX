from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from telemetryx.features.engineering import (
    BOOLEAN_FEATURE_COLUMNS,
    CATEGORICAL_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
)
from telemetryx.features.leakage import (
    FeatureLeakageError,
    select_model_features,
    validate_model_feature_columns,
)


class PreprocessingError(ValueError):
    """Raised when model preprocessing cannot be performed safely."""


@dataclass(frozen=True, slots=True)
class FittedPreprocessor:
    """A preprocessing transformer fitted exclusively on training data."""

    transformer: ColumnTransformer
    model_columns: tuple[str, ...]
    output_feature_names: tuple[str, ...]

    @property
    def input_feature_count(self) -> int:
        """Return the number of raw model-input columns."""
        return len(self.model_columns)

    @property
    def output_feature_count(self) -> int:
        """Return the number of transformed model-input columns."""
        return len(self.output_feature_names)


def build_preprocessor(
    *,
    model_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
) -> ColumnTransformer:
    """
    Build an unfitted TelemetryX preprocessing transformer.

    Categorical columns are imputed and one-hot encoded. Numeric and Boolean
    columns are median-imputed and standardized.

    The returned transformer is deliberately unfitted. It must later be
    fitted using training data only.

    Parameters
    ----------
    model_columns:
        Safe model-input columns to preprocess.

    Returns
    -------
    ColumnTransformer
        Unfitted scikit-learn preprocessing pipeline.

    Raises
    ------
    PreprocessingError
        If columns violate the TelemetryX feature contract.
    """
    try:
        validated_columns = validate_model_feature_columns(model_columns)
    except (
        TypeError,
        FeatureLeakageError,
    ) as exc:
        raise PreprocessingError(
            "Model columns failed validation before preprocessor construction."
        ) from exc

    categorical_columns = tuple(
        column for column in validated_columns if column in CATEGORICAL_FEATURE_COLUMNS
    )

    numeric_columns = tuple(
        column
        for column in validated_columns
        if column in NUMERIC_FEATURE_COLUMNS or column in BOOLEAN_FEATURE_COLUMNS
    )

    covered_columns = set(categorical_columns) | set(numeric_columns)

    uncovered_columns = [
        column for column in validated_columns if column not in covered_columns
    ]

    if uncovered_columns:
        raise PreprocessingError(
            "One or more model features do not have a preprocessing "
            "strategy: "
            f"{', '.join(uncovered_columns)}."
        )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                    keep_empty_features=True,
                ),
            ),
            (
                "one_hot_encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float64,
                ),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                list(categorical_columns),
            ),
            (
                "numeric",
                numeric_pipeline,
                list(numeric_columns),
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def fit_preprocessor(
    training_features: pd.DataFrame,
    *,
    model_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
) -> FittedPreprocessor:
    """
    Fit preprocessing using training observations only.

    Callers should pass only the engineered training split to this function.
    Validation and test observations must never participate in ``fit``.

    Parameters
    ----------
    training_features:
        Engineered feature frame belonging exclusively to the training split.
    model_columns:
        Safe model-input columns to use.

    Returns
    -------
    FittedPreprocessor
        Fitted transformer plus its input and output feature schemas.

    Raises
    ------
    TypeError
        If ``training_features`` is not a pandas DataFrame.
    PreprocessingError
        If feature selection or transformer fitting fails.
    """
    if not isinstance(
        training_features,
        pd.DataFrame,
    ):
        raise TypeError("training_features must be provided as a pandas DataFrame.")

    try:
        validated_columns = validate_model_feature_columns(model_columns)

        training_matrix = select_model_features(
            training_features,
            model_columns=validated_columns,
        )
    except (
        TypeError,
        FeatureLeakageError,
    ) as exc:
        raise PreprocessingError(
            "Training features failed validation before preprocessing."
        ) from exc

    if training_matrix.empty:
        raise PreprocessingError("Training features cannot be empty.")

    transformer = build_preprocessor(model_columns=validated_columns)

    try:
        transformer.fit(training_matrix)
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise PreprocessingError(
            "Failed to fit the preprocessing transformer on training data."
        ) from exc

    try:
        output_feature_names = tuple(
            str(value) for value in (transformer.get_feature_names_out().tolist())
        )
    except (
        AttributeError,
        ValueError,
    ) as exc:
        raise PreprocessingError(
            "Could not determine transformed feature names."
        ) from exc

    if not output_feature_names:
        raise PreprocessingError("The fitted preprocessor produced no output features.")

    if len(set(output_feature_names)) != len(output_feature_names):
        raise PreprocessingError(
            "The fitted preprocessor produced duplicate output feature names."
        )

    return FittedPreprocessor(
        transformer=transformer,
        model_columns=validated_columns,
        output_feature_names=output_feature_names,
    )


def transform_features(
    fitted: FittedPreprocessor,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Transform an engineered feature frame using a fitted preprocessor.

    This function never calls ``fit``. The same preprocessing statistics
    learned from training are therefore reused for validation, test and later
    replay observations.

    Parameters
    ----------
    fitted:
        Preprocessor previously fitted on training data.
    features:
        Engineered feature frame to transform.

    Returns
    -------
    pd.DataFrame
        Dense numeric matrix suitable for model training or inference.

    Raises
    ------
    TypeError
        If the arguments have incorrect types.
    PreprocessingError
        If transformation fails or produces invalid numeric values.
    """
    if not isinstance(
        fitted,
        FittedPreprocessor,
    ):
        raise TypeError("fitted must be provided as a FittedPreprocessor.")

    if not isinstance(
        features,
        pd.DataFrame,
    ):
        raise TypeError("features must be provided as a pandas DataFrame.")

    try:
        model_features = select_model_features(
            features,
            model_columns=fitted.model_columns,
        )
    except (
        TypeError,
        FeatureLeakageError,
    ) as exc:
        raise PreprocessingError(
            "Features failed validation before transformation."
        ) from exc

    try:
        transformed = fitted.transformer.transform(model_features)
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise PreprocessingError(
            "Failed to transform features using the fitted preprocessor."
        ) from exc

    transformed_array = np.asarray(
        transformed,
        dtype=np.float64,
    )

    if transformed_array.ndim != 2:
        raise PreprocessingError(
            "The transformed feature matrix must be two-dimensional."
        )

    expected_shape = (
        len(features),
        len(fitted.output_feature_names),
    )

    if transformed_array.shape != expected_shape:
        raise PreprocessingError(
            "The transformed feature matrix has an unexpected shape: "
            f"expected {expected_shape}, received "
            f"{transformed_array.shape}."
        )

    if not bool(np.isfinite(transformed_array).all()):
        raise PreprocessingError(
            "The transformed feature matrix contains non-finite values."
        )

    return pd.DataFrame(
        transformed_array,
        columns=list(fitted.output_feature_names),
        index=features.index.copy(),
        dtype="float64",
    )


def fit_transform_training_features(
    training_features: pd.DataFrame,
    *,
    model_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
) -> tuple[
    FittedPreprocessor,
    pd.DataFrame,
]:
    """
    Fit the preprocessor on training data and transform that same split.

    This convenience function keeps the fitting operation explicit while
    returning both the fitted transformer and numeric training matrix.

    Parameters
    ----------
    training_features:
        Engineered feature frame belonging only to training races.
    model_columns:
        Safe predictive input columns.

    Returns
    -------
    tuple[FittedPreprocessor, pd.DataFrame]
        Fitted preprocessing state and transformed training matrix.
    """
    fitted = fit_preprocessor(
        training_features,
        model_columns=model_columns,
    )

    transformed = transform_features(
        fitted,
        training_features,
    )

    return (
        fitted,
        transformed,
    )
