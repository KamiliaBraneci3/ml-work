"""
preprocess.py
────────────────────────────────────────────────────────────────────────────
Feature engineering and preprocessing pipeline.

WHAT WE DO AND *WHY*
=====================

1. HANDLE SKEWED DISTRIBUTIONS
   ─────────────────────────────
   `caffeine_mg` is right-skewed (many students at 0 or 80 mg, few at 300 mg).
   Tree-based models tolerate skew natively, but the upcoming SHAP analysis
   and any downstream linear model benefit from a more symmetric distribution.
   → We apply log1p (log(x+1)) to avoid log(0) issues.

2. CYCLIC ENCODING FOR `hour_of_day`
   ────────────────────────────────────
   Hours 8 and 18 are numerically far apart (difference = 10) but they share
   a similar context — "first/last lecture of the day, relatively alert".
   More importantly, 12 and 13 (numerically close) straddle the post-lunch dip.
   Raw integer encoding tricks linear models into thinking 23 is "further from"
   8 than 12 is, which is wrong in a circular sense.
   → We project hour onto the unit circle: sin(2π·h/24), cos(2π·h/24).
   Tree models don't need this, but it's best practice and costs nothing.

3. CYCLIC ENCODING FOR `day_of_week`
   ─────────────────────────────────
   Same logic: day 4 (Friday) and day 0 (Monday) are conceptually adjacent
   in weekly fatigue cycles.
   → sin(2π·d/5), cos(2π·d/5).

4. STANDARDISE CONTINUOUS FEATURES
   ────────────────────────────────
   RandomForest is scale-invariant, but scaling is done here so the same
   pipeline can be swapped for Logistic Regression, SVM, etc. without
   touching this file.  StandardScaler ensures mean=0, std=1.

5. NO IMPUTATION NEEDED
   ─────────────────────
   Our synthetic data has no missing values by construction.  In production
   you would add a SimpleImputer step here.

OUTPUT
──────
Returns an (X, y) pair and the fitted ColumnTransformer so it can be
serialised alongside the model for inference on new data.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


# ── Column groups ────────────────────────────────────────────────────────
CONTINUOUS_COLS  = ["sleep_hours", "lecture_duration_min", "room_temp_celsius"]
SKEWED_COLS      = ["caffeine_mg"]
CYCLIC_HOUR_COL  = "hour_of_day"
CYCLIC_DAY_COL   = "day_of_week"
ORDINAL_COL      = ["subject_difficulty"]   # keep as-is; tree handles ordinal naturally
TARGET_COL       = "fell_asleep"


# ── Helper: cyclic transform ─────────────────────────────────────────────
from sklearn.base import BaseEstimator, TransformerMixin

class CyclicEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes a single numeric column as (sin, cos) on a circular period.
    Fully picklable — safe to serialise with joblib.
    """
    def __init__(self, period: int):
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x = np.asarray(X).ravel()
        return np.column_stack([
            np.sin(2 * np.pi * x / self.period),
            np.cos(2 * np.pi * x / self.period),
        ])

def make_cyclic_transformer(period: int):
    return CyclicEncoder(period=period)


def build_preprocessor() -> ColumnTransformer:
    """Construct the sklearn ColumnTransformer (unfitted)."""

    log1p_scale = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, validate=False)),
        ("scale", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ("continuous", StandardScaler(),              CONTINUOUS_COLS),
            ("caffeine",   log1p_scale,                   SKEWED_COLS),
            ("ordinal",    StandardScaler(),              ORDINAL_COL),
            ("hour_cyc",   make_cyclic_transformer(24),  [CYCLIC_HOUR_COL]),
            ("day_cyc",    make_cyclic_transformer(5),   [CYCLIC_DAY_COL]),
        ],
        remainder="drop",
    )


def get_feature_names() -> list[str]:
    """Human-readable names matching ColumnTransformer output order."""
    return (
        CONTINUOUS_COLS
        + SKEWED_COLS
        + ORDINAL_COL
        + [f"hour_{fn}" for fn in ["sin", "cos"]]
        + [f"day_{fn}"  for fn in ["sin", "cos"]]
    )


def preprocess(df: pd.DataFrame, preprocessor: ColumnTransformer = None):
    """
    Split features/target, fit (or apply) the preprocessor.

    Parameters
    ----------
    df            : raw DataFrame from generate_data.py
    preprocessor  : if None, a new one is built and fitted (training mode).
                    Pass a fitted instance for inference mode.

    Returns
    -------
    X_transformed : np.ndarray
    y             : pd.Series  (0/1)
    preprocessor  : fitted ColumnTransformer
    """
    X_raw = df.drop(columns=[TARGET_COL])
    y     = df[TARGET_COL]

    if preprocessor is None:
        preprocessor = build_preprocessor()
        X_transformed = preprocessor.fit_transform(X_raw)
    else:
        X_transformed = preprocessor.transform(X_raw)

    return X_transformed, y, preprocessor
