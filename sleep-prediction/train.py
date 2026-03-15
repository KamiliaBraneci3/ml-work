"""
train.py
────────────────────────────────────────────────────────────────────────────
Model selection, cross-validation, and final training.

MODEL CHOICE — Random Forest Classifier
════════════════════════════════════════
We choose a Random Forest over alternatives for several reasons:

  ✦ Non-linearity       Our features interact in non-linear ways (e.g., the
                        post-lunch dip only matters *when combined* with low
                        sleep, not independently). Random Forest captures these
                        automatically without explicit feature crosses.

  ✦ Robustness          Ensemble averaging makes it far less sensitive to
                        outliers than individual decision trees or SVMs.

  ✦ Built-in importance Mean Decrease Impurity (MDI) gives us a first-pass
                        feature ranking for free — SHAP adds depth later.

  ✦ No scaling required Trees split on thresholds, not distances, so the
                        StandardScaler in our pipeline is there for portability
                        to other model types, not a hard requirement here.

  ✦ Interpretability    Not a black box: depth-limited trees can be inspected,
                        and SHAP explanations are well-supported for forests.

HYPERPARAMETER TUNING — 5-fold CV Grid Search
═══════════════════════════════════════════════
We search over:
  • n_estimators    : more trees → lower variance, slower training
  • max_depth       : shallow = high bias; deep = high variance
  • min_samples_leaf: prevents leaves so small they memorise noise
  • class_weight    : 'balanced' compensates for the ~35 % positive rate

The grid is intentionally modest to keep runtime under 30 s on a laptop.
"""

import os
import pathlib

import joblib
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

load_dotenv()

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
MODEL_PATH  = os.getenv("MODEL_PATH", "models/sleep_model.joblib")

PARAM_GRID = {
    "n_estimators":     [100, 200],
    "max_depth":        [4, 6, None],
    "min_samples_leaf": [5, 10, 20],
    "class_weight":     ["balanced", None],
}


def train(X_train: np.ndarray, y_train, verbose: bool = True):
    """
    Run a stratified 5-fold grid search and return the best estimator.

    Parameters
    ----------
    X_train  : preprocessed feature matrix
    y_train  : binary labels
    verbose  : print CV results

    Returns
    -------
    best_model : fitted RandomForestClassifier
    cv_results : dict from GridSearchCV
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_SEED),
        param_grid=PARAM_GRID,
        scoring="roc_auc",          # AUC is more informative than accuracy
        cv=cv,                      #   for imbalanced labels
        n_jobs=-1,
        verbose=0,
    )

    grid_search.fit(X_train, y_train)

    if verbose:
        print("\n📊  Cross-Validation Results")
        print(f"    Best params  : {grid_search.best_params_}")
        print(f"    Best CV AUC  : {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.cv_results_


def save_artifacts(model, preprocessor, feature_names: list[str]) -> None:
    """Serialise model + preprocessor together for reproducible inference."""
    pathlib.Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model":         model,
        "preprocessor":  preprocessor,
        "feature_names": feature_names,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\n💾  Artifacts saved → {MODEL_PATH}")


def load_artifacts() -> dict:
    return joblib.load(MODEL_PATH)
