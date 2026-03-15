"""
main.py
────────────────────────────────────────────────────────────────────────────
End-to-end pipeline orchestrator for the Lecture Sleep Predictor.

Run order:
    python main.py

Steps executed:
    1. Generate synthetic dataset
    2. EDA plots
    3. Preprocess features
    4. Train + cross-validate Random Forest
    5. Evaluate on held-out test set
    6. SHAP global & individual interpretation
    7. Save model artifacts
"""

import os
import pathlib

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from generate_data import generate
from preprocess    import preprocess, get_feature_names
from train         import train, save_artifacts
from evaluate      import (
    plot_eda,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_permutation_importance,
    plot_local_sensitivity,
    print_classification_metrics,
)

load_dotenv()
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
TEST_SIZE   = float(os.getenv("TEST_SIZE",   0.2))
DATA_PATH   = os.getenv("DATA_PATH", "data/lecture_data.csv")


def main():
    print("\n" + "═" * 58)
    print("  LECTURE SLEEP PREDICTOR — Full ML Pipeline")
    print("═" * 58)

    # ── STEP 1: Data generation ───────────────────────────────────────────
    print("\n[1/6]  Generating synthetic dataset …")
    df = generate()
    pathlib.Path(DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"       {len(df)} records  |  sleep rate: {df['fell_asleep'].mean():.1%}")

    # ── STEP 2: EDA ───────────────────────────────────────────────────────
    print("\n[2/6]  Exploratory Data Analysis …")
    plot_eda(df)

    # ── STEP 3: Train / test split ────────────────────────────────────────
    print("\n[3/6]  Preprocessing …")
    df_train, df_test = train_test_split(
        df, test_size=TEST_SIZE,
        stratify=df["fell_asleep"],
        random_state=RANDOM_SEED,
    )

    X_train, y_train, preprocessor = preprocess(df_train)
    X_test,  y_test,  _            = preprocess(df_test, preprocessor)

    feature_names = get_feature_names()
    print(f"       Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"       Features after encoding: {feature_names}")

    # ── STEP 4: Training ──────────────────────────────────────────────────
    print("\n[4/6]  Training Random Forest (grid search CV) …")
    model, _ = train(X_train, y_train, verbose=True)

    # ── STEP 5: Evaluation ────────────────────────────────────────────────
    print("\n[5/6]  Evaluating on test set …")
    y_pred, y_proba = print_classification_metrics(
        model, X_test, y_test, feature_names
    )
    plot_confusion_matrix(model, X_test, y_test)
    plot_roc_curve(y_test, y_proba)
    plot_feature_importance(model, feature_names)

    # ── STEP 6: Interpretation ────────────────────────────────────────────
    print("\n[6/6]  Interpretation …")
    plot_permutation_importance(model, X_test, y_test, feature_names)
    plot_local_sensitivity(model, X_test, y_test, y_proba,
                           feature_names, preprocessor, df_test)

    # ── STEP 7: Save ──────────────────────────────────────────────────────
    save_artifacts(model, preprocessor, feature_names)

    print("\n✅  Pipeline complete!  Check the plots/ directory for visualisations.")
    print("═" * 58 + "\n")


if __name__ == "__main__":
    main()
