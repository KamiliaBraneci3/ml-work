"""
evaluate.py
────────────────────────────────────────────────────────────────────────────
Model evaluation, visualisation, and interpretation.

METRICS EXPLAINED
=================
  Accuracy   — % of all predictions correct.  Misleading on imbalanced data
               (a model that always predicts "awake" would score ~65 %).

  ROC-AUC    — Area under the Receiver-Operating-Characteristic curve.
               Measures the model's ability to *rank* sleepy students above
               alert ones regardless of threshold.  0.5 = random, 1.0 = perfect.

  Precision  — Of students predicted to fall asleep, how many actually did?
               Important when false alarms are costly.

  Recall     — Of students who actually fell asleep, how many did we catch?
               Important when missing a case is costly.

  F1-Score   — Harmonic mean of precision and recall.

INTERPRETATION — Permutation Importance + Local Sensitivity
=============================================================
We use two complementary techniques:

  1. Permutation Importance (global)
     Randomly shuffles one feature at a time on the TEST set and measures the
     AUC drop. If shuffling "sleep_hours" causes AUC to drop by 0.10, that
     feature is critical. Unlike tree MDI (Mean Decrease Impurity), permutation
     importance is model-agnostic and measured on held-out data — so it's
     not biased towards high-cardinality features.

  2. Local Sensitivity Analysis (individual)
     For the most ambiguous student (predicted prob ≈ 0.5), we sweep each
     feature across its realistic range while holding everything else fixed,
     then plot how the predicted sleep probability changes. This answers
     "what would have changed this student's outcome?" — the same intuition
     as a SHAP waterfall plot, with no extra library.
"""

import os
import pathlib

import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
)

load_dotenv()
PLOT_DIR = os.getenv("PLOT_DIR", "plots/")


def _ensure_dir():
    pathlib.Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)


# ── 1. Classification report ─────────────────────────────────────────────

def print_classification_metrics(model, X_test, y_test, feature_names):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "═" * 58)
    print("  MODEL EVALUATION ON HELD-OUT TEST SET")
    print("═" * 58)
    print(classification_report(y_test, y_pred,
                                 target_names=["Awake (0)", "Asleep (1)"]))
    print(f"  ROC-AUC  :  {roc_auc_score(y_test, y_proba):.4f}")
    print("═" * 58)

    return y_pred, y_proba


# ── 2. Confusion matrix ──────────────────────────────────────────────────

def plot_confusion_matrix(model, X_test, y_test):
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=["Awake", "Asleep"],
        cmap="Blues", ax=ax,
    )
    ax.set_title("Confusion Matrix — Test Set", fontsize=13, pad=12)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "01_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📈  Saved → {path}")


# ── 3. ROC curve ─────────────────────────────────────────────────────────

def plot_roc_curve(y_test, y_proba):
    _ensure_dir()
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc          = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, color="#e07b4f", label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#aaaaaa", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=13)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "02_roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📈  Saved → {path}")


# ── 4. Feature importance (MDI) ──────────────────────────────────────────

def plot_feature_importance(model, feature_names):
    _ensure_dir()
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#5c8a91" if i < len(importances) - 3 else "#e07b4f"
              for i in range(len(importances))]
    importances.plot(kind="barh", ax=ax, color=colors)
    ax.set_xlabel("Mean Decrease Impurity")
    ax.set_title("Feature Importance (Random Forest MDI)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "03_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📈  Saved → {path}")


# ── 5. Permutation importance ────────────────────────────────────────────

def plot_permutation_importance(model, X_test, y_test, feature_names,
                                 random_state: int = 42):
    """
    Permutation importance on the test set.

    WHY THIS OVER MDI?
    ──────────────────
    MDI (the bar chart above) is computed on training splits and is biased
    towards high-cardinality features. Permutation importance is measured
    on the held-out test set and reflects true generalisation: it answers
    "how much does this feature actually help on unseen data?"

    We report mean ± std across 10 shuffle repetitions to show stability.
    """
    _ensure_dir()
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        scoring="roc_auc",
        random_state=random_state,
    )

    pi = pd.DataFrame({
        "feature": feature_names,
        "mean":    result.importances_mean,
        "std":     result.importances_std,
    }).sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#e07b4f" if m > 0 else "#aaaaaa" for m in pi["mean"]]
    ax.barh(pi["feature"], pi["mean"], xerr=pi["std"],
            color=colors, capsize=3, error_kw={"elinewidth": 1})
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Mean AUC decrease when feature is permuted")
    ax.set_title("Permutation Importance — Test Set (10 repeats)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "04_permutation_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📈  Saved → {path}")

    return pi


# ── 6. Local sensitivity analysis for one student ────────────────────────

def plot_local_sensitivity(model, X_test, y_test, y_proba, feature_names,
                            preprocessor, df_raw: pd.DataFrame):
    """
    For the most ambiguous student (predicted prob ≈ 0.5), sweep each
    RAW feature across its realistic range while holding others fixed.

    This answers: "which knob, if turned, would most change this student's
    outcome?"  It is a manual, intuitive alternative to SHAP force plots.
    """
    _ensure_dir()
    idx      = int(np.argmin(np.abs(y_proba - 0.5)))
    base_row = df_raw.drop(columns=["fell_asleep"]).iloc[idx].copy()

    sweep_config = {
        "sleep_hours":           np.linspace(3, 9, 60),
        "caffeine_mg":           np.linspace(0, 300, 60),
        "lecture_duration_min":  np.linspace(45, 180, 60),
        "room_temp_celsius":     np.linspace(16, 30, 60),
        "hour_of_day":           np.arange(8, 19),
    }

    fig, axes = plt.subplots(1, len(sweep_config), figsize=(16, 3.5), sharey=True)

    for ax, (feat, sweep_vals) in zip(axes, sweep_config.items()):
        probs = []
        for v in sweep_vals:
            row = base_row.copy()
            row[feat] = v
            row_df = pd.DataFrame([row])
            X_enc  = preprocessor.transform(row_df)
            probs.append(model.predict_proba(X_enc)[0, 1])

        ax.plot(sweep_vals, probs, color="#e07b4f", lw=2)
        ax.axhline(0.5, color="#aaaaaa", lw=0.8, linestyle="--")
        ax.axvline(base_row[feat], color="#5c8a91", lw=1.5, linestyle=":")
        ax.set_title(feat.replace("_", "\n"), fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Feature value", fontsize=7)

    axes[0].set_ylabel("P(fall asleep)")
    fig.suptitle(
        f"Local Sensitivity — Student #{idx}  "
        f"(Actual: {'Asleep' if y_test.iloc[idx] == 1 else 'Awake'}, "
        f"Predicted prob: {y_proba[idx]:.2f})\n"
        f"Blue dotted line = student's actual value",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "05_local_sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📈  Saved → {path}")


# ── 7. EDA — distribution of key features by label ───────────────────────

def plot_eda(df: pd.DataFrame):
    _ensure_dir()
    features = ["sleep_hours", "caffeine_mg", "lecture_duration_min",
                "room_temp_celsius", "hour_of_day"]

    fig, axes = plt.subplots(1, len(features), figsize=(16, 4), sharey=False)
    palette = {0: "#5c8a91", 1: "#e07b4f"}
    labels  = {0: "Awake", 1: "Asleep"}

    for ax, feat in zip(axes, features):
        for label, color in palette.items():
            subset = df[df["fell_asleep"] == label][feat]
            sns.kdeplot(subset, ax=ax, color=color,
                        fill=True, alpha=0.4, label=labels[label])
        ax.set_title(feat.replace("_", "\n"), fontsize=9)
        ax.set_xlabel("")
        ax.legend(fontsize=7)

    fig.suptitle("Feature Distributions — Awake vs Asleep Students",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "00_eda_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📈  Saved → {path}")
