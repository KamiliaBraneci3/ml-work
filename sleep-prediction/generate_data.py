"""
generate_data.py
────────────────────────────────────────────────────────────────────────────
Synthetic dataset generator for the Lecture Sleep Predictor.

WHY SYNTHETIC DATA?
-------------------
Real datasets about students falling asleep are scarce and ethically
sensitive.  Instead, we encode behavioural-science knowledge directly into
the generation process:

  • Sleep debt         → fewer hours → higher drowsiness
  • Circadian rhythm   → post-lunch dip  (13 h – 15 h)
  • Environment        → warm rooms accelerate fatigue
  • Stimulants         → caffeine partially offsets sleepiness
  • Lecture length     → longer exposure = higher probability
  • Weekly fatigue     → sleepiness accumulates Thursday → Friday
  • Subject difficulty → paradoxically, very hard topics cause
                         "cognitive overload fatigue"; moderate
                         difficulty keeps the brain engaged

The label `fell_asleep` is drawn from a Bernoulli distribution whose
probability is the output of a logistic function of the above factors.
This produces realistic, non-trivial relationships without a clean
decision boundary — exactly what a real ML problem looks like.
"""

import os
import pathlib

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

N_SAMPLES   = int(os.getenv("N_SAMPLES",   2000))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
DATA_PATH   = os.getenv("DATA_PATH", "data/lecture_data.csv")


def _logistic(x: np.ndarray) -> np.ndarray:
    """Numerically-stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def generate(n: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── Raw features ─────────────────────────────────────────────────────
    sleep_hours       = rng.normal(loc=6.5, scale=1.2, size=n).clip(3, 9)
    lecture_duration  = rng.uniform(45, 180, size=n)           # minutes
    hour_of_day       = rng.integers(8, 19, size=n)            # 8 AM – 6 PM
    room_temp         = rng.normal(loc=22, scale=2.5, size=n).clip(16, 30)  # °C
    subject_difficulty= rng.integers(1, 6, size=n)             # 1 (easy) – 5 (hard)
    caffeine_mg       = rng.choice(
        [0, 40, 80, 120, 200, 300],
        p=[0.25, 0.20, 0.25, 0.15, 0.10, 0.05],
        size=n,
    )
    day_of_week       = rng.integers(0, 5, size=n)             # 0=Mon … 4=Fri

    # ── Linear score  (domain-knowledge weights) ─────────────────────────
    #   Positive coefficient  → pushes toward falling asleep
    #   Negative coefficient  → keeps the student awake
    score = (
          4.0                                  # intercept (baseline)
        - 0.55 * sleep_hours                   # more sleep → less sleepy
        + 0.018 * lecture_duration             # longer lecture → more fatigue
        + 0.25 * np.maximum(0, 15 - hour_of_day)  # morning alertness
        - 0.30 * np.maximum(0, hour_of_day - 15)  # evening fatigue
        + 1.20 * ((hour_of_day >= 13) & (hour_of_day <= 15)).astype(float)  # post-lunch dip
        + 0.12 * room_temp                     # warm room → sleepy
        + 0.20 * (subject_difficulty == 5).astype(float)   # overwhelm
        + 0.15 * (subject_difficulty == 1).astype(float)   # boredom
        - 0.0045 * caffeine_mg                 # caffeine offset
        + 0.18 * day_of_week                   # accumulated weekly fatigue
        + rng.normal(0, 0.5, size=n)           # individual noise
    )

    prob_sleep = _logistic(score - 6.0)        # shift to get ~35 % base rate
    fell_asleep = rng.binomial(1, prob_sleep).astype(int)

    df = pd.DataFrame({
        "sleep_hours":        np.round(sleep_hours, 2),
        "lecture_duration_min": np.round(lecture_duration, 1),
        "hour_of_day":        hour_of_day,
        "room_temp_celsius":  np.round(room_temp, 1),
        "subject_difficulty": subject_difficulty,
        "caffeine_mg":        caffeine_mg,
        "day_of_week":        day_of_week,          # 0=Mon … 4=Fri
        "fell_asleep":        fell_asleep,
    })

    return df


if __name__ == "__main__":
    pathlib.Path(DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(DATA_PATH, index=False)

    print(f"✅  Dataset saved → {DATA_PATH}")
    print(f"    Shape        : {df.shape}")
    print(f"    Sleep rate   : {df['fell_asleep'].mean():.1%}")
    print(df.describe().round(2).to_string())
