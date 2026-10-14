"""
Synthetic loan application generator.

Real loan books are private, so for a portfolio project we generate a
realistic synthetic one instead. The generator models the relationships
you'd expect in real data: lower credit scores and higher debt-to-income
ratios drive higher default probabilities, and we bake that into the
labels so the trained models have real signal to learn from.

Run:
    python -m src.data_gen --n 50000 --out data/raw/applications.parquet
"""

import argparse
import numpy as np
import pandas as pd

from src.config import from_env


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(n: int, base_rate: float, seed: int) -> pd.DataFrame:
    """Generate n synthetic loan applications with a default label."""
    rng = np.random.default_rng(seed)

    credit_score = rng.integers(350, 850, size=n)
    annual_income = rng.lognormal(mean=11.0, sigma=0.55, size=n).clip(15_000, 500_000)
    dti = rng.beta(a=2.5, b=5.0, size=n)  # debt-to-income ratio, 0..1
    loan_amount = rng.lognormal(mean=10.0, sigma=0.8, size=n).clip(1_000, 250_000)
    term_months = rng.choice([36, 60], size=n, p=[0.6, 0.4])
    employment_years = rng.integers(0, 30, size=n)
    home_ownership = rng.choice(
        ["RENT", "OWN", "MORTGAGE"], size=n, p=[0.45, 0.25, 0.30]
    )
    interest_rate = (
        0.05
        + (850 - credit_score) / 500 * 0.12
        + rng.normal(0, 0.008, size=n)
    ).clip(0.04, 0.30)

    # Latent default risk: worse credit + high DTI + big loan relative to
    # income all push the probability up. We shift the intercept so the
    # overall default rate lands near base_rate.
    logit = (
        -2.2
        + (700 - credit_score) / 100 * 0.9
        + dti * 3.0
        + (loan_amount / annual_income) * 1.5
        - employment_years / 30 * 0.4
    )
    # Calibrate roughly to the requested base rate
    logit += np.log(base_rate / (1 - base_rate)) - np.log(
        _sigmoid(logit).mean() / (1 - _sigmoid(logit).mean())
    )
    default_prob = _sigmoid(logit)
    defaulted = (rng.random(n) < default_prob).astype(int)

    return pd.DataFrame(
        {
            "application_id": [f"APP-{i:07d}" for i in range(n)],
            "credit_score": credit_score,
            "annual_income": np.round(annual_income, 2),
            "dti": np.round(dti, 4),
            "loan_amount": np.round(loan_amount, 2),
            "term_months": term_months,
            "employment_years": employment_years,
            "home_ownership": home_ownership,
            "interest_rate": np.round(interest_rate, 4),
            "defaulted": defaulted,
        }
    )


def main() -> None:
    settings = from_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=settings.n_applications)
    parser.add_argument(
        "--out", default=f"{settings.raw_dir}/applications.parquet"
    )
    args = parser.parse_args()

    settings.ensure_dirs()
    df = generate(args.n, settings.default_base_rate, settings.random_seed)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {len(df):,} applications to {args.out}")
    print(f"Default rate: {df['defaulted'].mean():.3f}")


if __name__ == "__main__":
    main()
