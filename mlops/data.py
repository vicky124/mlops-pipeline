"""Synthetic customer-churn data with controllable drift, plus schema validation.

The generator stands in for a feature-store extract. It has a known ground truth, so tests
and demos can inject *covariate drift* (input distributions move) and *concept drift* (the
relationship between features and churn changes) on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NUMERIC = ["tenure_months", "monthly_charges", "support_tickets", "usage_gb"]
CATEGORICAL = ["contract", "payment_method", "region", "has_partner"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "churned"

CATEGORIES = {
    "contract": ["month-to-month", "one-year", "two-year"],
    "payment_method": ["card", "bank-transfer", "e-check", "mailed-check"],
    "region": ["north", "south", "east", "west"],
    "has_partner": ["yes", "no"],
}


@dataclass
class DriftSpec:
    charges_shift: float = 0.0     # added to monthly charges (price increase)
    m2m_share: float = 0.55        # share of month-to-month contracts
    ticket_rate: float = 1.2       # mean support tickets per customer
    concept_shift: float = 0.0     # >0 makes support tickets matter more for churn


def generate(n: int = 10_000, seed: int = 7, drift: DriftSpec | None = None) -> pd.DataFrame:
    d = drift or DriftSpec()
    rng = np.random.default_rng(seed)
    rest = (1 - d.m2m_share) / 2
    contract = rng.choice(CATEGORIES["contract"], n, p=[d.m2m_share, rest, rest])
    tenure = np.where(
        contract == "month-to-month", rng.gamma(2.0, 8.0, n), rng.gamma(4.0, 9.0, n)
    ).clip(0, 72).round()
    charges = (rng.normal(65, 22, n) + d.charges_shift).clip(18, 150).round(2)
    tickets = rng.poisson(d.ticket_rate, n)
    usage = rng.lognormal(3.2, 0.7, n).round(1)
    payment = rng.choice(CATEGORIES["payment_method"], n, p=[0.35, 0.25, 0.3, 0.1])
    region = rng.choice(CATEGORIES["region"], n)
    partner = rng.choice(CATEGORIES["has_partner"], n, p=[0.45, 0.55])

    logit = (
        -1.6
        + 1.3 * (contract == "month-to-month")
        - 0.5 * (contract == "two-year")
        - 0.045 * tenure
        + 0.018 * (charges - 65)
        + (0.38 + d.concept_shift) * tickets
        + 0.45 * (payment == "e-check")
        - 0.25 * (partner == "yes")
        - 0.004 * (usage - 25)
    )
    churned = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({
        "tenure_months": tenure, "monthly_charges": charges, "support_tickets": tickets,
        "usage_gb": usage, "contract": contract, "payment_method": payment, "region": region,
        "has_partner": partner, TARGET: churned.astype(int),
    })


class SchemaError(ValueError):
    pass


def validate(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Fail fast on the data problems that silently wreck a model."""
    expected = FEATURES + ([TARGET] if require_target else [])
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    problems = []
    for col in NUMERIC:
        if not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"{col} is not numeric")
        elif (df[col] < 0).any():
            problems.append(f"{col} has negative values")
    for col in FEATURES:
        null_share = df[col].isna().mean()
        if null_share > 0.05:
            problems.append(f"{col} is {null_share:.0%} null")
    if require_target:
        if not set(df[TARGET].unique()) <= {0, 1}:
            problems.append(f"{TARGET} must be 0/1")
        elif df[TARGET].nunique() < 2:
            problems.append(f"{TARGET} has a single class")
    if problems:
        raise SchemaError("; ".join(problems))
    return df
