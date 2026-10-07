"""Feature pipeline, candidate models, and evaluation (incl. cost-based thresholding)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from mlops.data import CATEGORICAL, NUMERIC


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("skewed", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                             ("scale", StandardScaler())]), ["usage_gb", "support_tickets"]),
        ("numeric", StandardScaler(), [c for c in NUMERIC if c not in ("usage_gb", "support_tickets")]),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


CANDIDATES = {
    "logreg": (
        LogisticRegression(max_iter=2_000),
        {"model__C": loguniform(1e-3, 10), "model__class_weight": [None, "balanced"]},
    ),
    "hist_gbm": (
        HistGradientBoostingClassifier(random_state=0),
        {"model__learning_rate": loguniform(0.02, 0.3), "model__max_depth": randint(2, 7),
         "model__l2_regularization": uniform(0, 1.0), "model__max_iter": randint(100, 400)},
    ),
}


def search(name: str, X: pd.DataFrame, y: pd.Series, n_iter: int, seed: int) -> RandomizedSearchCV:
    estimator, space = CANDIDATES[name]
    pipe = Pipeline([("prep", preprocessor()), ("model", estimator)])
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    return RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="roc_auc", cv=cv,
                              random_state=seed, n_jobs=-1, refit=True).fit(X, y)


@dataclass
class CostModel:
    """Business costs used to pick the decision threshold, in arbitrary currency units."""

    false_negative: float = 5.0  # a churner we missed (lost customer lifetime value)
    false_positive: float = 1.0  # a retention offer sent to someone who'd have stayed


def best_threshold(y: np.ndarray, scores: np.ndarray, cost: CostModel) -> tuple[float, float]:
    """Threshold that minimises expected cost per customer."""
    grid = np.linspace(0.02, 0.98, 97)
    costs = [
        (cost.false_negative * ((scores < t) & (y == 1)).sum()
         + cost.false_positive * ((scores >= t) & (y == 0)).sum()) / len(y)
        for t in grid
    ]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def evaluate(y: np.ndarray, scores: np.ndarray, threshold: float, cost: CostModel) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    fn = ((pred == 0) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    return {
        "roc_auc": roc_auc_score(y, scores),
        "pr_auc": average_precision_score(y, scores),
        "brier": brier_score_loss(y, scores),
        "log_loss": log_loss(y, scores),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "expected_cost": float((cost.false_negative * fn + cost.false_positive * fp) / len(y)),
        "threshold": threshold,
    }


def slice_metrics(df: pd.DataFrame, y: np.ndarray, scores: np.ndarray, column: str) -> dict[str, float]:
    """AUC per segment: a model can look fine overall and fail one customer group."""
    out = {}
    for value in sorted(df[column].unique()):
        mask = (df[column] == value).to_numpy()
        if len(np.unique(y[mask])) == 2:
            out[f"auc_{column}_{value}"] = roc_auc_score(y[mask], scores[mask])
    return out
