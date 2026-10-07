"""Training pipeline: validate -> search -> evaluate -> quality gates -> register -> promote.

    python -m mlops.train                       # synthetic data, default settings
    python -m mlops.train --data data/churn.csv --n-iter 20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict, train_test_split

from mlops import data as data_mod
from mlops.drift import profile
from mlops.modeling import CANDIDATES, CostModel, best_threshold, evaluate, search, slice_metrics

log = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("MODEL_NAME", "churn-classifier")
CHAMPION = "champion"
SKOPS_TRUSTED = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


@dataclass
class Gates:
    min_roc_auc: float = 0.75
    max_brier: float = 0.20
    # A challenger must not be meaningfully worse than the champion on the same holdout.
    max_auc_regression: float = 0.005


@dataclass
class TrainResult:
    run_id: str
    model_version: str
    metrics: dict[str, float]
    promoted: bool
    reasons: list[str] = field(default_factory=list)


def check_gates(metrics: dict[str, float], champion_auc: float | None, gates: Gates) -> list[str]:
    failures = []
    if metrics["roc_auc"] < gates.min_roc_auc:
        failures.append(f"roc_auc {metrics['roc_auc']:.4f} < {gates.min_roc_auc}")
    if metrics["brier"] > gates.max_brier:
        failures.append(f"brier {metrics['brier']:.4f} > {gates.max_brier}")
    if champion_auc is not None and metrics["roc_auc"] < champion_auc - gates.max_auc_regression:
        failures.append(f"roc_auc {metrics['roc_auc']:.4f} regresses champion {champion_auc:.4f}")
    return failures


def champion_auc_on(X: pd.DataFrame, y: pd.Series, client: MlflowClient) -> float | None:
    """Re-score the current champion on *this* holdout so the comparison is apples to apples."""
    try:
        client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
    except MlflowException:
        return None
    from sklearn.metrics import roc_auc_score

    champion = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION}")
    return float(roc_auc_score(y, champion.predict_proba(X)[:, 1]))


def train(df: pd.DataFrame, *, n_iter: int = 10, seed: int = 42, gates: Gates | None = None,
          cost: CostModel | None = None, experiment: str = "churn") -> TrainResult:
    gates, cost = gates or Gates(), cost or CostModel()
    data_mod.validate(df)
    X, y = df[data_mod.FEATURES], df[data_mod.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
                                                        random_state=seed)
    mlflow.set_experiment(experiment)
    client = MlflowClient()

    with mlflow.start_run(run_name="train") as run:
        mlflow.log_params({"rows": len(df), "positive_rate": round(float(y.mean()), 4),
                           "n_iter": n_iter, "seed": seed, **{f"cost_{k}": v for k, v in asdict(cost).items()}})
        best_name, best_search = None, None
        for name in CANDIDATES:
            with mlflow.start_run(run_name=name, nested=True):
                result = search(name, X_train, y_train, n_iter, seed)
                mlflow.log_params({k.removeprefix("model__"): v for k, v in result.best_params_.items()})
                mlflow.log_metric("cv_roc_auc", result.best_score_)
                log.info("%s cv_roc_auc=%.4f", name, result.best_score_)
            if best_search is None or result.best_score_ > best_search.best_score_:
                best_name, best_search = name, result

        model = best_search.best_estimator_
        scores = model.predict_proba(X_test)[:, 1]
        # Tune the threshold on out-of-fold training predictions: in-sample scores are
        # over-confident, and tuning on X_test would leak the holdout into the model.
        oof = cross_val_predict(clone(model), X_train, y_train, cv=4, method="predict_proba")[:, 1]
        threshold, _ = best_threshold(y_train.to_numpy(), oof, cost)
        metrics = evaluate(y_test.to_numpy(), scores, threshold, cost)
        metrics |= slice_metrics(X_test, y_test.to_numpy(), scores, "contract")
        mlflow.log_metrics(metrics)
        mlflow.set_tag("best_model", best_name)

        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "reference_profile.json"
            ref.write_text(json.dumps(profile(X_train, model.predict_proba(X_train)[:, 1])))
            mlflow.log_artifact(str(ref))

        info = mlflow.sklearn.log_model(
            model, name="model", registered_model_name=MODEL_NAME,
            signature=infer_signature(X_test, scores), input_example=X_test.head(3),
            pyfunc_predict_fn="predict_proba",
            # MLflow serializes with skops, which refuses unreviewed types on load. We trained
            # this model ourselves, so we trust exactly the one type HistGradientBoosting needs.
            skops_trusted_types=SKOPS_TRUSTED,
            metadata={"threshold": threshold, "run_id": run.info.run_id, "model_family": best_name},
        )
        version = str(info.registered_model_version)

        champion_auc = champion_auc_on(X_test, y_test, client)
        if champion_auc is not None:
            mlflow.log_metric("champion_roc_auc_same_holdout", champion_auc)
        failures = check_gates(metrics, champion_auc, gates)
        promoted = not failures
        if promoted:
            client.set_registered_model_alias(MODEL_NAME, CHAMPION, version)
        client.set_model_version_tag(MODEL_NAME, version, "gate", "passed" if promoted else "failed")
        if failures:
            client.set_model_version_tag(MODEL_NAME, version, "gate_failures", "; ".join(failures))
        mlflow.set_tag("promoted", str(promoted))

    return TrainResult(run.info.run_id, version, metrics, promoted, failures)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data", help="CSV with features + 'churned' (default: synthetic)")
    p.add_argument("--rows", type=int, default=10_000)
    p.add_argument("--n-iter", type=int, default=10)
    p.add_argument("--min-auc", type=float, default=Gates.min_roc_auc)
    args = p.parse_args()

    df = pd.read_csv(args.data) if args.data else data_mod.generate(args.rows)
    result = train(df, n_iter=args.n_iter, gates=Gates(min_roc_auc=args.min_auc))
    print(json.dumps(asdict(result), indent=2, default=float))
    raise SystemExit(0 if result.promoted else 3)  # non-zero lets CI flag a rejected model


if __name__ == "__main__":
    main()
