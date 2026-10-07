import numpy as np
import pandas as pd
import pytest

from mlops.data import FEATURES, DriftSpec, SchemaError, generate, validate
from mlops.drift import compare, profile, psi
from mlops.modeling import CostModel, best_threshold, evaluate, slice_metrics


def test_generator_is_deterministic_and_valid():
    a, b = generate(500, seed=1), generate(500, seed=1)
    pd.testing.assert_frame_equal(a, b)
    validate(a)
    assert 0.15 < a["churned"].mean() < 0.5


def test_drift_spec_moves_distributions():
    base = generate(5_000, seed=1)
    drifted = generate(5_000, seed=1, drift=DriftSpec(charges_shift=30, m2m_share=0.8))
    assert drifted["monthly_charges"].mean() - base["monthly_charges"].mean() > 25
    assert (drifted["contract"] == "month-to-month").mean() > 0.75


@pytest.mark.parametrize(("mutate", "message"), [
    (lambda df: df.drop(columns=["region"]), "missing columns"),
    (lambda df: df.assign(tenure_months=-1), "negative"),
    (lambda df: df.assign(churned=0), "single class"),
    (lambda df: df.assign(usage_gb=np.where(np.arange(len(df)) % 5 == 0, np.nan, df["usage_gb"])), "null"),
])
def test_validation_catches_bad_data(mutate, message):
    with pytest.raises(SchemaError, match=message):
        validate(mutate(generate(200)))


def test_psi_basics():
    assert psi([0.25] * 4, [0.25] * 4) == pytest.approx(0)
    assert psi([0.25] * 4, [0.7, 0.1, 0.1, 0.1]) > 0.25


def test_no_drift_on_same_distribution():
    reference = profile(generate(8_000, seed=1)[FEATURES])
    report = compare(reference, generate(3_000, seed=2)[FEATURES])
    assert report["status"] == "ok", report["drifted_features"]


def test_covariate_drift_detected_on_the_right_features():
    reference = profile(generate(8_000, seed=1)[FEATURES])
    shifted = generate(3_000, seed=2, drift=DriftSpec(charges_shift=35, m2m_share=0.85, ticket_rate=3))
    report = compare(reference, shifted[FEATURES])
    assert report["status"] == "drift"
    assert {"monthly_charges", "contract", "support_tickets"} <= set(report["drifted_features"])
    assert "region" not in report["drifted_features"]


def test_unseen_categories_are_reported():
    reference = profile(generate(2_000)[FEATURES])
    current = generate(500)[FEATURES].assign(payment_method="crypto")
    assert compare(reference, current)["features"]["payment_method"]["unseen_categories"] == ["crypto"]


def test_threshold_minimises_expected_cost():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2_000)
    scores = np.clip(y * 0.3 + rng.random(2_000) * 0.7, 0, 1)
    t_cheap_fn, _ = best_threshold(y, scores, CostModel(false_negative=1, false_positive=5))
    t_costly_fn, _ = best_threshold(y, scores, CostModel(false_negative=5, false_positive=1))
    assert t_costly_fn < t_cheap_fn  # expensive misses -> flag more customers
    metrics = evaluate(y, scores, t_costly_fn, CostModel())
    assert metrics["recall"] > metrics["precision"]


def test_slice_metrics_per_segment():
    df = generate(2_000)
    out = slice_metrics(df, df["churned"].to_numpy(), df["support_tickets"].to_numpy() / 10, "contract")
    assert set(out) == {"auc_contract_month-to-month", "auc_contract_one-year", "auc_contract_two-year"}
