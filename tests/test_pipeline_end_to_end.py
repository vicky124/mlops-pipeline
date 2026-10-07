"""Train -> register -> promote -> serve -> monitor, against a throwaway MLflow registry."""

import json

import pandas as pd
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient

from mlops import monitor
from mlops.data import FEATURES, DriftSpec, generate
from mlops.serve import LoadedModel, create_app
from mlops.train import MODEL_NAME, Gates, check_gates, train


def test_gates():
    good = {"roc_auc": 0.8, "brier": 0.15}
    assert check_gates(good, None, Gates()) == []
    assert check_gates(good, 0.803, Gates()) == []  # within the 0.005 tolerance
    assert "regresses champion" in check_gates(good, 0.85, Gates())[0]
    assert len(check_gates({"roc_auc": 0.6, "brier": 0.3}, None, Gates())) == 2


def test_full_lifecycle(tracking, tmp_path):
    df = generate(3_000, seed=3)

    # 1. First model passes the gates and becomes champion.
    first = train(df, n_iter=2)
    assert first.promoted, first.reasons
    assert first.metrics["roc_auc"] > 0.75
    client = MlflowClient()
    assert str(client.get_model_version_by_alias(MODEL_NAME, "champion").version) == first.model_version

    # 2. A challenger trained on label-shuffled data must be rejected; champion is unchanged.
    noisy = df.assign(churned=df["churned"].sample(frac=1, random_state=0).to_numpy())
    second = train(noisy, n_iter=1)
    assert not second.promoted
    assert any("roc_auc" in r for r in second.reasons)
    assert str(client.get_model_version_by_alias(MODEL_NAME, "champion").version) == first.model_version
    assert client.get_model_version(MODEL_NAME, second.model_version).tags["gate"] == "failed"

    # 3. Serve the champion and log predictions.
    log_path = tmp_path / "predictions.jsonl"
    model = LoadedModel(f"models:/{MODEL_NAME}@champion")
    assert 0 < model.threshold < 1
    api = TestClient(create_app(model, prediction_log=str(log_path)))
    assert api.get("/ping").json()["status"] == "ok"

    records = generate(5, seed=11)[FEATURES].to_dict("records")
    res = api.post("/predict", json={"records": records})
    assert res.status_code == 200
    preds = res.json()["predictions"]
    assert len(preds) == 5 and all(0 <= p["churn_probability"] <= 1 for p in preds)

    csv = generate(4, seed=12)[FEATURES].to_csv(index=False)
    res = api.post("/invocations", content=csv, headers={"content-type": "text/csv"})
    assert res.status_code == 200 and len(res.json()["predictions"]) == 4
    bad = api.post("/invocations", content=json.dumps([{"tenure_months": 1}]),
                   headers={"content-type": "application/json"})
    assert bad.status_code == 400
    assert api.post("/predict", json={"records": [records[0] | {"contract": "weekly"}]}).status_code == 422

    # 4. Drifted production traffic trips the monitor (exit code 2 path).
    drifted = generate(1_500, seed=13, drift=DriftSpec(charges_shift=35, m2m_share=0.85, ticket_rate=3))
    for start in range(0, len(drifted), 500):
        chunk = drifted.iloc[start:start + 500][FEATURES].to_dict("records")
        api.post("/predict", json={"records": chunk})
    reference = monitor.load_reference(None, MODEL_NAME)  # fetched from the champion's run
    report = monitor.run(str(log_path), reference, tmp_path / "reports", last=1_500)
    assert report["status"] == "drift"
    assert "monthly_charges" in report["drifted_features"]
    assert (tmp_path / "reports" / "drift_report.html").exists()
    assert report["features"]["__prediction__"]["psi"] > 0.1  # scores shifted too
    assert pd.read_json(log_path, lines=True)["model_version"].nunique() == 1
