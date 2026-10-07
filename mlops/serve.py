"""Model server. Works locally and as a SageMaker inference container (/ping, /invocations).

    MODEL_URI=models:/churn-classifier@champion uvicorn mlops.serve:app_factory --factory --port 8080
    # in SageMaker the model artifact is extracted to /opt/ml/model (the default)
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from mlops.data import FEATURES, SchemaError, validate


class Customer(BaseModel):
    tenure_months: float = Field(ge=0, le=600)
    monthly_charges: float = Field(ge=0)
    support_tickets: int = Field(ge=0)
    usage_gb: float = Field(ge=0)
    contract: Literal["month-to-month", "one-year", "two-year"]
    payment_method: str
    region: str
    has_partner: Literal["yes", "no"]


class PredictRequest(BaseModel):
    records: list[Customer] = Field(min_length=1, max_length=1_000)


class LoadedModel:
    def __init__(self, uri: str):
        import mlflow.pyfunc  # imported lazily: heavy, and tests can inject a fake

        self.uri = uri
        self.pyfunc = mlflow.pyfunc.load_model(uri)
        meta = self.pyfunc.metadata
        self.threshold = float((meta.metadata or {}).get("threshold", 0.5))
        self.version = (meta.metadata or {}).get("run_id", "unknown")

    def predict_proba(self, df: pd.DataFrame):
        return self.pyfunc.predict(df)[:, 1]  # pyfunc_predict_fn="predict_proba"


class PredictionLog:
    """Append-only JSONL of inputs + scores for drift monitoring.
    (In production, swap for Kinesis Firehose -> S3; the monitor reads the same schema.)"""

    def __init__(self, path: str | None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()

    def write(self, df: pd.DataFrame, scores, model_version: str) -> None:
        if not self.path:
            return
        now = time.time()
        lines = [json.dumps({**row, "score": float(s), "ts": now, "model_version": model_version})
                 for row, s in zip(df.to_dict("records"), scores, strict=True)]
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def create_app(model=None, prediction_log: str | None = None) -> FastAPI:
    app = FastAPI(title="Churn model server")
    state = {"model": model}
    plog = PredictionLog(prediction_log if prediction_log is not None else os.environ.get("PREDICTION_LOG"))
    registry = CollectorRegistry()
    predictions = Counter("predictions_total", "Rows scored", ["label"], registry=registry)
    latency = Histogram("predict_latency_seconds", "Scoring latency", registry=registry)

    def get_model():
        if state["model"] is None:
            state["model"] = LoadedModel(os.environ.get("MODEL_URI", "/opt/ml/model"))
        return state["model"]

    def score(df: pd.DataFrame) -> list[dict]:
        m = get_model()
        with latency.time():
            probs = m.predict_proba(df[FEATURES])
        plog.write(df[FEATURES], probs, m.version)
        out = []
        for p in probs:
            label = int(p >= m.threshold)
            predictions.labels(str(label)).inc()
            out.append({"churn_probability": round(float(p), 5), "churn": label})
        return out

    @app.get("/health")
    @app.get("/ping")  # SageMaker health check
    def ping():
        m = get_model()
        return {"status": "ok", "model_version": m.version, "threshold": m.threshold}

    @app.post("/predict")
    def predict(req: PredictRequest):
        df = pd.DataFrame([r.model_dump() for r in req.records])
        return {"predictions": score(df), "model_version": get_model().version}

    @app.post("/invocations")  # SageMaker inference contract: JSON or CSV body
    async def invocations(request: Request):
        body = await request.body()
        ctype = request.headers.get("content-type", "application/json").split(";")[0]
        try:
            if ctype == "text/csv":
                df = pd.read_csv(io.BytesIO(body))
            else:
                payload = json.loads(body)
                records = payload["records"] if isinstance(payload, dict) else payload
                df = pd.DataFrame([Customer(**r).model_dump() for r in records])
            validate(df, require_target=False)
        except (SchemaError, KeyError, ValueError) as exc:
            raise HTTPException(400, f"invalid input: {exc}") from exc
        return {"predictions": score(df)}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


def app_factory() -> FastAPI:
    return create_app()
