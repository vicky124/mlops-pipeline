"""Data and prediction drift: PSI plus KS (numeric) / chi-square (categorical) tests.

`profile()` runs at training time and is stored with the model. `compare()` runs later
against production inputs. Keeping the reference with the model means monitoring always
compares against the data the *serving* model was trained on.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chisquare, ks_2samp

from mlops.data import CATEGORICAL, NUMERIC

PSI_WARN, PSI_DRIFT = 0.1, 0.25
P_VALUE = 0.01
EPS = 1e-6
REFERENCE_SAMPLE = 2_000


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, float), EPS, None)
    a = np.clip(np.asarray(actual, float), EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def _numeric_bins(values: np.ndarray, bins: int = 10) -> list[float]:
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    return [-np.inf, *edges[1:-1].tolist(), np.inf]


def _proportions(values: np.ndarray, edges: list[float]) -> list[float]:
    counts, _ = np.histogram(values, bins=edges)
    return (counts / max(1, counts.sum())).tolist()


def profile(df: pd.DataFrame, scores: np.ndarray | None = None, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out: dict = {"rows": len(df), "numeric": {}, "categorical": {}}
    columns = {c: df[c].to_numpy(float) for c in NUMERIC}
    if scores is not None:
        columns["__prediction__"] = np.asarray(scores, float)
    for col, values in columns.items():
        edges = _numeric_bins(values)
        sample = rng.choice(values, min(REFERENCE_SAMPLE, len(values)), replace=False)
        out["numeric"][col] = {
            "edges": [e if np.isfinite(e) else None for e in edges],
            "proportions": _proportions(values, edges),
            "mean": float(values.mean()), "std": float(values.std()),
            "sample": sample.round(4).tolist(),
        }
    for col in CATEGORICAL:
        out["categorical"][col] = df[col].astype(str).value_counts(normalize=True).to_dict()
    return out


def _status(psi_value: float, p_value: float) -> str:
    # Both must agree for "drift": PSI catches meaningful shifts, the test guards against noise.
    if psi_value >= PSI_DRIFT and p_value < P_VALUE:
        return "drift"
    if psi_value >= PSI_WARN:
        return "warn"
    return "ok"


def compare(reference: dict, current: pd.DataFrame, scores: np.ndarray | None = None) -> dict:
    features: dict[str, dict] = {}
    columns = {c: current[c].to_numpy(float) for c in NUMERIC}
    if scores is not None and "__prediction__" in reference["numeric"]:
        columns["__prediction__"] = np.asarray(scores, float)
    for col, values in columns.items():
        ref = reference["numeric"][col]
        edges = [(-np.inf if i == 0 else np.inf) if e is None else e for i, e in enumerate(ref["edges"])]
        value = psi(ref["proportions"], _proportions(values, edges))
        p = float(ks_2samp(ref["sample"], values).pvalue)
        features[col] = {"type": "numeric", "psi": value, "test": "ks", "p_value": p,
                         "ref_mean": ref["mean"], "cur_mean": float(values.mean()),
                         "status": _status(value, p)}
    for col in CATEGORICAL:
        ref_props: dict[str, float] = reference["categorical"][col]
        counts = current[col].astype(str).value_counts()
        cats = sorted(set(ref_props) | set(counts.index))
        expected = np.array([ref_props.get(c, 0.0) for c in cats])
        observed = np.array([counts.get(c, 0) for c in cats], float)
        value = psi(expected, observed / max(1, observed.sum()))
        exp_counts = np.clip(expected, EPS, None)
        exp_counts = exp_counts / exp_counts.sum() * observed.sum()
        p = float(chisquare(observed, exp_counts).pvalue)
        unseen = [c for c in cats if c not in ref_props]
        features[col] = {"type": "categorical", "psi": value, "test": "chi2", "p_value": p,
                         "unseen_categories": unseen, "status": _status(value, p)}

    inputs = {k: v for k, v in features.items() if k != "__prediction__"}
    drifted = [k for k, v in inputs.items() if v["status"] == "drift"]
    share = len(drifted) / len(inputs)
    overall = "drift" if share >= 0.25 or features.get("__prediction__", {}).get("status") == "drift" \
        else "warn" if drifted or any(v["status"] == "warn" for v in features.values()) else "ok"
    return {"status": overall, "rows": len(current), "drifted_features": drifted,
            "drift_share": share, "features": features}


def write_html(report: dict, path: Path) -> None:
    colors = {"ok": "#2e7d32", "warn": "#b26a00", "drift": "#c62828"}
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{f['type']}</td><td>{f['psi']:.3f}</td>"
        f"<td>{f['test']} p={f['p_value']:.2g}</td>"
        f"<td style='color:{colors[f['status']]};font-weight:600'>{f['status']}</td></tr>"
        for name, f in sorted(report["features"].items(), key=lambda kv: -kv[1]["psi"])
    )
    path.write_text(f"""<!doctype html><meta charset="utf-8"><title>Drift report</title>
<style>body{{font:14px system-ui;margin:24px}}
td,th{{padding:6px 12px;border-bottom:1px solid #ddd;text-align:left}}</style>
<h1 style="color:{colors[report['status']]}">Drift status: {report['status']}</h1>
<p>{report['rows']} rows compared · drifted features: {', '.join(report['drifted_features']) or 'none'}</p>
<table><tr><th>Feature</th><th>Type</th><th>PSI</th><th>Test</th><th>Status</th></tr>{rows}</table>
<pre>{html.escape(json.dumps({k: v for k, v in report.items() if k != 'features'}, indent=2))}</pre>""",
                    encoding="utf-8")
