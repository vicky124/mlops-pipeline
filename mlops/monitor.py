"""Compare recent production inputs against the champion's training profile.

    python -m mlops.monitor --log predictions.jsonl --reference reference_profile.json
    python -m mlops.monitor --log predictions.jsonl          # fetch profile of models:/...@champion

Exit codes: 0 ok, 1 warn, 2 drift (a scheduled CI job can trigger retraining on 2).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from mlops.data import FEATURES
from mlops.drift import compare, write_html

EXIT = {"ok": 0, "warn": 1, "drift": 2}


def load_reference(path: str | None, model_name: str) -> dict:
    if path:
        return json.loads(Path(path).read_text())
    import mlflow
    from mlflow.tracking import MlflowClient

    version = MlflowClient().get_model_version_by_alias(model_name, "champion")
    local = mlflow.artifacts.download_artifacts(run_id=version.run_id, artifact_path="reference_profile.json")
    return json.loads(Path(local).read_text())


def load_log(path: str, last: int | None) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    return df.tail(last) if last else df


def run(log_path: str, reference: dict, out_dir: Path, last: int | None = None, min_rows: int = 200) -> dict:
    current = load_log(log_path, last)
    if len(current) < min_rows:
        report = {"status": "ok", "rows": len(current), "drifted_features": [], "drift_share": 0.0,
                  "features": {}, "note": f"fewer than {min_rows} rows; not enough evidence"}
    else:
        report = compare(reference, current[FEATURES], current["score"].to_numpy())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "drift_report.json").write_text(json.dumps(report, indent=2))
    if report["features"]:
        write_html(report, out_dir / "drift_report.html")
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, help="prediction log (JSONL)")
    p.add_argument("--reference", help="reference_profile.json (default: champion's artifact)")
    p.add_argument("--model-name", default="churn-classifier")
    p.add_argument("--last", type=int, help="only the most recent N predictions")
    p.add_argument("--out", default="reports")
    args = p.parse_args()

    report = run(args.log, load_reference(args.reference, args.model_name), Path(args.out), args.last)
    print(f"status={report['status']} rows={report['rows']} drifted={report['drifted_features']}")
    for name, f in sorted(report["features"].items(), key=lambda kv: -kv[1]["psi"])[:5]:
        print(f"  {name:<18} psi={f['psi']:.3f} {f['test']} p={f['p_value']:.2g} {f['status']}")
    sys.exit(EXIT[report["status"]])


if __name__ == "__main__":
    main()
