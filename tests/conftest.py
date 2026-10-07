import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    """Isolated MLflow tracking + registry per test."""
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.chdir(tmp_path)  # artifacts default to ./mlruns under the cwd
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    yield uri
    mlflow.set_tracking_uri(None)
