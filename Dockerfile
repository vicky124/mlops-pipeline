# Serving image. Works for local docker runs and as a SageMaker inference container,
# which starts the image with the argument "serve" and expects port 8080.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1 MODEL_URI=/opt/ml/model
WORKDIR /app

COPY pyproject.toml ./
COPY mlops ./mlops
RUN pip install --no-cache-dir .

COPY docker/serve /usr/local/bin/serve
RUN chmod +x /usr/local/bin/serve && useradd --uid 10001 --create-home app
USER app

EXPOSE 8080
ENTRYPOINT ["serve"]
