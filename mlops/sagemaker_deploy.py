"""Deploy the champion model to a SageMaker real-time endpoint with canary rollout.

    python -m mlops.sagemaker_deploy --image <acct>.dkr.ecr.<region>.amazonaws.com/churn-serve:latest \
        --bucket my-ml-artifacts --role arn:aws:iam::<acct>:role/SageMakerExecution

1. Download models:/churn-classifier@champion and package it as model.tar.gz in S3.
2. Create a SageMaker Model (our serving image + that artifact), plus an endpoint config.
3. Create the endpoint, or update it blue/green: 10% canary traffic for 10 minutes, with
   automatic rollback if the 5xx alarm fires.
"""

from __future__ import annotations

import argparse
import tarfile
import tempfile
import time
from pathlib import Path

import boto3


def package_model(model_uri: str, workdir: Path) -> Path:
    import mlflow

    local = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=str(workdir / "model")))
    archive = workdir / "model.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for item in local.iterdir():  # SageMaker extracts the archive into /opt/ml/model
            tar.add(item, arcname=item.name)
    return archive


def ensure_rollback_alarm(cloudwatch, endpoint: str) -> str:
    name = f"{endpoint}-5xx"
    cloudwatch.put_metric_alarm(
        AlarmName=name, Namespace="AWS/SageMaker", MetricName="Invocation5XXErrors",
        Dimensions=[{"Name": "EndpointName", "Value": endpoint}, {"Name": "VariantName", "Value": "primary"}],
        Statistic="Sum", Period=60, EvaluationPeriods=2, Threshold=5,
        ComparisonOperator="GreaterThanOrEqualToThreshold", TreatMissingData="notBreaching",
    )
    return name


def deploy(*, model_uri: str, image: str, bucket: str, role: str, endpoint: str,
           instance_type: str, instance_count: int, region: str | None) -> str:
    session = boto3.session.Session(region_name=region)
    sm, s3, cw = session.client("sagemaker"), session.client("s3"), session.client("cloudwatch")
    stamp = time.strftime("%Y%m%d-%H%M%S")

    with tempfile.TemporaryDirectory() as tmp:
        archive = package_model(model_uri, Path(tmp))
        key = f"{endpoint}/{stamp}/model.tar.gz"
        s3.upload_file(str(archive), bucket, key)

    model_name = f"{endpoint}-{stamp}"
    sm.create_model(
        ModelName=model_name, ExecutionRoleArn=role,
        PrimaryContainer={"Image": image, "ModelDataUrl": f"s3://{bucket}/{key}",
                          "Environment": {"MODEL_URI": "/opt/ml/model"}},
    )
    config_name = f"{endpoint}-{stamp}"
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{"VariantName": "primary", "ModelName": model_name,
                             "InstanceType": instance_type, "InitialInstanceCount": instance_count}],
        # Capture a sample of live requests/responses to S3 for monitoring and retraining.
        DataCaptureConfig={"EnableCapture": True, "InitialSamplingPercentage": 20,
                           "DestinationS3Uri": f"s3://{bucket}/{endpoint}/capture",
                           "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}]},
    )

    try:
        sm.describe_endpoint(EndpointName=endpoint)
        exists = True
    except sm.exceptions.ClientError:
        exists = False
    if not exists:
        sm.create_endpoint(EndpointName=endpoint, EndpointConfigName=config_name)
    else:
        alarm = ensure_rollback_alarm(cw, endpoint)
        sm.update_endpoint(
            EndpointName=endpoint, EndpointConfigName=config_name,
            DeploymentConfig={
                "BlueGreenUpdatePolicy": {
                    "TrafficRoutingConfiguration": {
                        "Type": "CANARY", "WaitIntervalInSeconds": 600,
                        "CanarySize": {"Type": "CAPACITY_PERCENT", "Value": 10},
                    },
                    "TerminationWaitInSeconds": 300,
                },
                "AutoRollbackConfiguration": {"Alarms": [{"AlarmName": alarm}]},
            },
        )
    sm.get_waiter("endpoint_in_service").wait(EndpointName=endpoint)
    return endpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-uri", default="models:/churn-classifier@champion")
    p.add_argument("--image", required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--role", required=True)
    p.add_argument("--endpoint", default="churn-classifier")
    p.add_argument("--instance-type", default="ml.m5.large")
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument("--region")
    args = p.parse_args()
    name = deploy(model_uri=args.model_uri, image=args.image, bucket=args.bucket, role=args.role,
                  endpoint=args.endpoint, instance_type=args.instance_type,
                  instance_count=args.instance_count, region=args.region)
    print(f"endpoint {name} is InService")


if __name__ == "__main__":
    main()
