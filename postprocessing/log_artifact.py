"""Logs a local file or directory as an artifact on a new MLflow run."""

import argparse

import mlflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=None, help="Local path to the file or directory to log")
    parser.add_argument("--experiment", default=None, help="MLflow experiment name")
    parser.add_argument("--run-id", default=None, help="Log into this existing run instead of creating a new one")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--parent-run-id", default=None)
    parser.add_argument("--artifact-path", default=None, help="Subdirectory inside the run's artifact root")
    parser.add_argument("--metric", nargs=2, metavar=("KEY", "VALUE"), action="append", default=[], help="Log a metric (repeatable: --metric key1 val1 --metric key2 val2)")
    args = parser.parse_args()

    if args.experiment:
        mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_id=args.run_id, run_name=args.run_name) as run:
        if args.parent_run_id:
            mlflow.set_tag("mlflow.parentRunId", args.parent_run_id)
        for key, value in args.metric:
            mlflow.log_metric(key, float(value))
        if args.path:
            mlflow.log_artifact(args.path, artifact_path=args.artifact_path)
        print("Run ID:", run.info.run_id)


if __name__ == "__main__":
    main()
