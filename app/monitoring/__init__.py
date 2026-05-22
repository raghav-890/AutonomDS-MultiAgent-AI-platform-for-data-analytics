"""
MLflow Monitoring Module
=========================
Tracks experiments, metrics, hyperparameters, and artifacts using
MLflow's local tracking server (fully free, no cloud needed).

Usage:
    tracker = MLflowTracker()
    with tracker.start_run(experiment_id, run_name) as run_id:
        tracker.log_params({"lr": 0.01, "n_estimators": 100})
        tracker.log_metrics({"accuracy": 0.92, "f1": 0.89})
        tracker.log_artifact(model_path)
"""
