import os
import mlflow
from ls.config.loader import load_config
from ls.engine.train import train_loop
from ls.data.dataloaders import build_dataloaders
from ls.models.builder import build_model
from ls.engine.utils import set_seed
from ls.engine.logging_utils import get_or_create_experiment, log_all_params


def main_single():
    cfg = load_config("configs/config.yaml")
    mlflow_cfg = load_config("configs/mlflow.yaml")
    MODEL_KEY = "ast"
    # Set seed for reproducibility
    set_seed(cfg.seed)

    # Build Dataset
    train_loader, test_loader = build_dataloaders(cfg.dataset, cfg.audio)

    # Build Model
    model = build_model(cfg.models, model_key=MODEL_KEY)

    # Authenticate with MLflow
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_cfg.tracking_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_cfg.tracking_password
    # Set the MLflow tracking URI
    mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
    # Start MLFlow experiment
    mlflow.set_experiment(experiment_id=get_or_create_experiment(mlflow_cfg.experiment_name))
    run_name = f"{MODEL_KEY}_{cfg.training.epochs}ep_single"

    with mlflow.start_run(run_name=run_name):

        # Log configuration parameters
        log_all_params(cfg)

        train_loop(cfg.training, model, train_loader, test_loader=test_loader)
        mlflow.end_run()


if __name__ == "__main__":
    main_single()