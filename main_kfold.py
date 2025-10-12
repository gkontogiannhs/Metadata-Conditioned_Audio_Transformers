from ls.engine.logging_utils import get_or_create_experiment, log_all_params
import mlflow
import numpy as np
import torch
import os

from ls.config.loader import load_config
from ls.data.dataloaders import build_train_val_kfold
from ls.models.builder import build_model
from ls.engine.train import train_loop
from ls.engine.eval import evaluate
from ls.engine.utils import set_seed
from ls.engine.utils import get_device


def main_kfold():
    cfg = load_config("../configs/config.yaml")
    mlflow_cfg = load_config("../configs/mlflow.yaml")
    MODEL_KEY = "ast"
    device = get_device()
    
    # Set seed for reproducibility
    set_seed(cfg.seed)

    # Build K-fold splits
    folds, test_loader = build_train_val_kfold(cfg.dataset, cfg.audio, n_splits=5, seed=cfg.seed)
    n_folds = len(folds)
    fold_icbhi_scores = []

    # Authenticate with MLflow
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_cfg.tracking_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_cfg.tracking_password
    # Set the MLflow tracking URI
    mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
    # Start MLFlow experiment
    mlflow.set_experiment(experiment_id=get_or_create_experiment(mlflow_cfg.experiment_name))
    run_name = f"{MODEL_KEY}_{n_folds}fold_{cfg.training.epochs}ep"

    with mlflow.start_run(run_name=run_name):

        for fold_idx, (train_loader, val_loader) in enumerate(folds, 1):
            print(f"\n===== Fold {fold_idx}/{n_folds} =====")

            # Nested run per fold
            with mlflow.start_run(run_name=f"Fold_{fold_idx}", nested=True):
                # Log configuration parameters
                log_all_params(cfg)

                # Build Model
                model = build_model(cfg.models, model_key=MODEL_KEY)

                # Train the model
                trained_model, criterion = train_loop(
                    cfg.training, model, train_loader, val_loader=val_loader, test_loader=test_loader, fold_idx=fold_idx
                )

                # Evaluate the best model on its validation set
                _, val_metrics = evaluate(
                    trained_model, val_loader,
                    criterion=criterion,
                    device=device
                )
                fold_icbhi_scores.append(val_metrics["icbhi_score"])

        # Aggregate results across folds
        mean_icbhi = float(np.mean(fold_icbhi_scores))
        std_icbhi = float(np.std(fold_icbhi_scores))
        mlflow.log_metrics({"icbhi_mean": mean_icbhi, "icbhi_std": std_icbhi})
        print(f"\ K-Fold Mean ICBHI Score: {mean_icbhi:.2f} ± {std_icbhi:.2f}")

        mlflow.end_run()


if __name__ == "__main__":
    main_kfold()