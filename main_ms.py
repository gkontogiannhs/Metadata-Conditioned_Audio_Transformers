import os
import mlflow
from ls.config.loader import load_config
from ls.engine.train import train_loop
from ls.data.dataloaders import build_dataloaders
from ls.models.builder import build_model
from ls.engine.utils import set_seed
from ls.engine.logging_utils import get_or_create_experiment, log_all_params


def main_multi_seed(n_seeds: int = 5):
    """
    Run the same experiment for multiple random seeds and log each run separately in MLflow.
    """
    cfg = load_config("configs/config.yaml")
    mlflow_cfg = load_config("configs/mlflow.yaml")
    MODEL_KEY = "ast"

    # Authenticate MLflow
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_cfg.tracking_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_cfg.tracking_password
    mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
    experiment_id = get_or_create_experiment(mlflow_cfg.experiment_name)

    base_seed = cfg.seed if hasattr(cfg, "seed") else 42
    seed_list = [base_seed + i for i in range(n_seeds)]

    for run_idx, seed in enumerate(seed_list, start=1):
        print(f"\n==============================")
        print(f"  Run {run_idx}/{n_seeds}  —  Seed = {seed}")
        print(f"==============================")

        # --- 1. Set seed ---
        set_seed(seed)

        # --- 2. Build data loaders (reinitialized for each seed) ---
        train_loader, val_loader = build_dataloaders(cfg.dataset, cfg.audio)

        # --- 3. Build model fresh each run ---
        model = build_model(cfg.models, model_key=MODEL_KEY)
        print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

        # --- 4. Start MLflow run ---
        run_name = f"{MODEL_KEY}_seed{seed}_{cfg.training.epochs}ep"
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            mlflow.log_param("seed", seed)
            mlflow.log_param("model_key", MODEL_KEY)
            log_all_params(cfg)

            # --- 5. Train ---
            train_loss, metrics = train_loop(
                cfg.training, model, train_loader, val_loader=val_loader, test_loader=val_loader
            )

        print(f"[Done] Seed {seed} finished and logged to MLflow")


if __name__ == "__main__":
    main_multi_seed(n_seeds=5)