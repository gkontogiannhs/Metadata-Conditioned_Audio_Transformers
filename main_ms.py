import os
import mlflow
from ls.config.loader import load_config
from ls.engine.train import train_loop
from ls.data.dataloaders import build_dataloaders
from ls.models.builder import build_model
from ls.engine.utils import set_seed
from ls.engine.logging_utils import get_or_create_experiment, log_all_params


def main_multi_seed(n_seeds: int = 5):
    cfg = load_config("configs/config.yaml")
    mlflow_cfg = load_config("configs/mlflow.yaml")
    MODEL_KEY = "ast"

    # -------------------------
    # Explicit MLflow setup
    # -------------------------
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_cfg.tracking_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_cfg.tracking_password
    mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)

    experiment_id = get_or_create_experiment(mlflow_cfg.experiment_name)

    base_seed = getattr(cfg, "seed", 42)
    seed_list = [base_seed + i for i in range(n_seeds)]

    # ================================================
    # PARENT RUN (keeps context for nested=True)
    # ================================================
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"{MODEL_KEY}_multi_seed_{n_seeds}runs",
    ) as parent_run:

        mlflow.log_param("n_seeds", n_seeds)
        mlflow.log_param("base_seed", base_seed)
        mlflow.log_param("model_key", MODEL_KEY)

        # store parent run info
        parent_run_id = parent_run.info.run_id

        for run_idx, seed in enumerate(seed_list, start=1):
            print(f"\n========== Run {run_idx}/{n_seeds} — Seed = {seed} ==========")
            set_seed(seed)
            train_loader, val_loader = build_dataloaders(cfg.dataset, cfg.audio)
            model = build_model(cfg.models, model_key=MODEL_KEY)

            # -----------------------------------------------
            # Nsested sub-run (inherits experiment)
            # -----------------------------------------------
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=f"{MODEL_KEY}_seed{seed}_{cfg.training.epochs}ep",
                nested=True,
            ):
                mlflow.set_tag("parent_run_id", parent_run_id)
                # mlflow.log_param("seed", seed)
                log_all_params(cfg)

                train_loss, metrics = train_loop(
                    cfg.training,
                    model,
                    train_loader,
                    val_loader=val_loader,
                    test_loader=val_loader,
                )

            print(f"[Done] Seed {seed} finished and logged to MLflow")

    print("All seeds complete — results stored in MLflow.")


if __name__ == "__main__":
    main_multi_seed(n_seeds=5)