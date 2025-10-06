import mlflow

def init_mlflow(cfg, run_name):
    # mlflow.set_experiment(cfg.experiment.name)
    # mlflow.start_run(run_name=run_name)
    
    # mlflow.log_params({
    #     "model": cfg.model.name,
    #     "epochs": cfg.training.epochs,
    #     "batch_size": cfg.dataset.batch_size,
    #     "optimizer": cfg.optimizer.name,
    #     "loss": cfg.loss.name,
    #     "lr": cfg.training.lr,
    #     "dataset": cfg.dataset.class_split
    # })
    pass

def get_or_create_experiment(experiment_name: str) -> str:
    """Retrieves the ID of an existing MLflow experiment or create a new one if it doesn"t exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)