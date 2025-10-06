import mlflow
import os
from ls.config.dataclasses import MLflowConfig

def setup_mlflow(cfg: MLflowConfig) -> str:
    """
    Initialize MLflow remote connection and prepare the experiment.

    Args:
        cfg: Full configuration object containing mlflow tracking info.

    Returns:
        experiment_id: ID of the active MLflow experiment.
    """
    # Authenticate with MLflow
    os.environ["MLFLOW_TRACKING_USERNAME"] = cfg.mlflow.tracking_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = cfg.mlflow.tracking_password

    # Set the MLflow tracking URI (remote or local)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    # Ensure experiment exists (create if needed)
    experiment_id = get_or_create_experiment(cfg.mlflow.experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    return experiment_id


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
    

def flatten_dict(d, parent_key='', sep='.'):
    """
    Recursively flatten nested dictionaries for MLflow logging.
    Example: {"a": {"b": 1}} -> {"a.b": 1}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dataclass_to_dict(dc):
    """Convert a dataclass or nested structure to pure Python dict."""
    from dataclasses import asdict, is_dataclass
    if is_dataclass(dc):
        return {k: dataclass_to_dict(v) for k, v in asdict(dc).items()}
    elif isinstance(dc, dict):
        return {k: dataclass_to_dict(v) for k, v in dc.items()}
    else:
        return dc


def log_all_params(cfg):
    """
    Log all configuration parameters from the dataclass-based config to MLflow.

    Args:
        cfg: Configuration dataclass (fully loaded)
    """
    cfg_dict = dataclass_to_dict(cfg)
    flat_cfg = flatten_dict(cfg_dict)

    for k, v in flat_cfg.items():
        if isinstance(v, str) and len(v) > 500:
            print(f"Truncating long string parameter {k} for MLflow logging.")
            flat_cfg[k] = v[:500] + "..."
        elif isinstance(v, (list, tuple)):
            flat_cfg[k] = str(v)

    mlflow.log_params(flat_cfg)