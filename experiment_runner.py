"""
Modular CLI-based training system for FiLM-AST research.
All experiments configurable from command line.

Usage Examples:
    # Train vanilla AST
    python main.py --model ast --name baseline_vanilla
    
    # Train FiLM with all metadata
    python main.py --model ast_film --name film_full
    
    # Train FiLM with only device metadata
    python main.py --model ast_film --name film_device_only \
        --film-device --no-film-site --no-film-continuous
    
    # Run ablation study
    python main.py --experiment ablation
    
    # Run all experiments
    python main.py --experiment all
    
    # Visualize trained model
    python main.py --visualize checkpoints/best_model.pt
"""

import os
import sys
import argparse
import mlflow
import torch
from pathlib import Path

from ls.config.loader import load_config
from ls.data.dataloaders import build_dataloaders
from ls.models.builder import build_model
from ls.engine.utils import set_seed
from ls.engine.logging_utils import get_or_create_experiment, log_all_params
from ls.engine.train import train_loop


# ============================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================

def get_experiment_configs():
    """
    Define all research experiments.
    Returns dict of experiment name -> configuration.
    """
    
    base_film = {
        "dev_emb_dim": 4,
        "site_emb_dim": 4,
        "metadata_hidden_dim": 64,
        "film_hidden_dim": 64,
        "use_improved_continuous_encoder": True,
        "layer_specific_encoding": False,
        "debug_film": False,
    }
    
    experiments = {
        # ===== BASELINES =====
        "baseline_vanilla": {
            "model": "ast",
            "description": "Vanilla AST without conditioning"
        },
        
        "baseline_film_full": {
            "model": "ast_film",
            "description": "FiLM-AST with all metadata",
            "film_config": {
                **base_film,
                "condition_on_device": True,
                "condition_on_site": True,
                "condition_on_rest": True,
                "conditioned_layers": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            }
        },
        
        # ===== ABLATION: METADATA SOURCES =====
        "ablation_device_only": {
            "model": "ast_film",
            "description": "FiLM with device metadata only",
            "film_config": {
                **base_film,
                "condition_on_device": True,
                "condition_on_site": False,
                "condition_on_rest": False,
                "conditioned_layers": (10, 11),
            }
        },
        
        "ablation_site_only": {
            "model": "ast_film",
            "description": "FiLM with site metadata only",
            "film_config": {
                **base_film,
                "condition_on_device": False,
                "condition_on_site": True,
                "condition_on_rest": False,
                "conditioned_layers": (10, 11),
            }
        },
        
        "ablation_continuous_only": {
            "model": "ast_film",
            "description": "FiLM with continuous metadata only (age, BMI, duration)",
            "film_config": {
                **base_film,
                "condition_on_device": False,
                "condition_on_site": False,
                "condition_on_rest": True,
                "conditioned_layers": (10, 11),
            }
        },
        
        "ablation_device_site": {
            "model": "ast_film",
            "description": "FiLM with device + site (no continuous)",
            "film_config": {
                **base_film,
                "condition_on_device": True,
                "condition_on_site": True,
                "condition_on_rest": False,
                "conditioned_layers": (10, 11),
            }
        },
        
        "ablation_device_continuous": {
            "model": "ast_film",
            "description": "FiLM with device + continuous (no site)",
            "film_config": {
                **base_film,
                "condition_on_device": True,
                "condition_on_site": False,
                "condition_on_rest": True,
                "conditioned_layers": (10, 11),
            }
        },
        
        "ablation_site_continuous": {
            "model": "ast_film",
            "description": "FiLM with site + continuous (no device)",
            "film_config": {
                **base_film,
                "condition_on_device": False,
                "condition_on_site": True,
                "condition_on_rest": True,
                "conditioned_layers": (10, 11),
            }
        },
        
        # ===== ABLATION: LAYER SELECTION =====
        "layers_early": {
            "model": "ast_film",
            "description": "FiLM applied to early layers (0-2)",
            "film_config": {
                **base_film,
                "condition_on_device": True,
                "condition_on_site": True,
                "condition_on_rest": True,
                "conditioned_layers": (0, 1, 2),
            }
        },
        
        "layers_middle": {
            "model": "ast_film",
            "description": "FiLM applied to middle layers (5-7)",
            "film_config": {
                **base_film,
                "condition_on_device": True,
                "condition_on_site": True,
                "condition_on_rest": True,
                "conditioned_layers": (5, 6, 7),
            }
        },
        
        "layers_late": {
            "model": "ast_film",
            "description": "FiLM applied to late layers (10-11)",
            "film_config": {
                **base_film,
                "condition_on_device": True,
                "condition_on_site": True,
                "condition_on_rest": True,
                "conditioned_layers": (10, 11),
            }
        },
        
        "layers_mixed": {
            "model": "ast_film",
            "description": "FiLM applied to mixed layers (0, 6, 11)",
            "film_config": {
                **base_film,
                "condition_on_device": True,
                "condition_on_site": True,
                "condition_on_rest": True,
                "conditioned_layers": (0, 6, 11),
            }
        },
        
        "layers_all": {
            "model": "ast_film",
            "description": "FiLM applied to all layers",
            "film_config": {
                **base_film,
                "condition_on_device": True,
                "condition_on_site": True,
                "condition_on_rest": True,
                "conditioned_layers": tuple(range(12)),
            }
        },
        
        # ===== ARCHITECTURE VARIANTS =====
        "arch_large_embeddings": {
            "model": "ast_film",
            "description": "FiLM with larger embeddings (16d)",
            "film_config": {
                **base_film,
                "condition_on_device": True,
                "condition_on_site": True,
                "condition_on_rest": True,
                "conditioned_layers": (10, 11),
                "dev_emb_dim": 16,
                "site_emb_dim": 16,
            }
        },
        
        "arch_deep_encoder": {
            "model": "ast_film",
            "description": "FiLM with deeper metadata encoder",
            "film_config": {
                **base_film,
                "condition_on_device": True,
                "condition_on_site": True,
                "condition_on_rest": True,
                "conditioned_layers": (10, 11),
                "metadata_hidden_dim": 128,
                "film_hidden_dim": 128,
            }
        },
        
        "arch_layer_specific": {
            "model": "ast_film",
            "description": "FiLM with layer-specific encoding",
            "film_config": {
                **base_film,
                "condition_on_device": True,
                "condition_on_site": True,
                "condition_on_rest": True,
                "conditioned_layers": (10, 11),
                "layer_specific_encoding": True,
            }
        },
    }
    
    return experiments


def get_experiment_suites():
    """Define experiment suites (groups of related experiments)."""
    return {
        "baseline": ["baseline_vanilla", "baseline_film_full"],
        
        "ablation": [
            "baseline_vanilla",
            "ablation_device_only",
            "ablation_site_only",
            "ablation_continuous_only",
            "ablation_device_site",
            "ablation_device_continuous",
            "ablation_site_continuous",
            "baseline_film_full",
        ],
        
        "layers": [
            "layers_early",
            "layers_middle",
            "layers_late",
            "layers_mixed",
            "layers_all",
        ],
        
        "architecture": [
            "baseline_film_full",
            "arch_large_embeddings",
            "arch_deep_encoder",
            "arch_layer_specific",
        ],
        
        "all": None,  # Run all experiments
    }


# ============================================================
# CORE TRAINING FUNCTION
# ============================================================

def train_single_experiment(
    cfg,
    mlflow_cfg,
    model_key: str,
    experiment_name: str,
    film_config: dict = None,
    num_devices: int = 4,
    num_sites: int = 7,
    rest_dim: int = 3,
):
    """
    Train a single experiment.
    
    Args:
        cfg: Main config
        mlflow_cfg: MLflow config
        model_key: "ast" or "ast_film"
        experiment_name: Name for this run
        film_config: FiLM configuration dict (if model_key == "ast_film")
        num_devices, num_sites, rest_dim: Metadata dimensions
    
    Returns:
        trained_model: The trained model
        final_metrics: Dict of final test metrics
    """
    
    print("\n" + "="*80)
    print(f"EXPERIMENT: {experiment_name}")
    print(f"MODEL: {model_key}")
    if film_config:
        print(f"FiLM CONFIG: {film_config}")
    print("="*80 + "\n")
    
    # Set seed
    set_seed(cfg.seed)
    
    # Build dataloaders
    train_loader, test_loader = build_dataloaders(cfg.dataset, cfg.audio)
    
    # Build model
    if model_key == "ast_film" and film_config:
        model = build_model(
            cfg.models,
            model_key=model_key,
            num_devices=num_devices,
            num_sites=num_sites,
            rest_dim=rest_dim,
            **film_config
        )
    else:
        model = build_model(
            cfg.models,
            model_key=model_key,
            num_devices=num_devices,
            num_sites=num_sites,
            rest_dim=rest_dim
        )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: {model_key}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # MLflow setup
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_cfg.tracking_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_cfg.tracking_password
    mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
    mlflow.set_experiment(experiment_id=get_or_create_experiment(mlflow_cfg.experiment_name))
    
    run_name = f"{experiment_name}_{cfg.training.epochs}ep"
    
    with mlflow.start_run(run_name=run_name):
        # Log configuration
        log_all_params(cfg)
        mlflow.log_param("experiment_name", experiment_name)
        mlflow.log_param("model_key", model_key)
        mlflow.log_param("total_params", total_params)
        mlflow.log_param("trainable_params", trainable_params)
        
        if film_config:
            for k, v in film_config.items():
                mlflow.log_param(f"film_{k}", str(v))
        
        # Train
        trained_model, _ = train_loop(
            cfg.training,
            model,
            train_loader,
            val_loader=test_loader,
            test_loader=test_loader,
            fold_idx=0
        )
        
        # Get final metrics (they're already logged by train_loop)
        # We can retrieve them from the active run
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(mlflow.active_run().info.run_id)
        final_metrics = {k: v for k, v in run.data.metrics.items() if k.startswith("Test_")}
        print(final_metrics)
        
        mlflow.end_run()
    
    print(f"\n✓ Experiment '{experiment_name}' completed!")
    print(f"Final ICBHI Score: {final_metrics.get('Test_icbhi_score', 'N/A'):.4f}")
    
    return trained_model, final_metrics


# ============================================================
# EXPERIMENT SUITE RUNNER
# ============================================================

def run_experiment_suite(cfg, mlflow_cfg, suite_name: str):
    """
    Run a suite of experiments.
    
    Args:
        cfg: Main config
        mlflow_cfg: MLflow config
        suite_name: Name of suite to run (e.g., "ablation", "baseline")
    """
    
    suites = get_experiment_suites()
    experiments = get_experiment_configs()
    
    if suite_name not in suites:
        print(f"ERROR: Unknown suite '{suite_name}'")
        print(f"Available suites: {list(suites.keys())}")
        return
    
    # Get experiments to run
    if suite_name == "all":
        exp_names = list(experiments.keys())
    else:
        exp_names = suites[suite_name]
    
    print("\n" + "="*80)
    print(f"RUNNING EXPERIMENT SUITE: {suite_name.upper()}")
    print(f"Total experiments: {len(exp_names)}")
    print("="*80 + "\n")
    
    results = {}
    
    for i, exp_name in enumerate(exp_names, 1):
        print(f"\n[{i}/{len(exp_names)}] Starting: {exp_name}")
        
        exp_config = experiments[exp_name]
        model_key = exp_config["model"]
        film_config = exp_config.get("film_config", None)
        
        try:
            _, metrics = train_single_experiment(
                cfg, mlflow_cfg,
                model_key=model_key,
                experiment_name=exp_name,
                film_config=film_config
            )
            results[exp_name] = {"status": "success", "metrics": metrics}
            
        except Exception as e:
            print(f"ERROR in {exp_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[exp_name] = {"status": "failed", "error": str(e)}
    
    # Print summary
    print("\n" + "="*80)
    print(f"SUITE '{suite_name.upper()}' COMPLETED")
    print("="*80)
    
    successful = sum(1 for r in results.values() if r["status"] == "success")
    failed = sum(1 for r in results.values() if r["status"] == "failed")
    
    print(f"Successful: {successful}/{len(exp_names)}")
    print(f"Failed: {failed}/{len(exp_names)}")
    
    # Print results table
    if successful > 0:
        print("\nResults Summary:")
        print("-" * 80)
        print(f"{'Experiment':<35} {'ICBHI Score':<15} {'Status'}")
        print("-" * 80)
        
        for exp_name, result in results.items():
            if result["status"] == "success":
                icbhi = result["metrics"].get("Test_icbhi_score", 0.0)
                print(f"{exp_name:<35} {icbhi:<15.4f} ✓")
            else:
                print(f"{exp_name:<35} {'N/A':<15} ✗ ({result['error'][:30]}...)")
        print("-" * 80)


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_checkpoint(checkpoint_path: str, cfg):
    """
    Load a checkpoint and generate visualizations.
    
    Args:
        checkpoint_path: Path to model checkpoint
        cfg: Config object
    """
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Determine model type from checkpoint
    # This is a heuristic - you might need to save model_key in checkpoint
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Check if it's a FiLM model (has film-specific keys)
    is_film = any('film_generators' in k or 'dev_emb' in k for k in state_dict.keys())
    
    if is_film:
        print("Detected FiLM-AST model")
        model_key = "ast_film"
        
        # Try to infer FiLM config from checkpoint
        # For safety, use default config
        film_config = {
            "condition_on_device": True,
            "condition_on_site": True,
            "condition_on_rest": True,
            "conditioned_layers": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            "use_improved_continuous_encoder": True,
        }
        
        model = build_model(
            cfg.models,
            model_key=model_key,
            num_devices=4,
            num_sites=7,
            rest_dim=3,
            **film_config
        )
    else:
        print("Detected vanilla AST model")
        model_key = "ast"
        model = build_model(cfg.models, model_key=model_key)
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model loaded on {device}")
    
    # Only visualize if it's a FiLM model
    if not is_film:
        print("\nWARNING: This is not a FiLM model. Visualization only works for FiLM-AST.")
        return
    
    # Load test data
    _, test_loader = build_dataloaders(cfg.dataset, cfg.audio)
    sample_batch = next(iter(test_loader))
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    try:
        from ls.visualization.tools import generate_full_visualization_suite
        
        output_dir = Path(checkpoint_path).parent / "visualizations"
        generate_full_visualization_suite(
            model,
            sample_batch,
            test_loader,
            output_dir=str(output_dir)
        )
        
        print(f"\n✓ Visualizations saved to: {output_dir}")
        
    except ImportError:
        print("ERROR: visualization_tools module not found!")
        print("Make sure visualization_tools.py is in ls/visualization/tools.py")
    except Exception as e:
        print(f"ERROR during visualization: {e}")
        import traceback
        traceback.print_exc()


# ============================================================
# CLI INTERFACE
# ============================================================

def parse_args():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="FiLM-AST Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train vanilla AST
  python main.py --model ast --name my_baseline
  
  # Train FiLM with all metadata
  python main.py --model ast_film --name film_full
  
  # Train FiLM with custom config
  python main.py --model ast_film --name custom \\
      --film-layers 9 10 11 \\
      --film-device --no-film-site --film-continuous
  
  # Run experiment suite
  python main.py --experiment ablation
  
  # List all available experiments
  python main.py --list-experiments
  
  # Visualize trained model
  python main.py --visualize checkpoints/best_model.pt
        """
    )
    
    # Mode selection (mutually exclusive)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", type=str, choices=["ast", "ast_film"],
                      help="Train a single model")
    mode.add_argument("--experiment", type=str,
                      help="Run a predefined experiment by name")
    mode.add_argument("--suite", type=str,
                      help="Run an experiment suite (baseline, ablation, layers, architecture, all)")
    mode.add_argument("--visualize", type=str, metavar="CHECKPOINT",
                      help="Visualize a trained model checkpoint")
    mode.add_argument("--list-experiments", action="store_true",
                      help="List all available experiments and suites")
    
    # Single model training options
    parser.add_argument("--name", type=str, default="experiment",
                        help="Experiment name (used for single model training)")
    
    # FiLM configuration (only used with --model ast_film)
    film_group = parser.add_argument_group("FiLM Configuration")
    film_group.add_argument("--film-device", dest="film_device", action="store_true", default=True,
                           help="Condition on device metadata (default: True)")
    film_group.add_argument("--no-film-device", dest="film_device", action="store_false",
                           help="Don't condition on device metadata")
    film_group.add_argument("--film-site", dest="film_site", action="store_true", default=True,
                           help="Condition on site metadata (default: True)")
    film_group.add_argument("--no-film-site", dest="film_site", action="store_false",
                           help="Don't condition on site metadata")
    film_group.add_argument("--film-continuous", dest="film_continuous", action="store_true", default=True,
                           help="Condition on continuous metadata (default: True)")
    film_group.add_argument("--no-film-continuous", dest="film_continuous", action="store_false",
                           help="Don't condition on continuous metadata")
    film_group.add_argument("--film-layers", type=int, nargs="+", default=[10, 11],
                           help="Layers to apply FiLM conditioning (default: 10 11)")
    film_group.add_argument("--film-dev-emb", type=int, default=4,
                           help="Device embedding dimension (default: 4)")
    film_group.add_argument("--film-site-emb", type=int, default=4,
                           help="Site embedding dimension (default: 4)")
    film_group.add_argument("--film-hidden", type=int, default=64,
                           help="FiLM hidden dimension (default: 64)")
    film_group.add_argument("--film-improved-encoder", dest="film_improved", action="store_true", default=True,
                           help="Use improved continuous encoder (default: True)")
    film_group.add_argument("--no-film-improved-encoder", dest="film_improved", action="store_false",
                           help="Use simple continuous encoder")
    film_group.add_argument("--film-layer-specific", dest="film_layer_specific", action="store_true", default=False,
                           help="Use layer-specific metadata encoding (default: False)")
    film_group.add_argument("--film-debug", action="store_true", default=False,
                           help="Enable FiLM debug output")
    
    # Config files
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to main config file (default: configs/config.yaml)")
    parser.add_argument("--mlflow-config", type=str, default="configs/mlflow.yaml",
                       help="Path to MLflow config file (default: configs/mlflow.yaml)")
    
    return parser.parse_args()


def list_experiments():
    """Print all available experiments and suites."""
    
    experiments = get_experiment_configs()
    suites = get_experiment_suites()
    
    print("\n" + "="*80)
    print("AVAILABLE EXPERIMENTS")
    print("="*80 + "\n")
    
    # Group by category
    categories = {
        "Baselines": [k for k in experiments.keys() if k.startswith("baseline")],
        "Metadata Ablation": [k for k in experiments.keys() if k.startswith("ablation")],
        "Layer Selection": [k for k in experiments.keys() if k.startswith("layers")],
        "Architecture Variants": [k for k in experiments.keys() if k.startswith("arch")],
    }
    
    for category, exp_names in categories.items():
        print(f"\n{category}:")
        print("-" * 80)
        for name in exp_names:
            desc = experiments[name]["description"]
            print(f"  {name:<35} {desc}")
    
    print("\n" + "="*80)
    print("AVAILABLE SUITES")
    print("="*80 + "\n")
    
    for suite_name, exp_names in suites.items():
        if suite_name == "all":
            print(f"  {suite_name:<20} Run all {len(experiments)} experiments")
        else:
            count = len(exp_names)
            print(f"  {suite_name:<20} {count} experiments")
    
    print("\n")


def main():
    """Main entry point."""
    
    args = parse_args()
    
    # List experiments and exit
    if args.list_experiments:
        list_experiments()
        return
    
    # Load configs
    cfg = load_config(args.config)
    mlflow_cfg = load_config(args.mlflow_config)
    
    # Visualize checkpoint
    if args.visualize:
        visualize_checkpoint(args.visualize, cfg)
        return
    
    # Run experiment suite
    if args.suite:
        run_experiment_suite(cfg, mlflow_cfg, args.suite)
        return
    
    # Run single predefined experiment
    if args.experiment:
        experiments = get_experiment_configs()
        if args.experiment not in experiments:
            print(f"ERROR: Unknown experiment '{args.experiment}'")
            print(f"Available experiments:")
            list_experiments()
            return
        
        exp_config = experiments[args.experiment]
        train_single_experiment(
            cfg, mlflow_cfg,
            model_key=exp_config["model"],
            experiment_name=args.experiment,
            film_config=exp_config.get("film_config")
        )
        return
    
    # Train single model with custom config
    if args.model:
        if args.model == "ast_film":
            # Build FiLM config from arguments
            film_config = {
                "condition_on_device": args.film_device,
                "condition_on_site": args.film_site,
                "condition_on_rest": args.film_continuous,
                "conditioned_layers": tuple(args.film_layers),
                "dev_emb_dim": args.film_dev_emb,
                "site_emb_dim": args.film_site_emb,
                "metadata_hidden_dim": args.film_hidden,
                "film_hidden_dim": args.film_hidden,
                "use_improved_continuous_encoder": args.film_improved,
                "layer_specific_encoding": args.film_layer_specific,
                "debug_film": args.film_debug,
            }
        else:
            film_config = None
        
        train_single_experiment(
            cfg, mlflow_cfg,
            model_key=args.model,
            experiment_name=args.name,
            film_config=film_config
        )


if __name__ == "__main__":
    main()