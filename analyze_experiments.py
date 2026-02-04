#!/usr/bin/env python
"""
analyze_experiments.py

Quick analysis of FiLM++ experiment results.
"""

import os
import json
import argparse
from pathlib import Path
import pandas as pd


def load_experiment_results(output_dir):
    """Load all experiment results from output directory."""
    results = []
    
    for exp_dir in Path(output_dir).iterdir():
        if not exp_dir.is_dir():
            continue
        
        config_path = exp_dir / "config.json"
        history_path = exp_dir / "history.json"
        
        if not config_path.exists() or not history_path.exists():
            continue
        
        with open(config_path) as f:
            config = json.load(f)
        
        with open(history_path) as f:
            history = json.load(f)
        
        # Get best epoch
        best_epoch = max(history, key=lambda x: x.get('val_icbhi_score', 0))
        
        results.append({
            'exp_name': exp_dir.name,
            'conditioned_layers': str(config.get('conditioned_layers', [])),
            'mask_sparsity_lambda': config.get('mask_sparsity_lambda', 0),
            'mask_init_scale': config.get('mask_init_scale', 0),
            'per_layer_masks': config.get('per_layer_masks', False),
            'dev_emb_dim': config.get('dev_emb_dim', 0),
            'site_emb_dim': config.get('site_emb_dim', 0),
            'lr': config.get('lr', 0),
            'dropout': config.get('dropout', 0),
            'freeze_backbone': config.get('freeze_backbone', False),
            'best_epoch': best_epoch.get('epoch', 0),
            'best_val_icbhi': best_epoch.get('val_icbhi_score', 0) * 100,
            'best_val_se': best_epoch.get('val_sensitivity', 0) * 100,
            'best_val_sp': best_epoch.get('val_specificity', 0) * 100,
            'best_val_f1': best_epoch.get('val_macro_f1', 0) * 100,
        })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--sort_by', type=str, default='best_val_icbhi')
    args = parser.parse_args()
    
    df = load_experiment_results(args.output_dir)
    
    if df.empty:
        print("No experiments found!")
        return
    
    # Sort by metric
    df = df.sort_values(args.sort_by, ascending=False)
    
    # Display
    print("\n" + "=" * 100)
    print("EXPERIMENT RESULTS (sorted by {})".format(args.sort_by))
    print("=" * 100)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.2f}'.format)
    
    display_cols = [
        'exp_name', 'best_val_icbhi', 'best_val_se', 'best_val_sp', 
        'best_val_f1', 'best_epoch'
    ]
    print(df[display_cols].to_string(index=False))
    
    print("\n" + "=" * 100)
    print("TOP 5 EXPERIMENTS")
    print("=" * 100)
    print(df.head(5).to_string(index=False))
    
    # Save to CSV
    csv_path = os.path.join(args.output_dir, "experiment_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()