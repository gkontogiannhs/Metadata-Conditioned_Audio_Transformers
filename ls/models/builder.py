import torch.nn as nn
from typing import Optional
from ls.config.dataclasses import ModelsConfig

def build_model(cfg, model_key, num_devices=4, num_sites=7, rest_dim=3, **film_kwargs):
    """
    Build model based on configuration.
    
    Args:
        cfg: Model configuration
        model_key: "ast" or "ast_film"
        num_devices, num_sites, rest_dim: Metadata dimensions
        **film_kwargs: Additional FiLM configuration (overrides defaults)
    
    Returns:
        model: Instantiated model
    """
    
    if model_key == "ast":
        # Vanilla AST without FiLM
        ast_cfg = cfg.get('ast', {})
        from ls.models.ast import ASTModel
        model = ASTModel(
            label_dim=2,  # Binary multi-label (crackle, wheeze)
            fstride=ast_cfg.get('fstride', 10),
            tstride=ast_cfg.get('tstride', 10),
            input_fdim=ast_cfg.get('input_fdim', 128),
            input_tdim=ast_cfg.get('input_tdim', 1024),
            imagenet_pretrain=ast_cfg.get('imagenet_pretrain', True),
            audioset_pretrain=ast_cfg.get('audioset_pretrain', False),
            audioset_ckpt_path=ast_cfg.get('audioset_ckpt_path', ''),
            model_size=ast_cfg.get('model_size', 'base384'),
            verbose=ast_cfg.get('verbose', True),
            dropout_p=ast_cfg.get('dropout_p', 0.3),
        )
        
    elif model_key == "ast_film":
        # FiLM-conditioned AST
        ast_cfg = cfg.get('ast', {})
        film_cfg = cfg.get('film', {})
        
        # Prepare AST kwargs
        ast_kwargs = {
            'label_dim': 2,
            'fstride': ast_cfg.get('fstride', 10),
            'tstride': ast_cfg.get('tstride', 10),
            'input_fdim': ast_cfg.get('input_fdim', 128),
            'input_tdim': ast_cfg.get('input_tdim', 1024),
            'imagenet_pretrain': ast_cfg.get('imagenet_pretrain', True),
            'audioset_pretrain': ast_cfg.get('audioset_pretrain', False),
            'audioset_ckpt_path': ast_cfg.get('audioset_ckpt_path', ''),
            'model_size': ast_cfg.get('model_size', 'base384'),
            'verbose': ast_cfg.get('verbose', True),
            'dropout_p': ast_cfg.get('dropout_p', 0.3),
        }
        
        # Merge film_kwargs with config (command-line overrides config)
        final_film_kwargs = {
            'dev_emb_dim': film_cfg.get('dev_emb_dim', 4),
            'site_emb_dim': film_cfg.get('site_emb_dim', 4),
            'conditioned_layers': film_cfg.get('conditioned_layers', (10, 11)),
            'metadata_hidden_dim': film_cfg.get('metadata_hidden_dim', 64),
            'film_hidden_dim': film_cfg.get('film_hidden_dim', 64),
            'dropout_p': ast_cfg.get('dropout_p', 0.3),
            'num_labels': 2,
            'debug_film': film_cfg.get('debug_film', False),
            'condition_on_device': film_cfg.get('condition_on_device', True),
            'condition_on_site': film_cfg.get('condition_on_site', True),
            'condition_on_rest': film_cfg.get('condition_on_rest', True),
            'use_improved_continuous_encoder': film_cfg.get('use_improved_continuous_encoder', True),
            'layer_specific_encoding': film_cfg.get('layer_specific_encoding', False),
        }
        final_film_kwargs.update(film_kwargs)  # Override with kwargs
        
        from ls.models.ast_film import ASTFiLM
        model = ASTFiLM(
            ast_kwargs=ast_kwargs,
            num_devices=num_devices,
            num_sites=num_sites,
            rest_dim=rest_dim,
            **final_film_kwargs
        )
    
    else:
        raise ValueError(f"Unknown model_key: {model_key}")
    
    return model