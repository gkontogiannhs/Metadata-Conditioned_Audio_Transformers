import torch.nn as nn
from typing import Optional
from ls.config.dataclasses import ModelsConfig
import torch


def build_model(cfg, model_key, num_devices=4, num_sites=7, rest_dim=3):
    """
    Build model based on configuration.
    
    Args:
        cfg: Model configuration
        model_key: "ast" or "ast_film"
        num_devices, num_sites, rest_dim: Metadata dimensions
    
    Returns:
        model: Instantiated model
    """
    print(cfg[model_key])
    if model_key == "ast":
        # Vanilla AST without FiLM
        ast_cfg = cfg[model_key]
        from ls.models.ast import ASTModel
        print(ast_cfg)
        model = ASTModel(
            label_dim=ast_cfg['label_dim'],  # Binary / multi-label (crackle, wheeze)
            fstride=ast_cfg.get('fstride', 10),
            tstride=ast_cfg.get('tstride', 10),
            input_fdim=ast_cfg.get('input_fdim', 128),
            input_tdim=ast_cfg.get('input_tdim', 1024),
            imagenet_pretrain=ast_cfg.get('imagenet_pretrain', True),
            audioset_pretrain=ast_cfg.get('audioset_pretrain', True),
            audioset_ckpt_path=ast_cfg['audioset_ckpt_path'],
            model_size=ast_cfg.get('model_size', 'base384'),
            verbose=ast_cfg.get('verbose', True),
            dropout_p=ast_cfg.get('dropout', 0.0),
        )
    elif model_key == "ast_meta_proj":
        from ls.models.ast_fus import ASTMetaProj
        ast_fus_cfg = cfg[model_key]
        ast_kwargs = {
            "label_dim": ast_fus_cfg['label_dim'],  # Binary / multi-label (crackle, wheeze)
            "fstride": ast_fus_cfg.get('fstride', 10),
            "tstride": ast_fus_cfg.get('tstride', 10),
            "input_fdim": ast_fus_cfg.get('input_fdim', 128),
            "input_tdim": ast_fus_cfg.get('input_tdim', 1024),
            "imagenet_pretrain": ast_fus_cfg.get('imagenet_pretrain', True),
            "audioset_pretrain": ast_fus_cfg.get('audioset_pretrain', True),
            "audioset_ckpt_path": ast_fus_cfg['audioset_ckpt_path'],
            "model_size": ast_fus_cfg.get('model_size', 'base384'),
            "verbose": ast_fus_cfg.get('verbose', True),
            "dropout_p": ast_fus_cfg.get('dropout', 0.3)
        }
        
        model = ASTMetaProj(
            ast_kwargs=ast_kwargs,
            num_devices=num_devices,
            num_sites=num_sites,
            dev_emb_dim=ast_fus_cfg.get("dev_emb_dim", 8),
            site_emb_dim=ast_fus_cfg.get("site_emb_dim", 14),
            rest_dim=rest_dim,
            hidden_dim=ast_fus_cfg.get("hidden_dim", 64),
            dropout_p=ast_fus_cfg.get("dropout_p", 0.3),
            num_labels=ast_fus_cfg.get("label_dim", 2),
            init_gate=ast_fus_cfg.get("init_gate", 0.5),
            # Ablation flags
            use_device=ast_fus_cfg.get("use_device", True),
            use_site=ast_fus_cfg.get("use_site", True),
            use_continuous=ast_fus_cfg.get("use_continuous", True),
            use_missing_flags=ast_fus_cfg.get("use_missing_flags", False),
        )

    elif model_key == "ast_film":
        # FiLM-conditioned AST
        film_cfg = cfg[model_key]
        
        # Prepare AST kwargs
        ast_kwargs = {
            "label_dim": film_cfg['label_dim'],  # Binary / multi-label (crackle, wheeze)
            "fstride": film_cfg.get('fstride', 10),
            "tstride": film_cfg.get('tstride', 10),
            "input_fdim": film_cfg.get('input_fdim', 128),
            "input_tdim": film_cfg.get('input_tdim', 1024),
            "imagenet_pretrain": film_cfg.get('imagenet_pretrain', True),
            "audioset_pretrain": film_cfg.get('audioset_pretrain', True),
            "audioset_ckpt_path": film_cfg['audioset_ckpt_path'],
            "model_size": film_cfg.get('model_size', 'base384'),
            "verbose": film_cfg.get('verbose', True),
            "dropout_p": film_cfg.get('dropout', 0.3)
        }

        from ls.models.ast_film import ASTFiLM
        model = ASTFiLM(
            ast_kwargs=ast_kwargs,
            num_devices=num_devices,
            num_sites=num_sites,
            rest_dim=rest_dim,
            dev_emb_dim=film_cfg["dev_emb_dim"],
            site_emb_dim=film_cfg["site_emb_dim"],
            metadata_hidden_dim=film_cfg["metadata_hidden_dim"],
            film_hidden_dim=film_cfg["film_hidden_dim"],
            dropout_p=film_cfg["dropout_p"],
            debug_film=film_cfg["debug_film"],
            condition_on_device=film_cfg["use_device"],
            condition_on_site=film_cfg["use_site"],
            condition_on_rest=film_cfg["use_continuous"],
            conditioned_layers=film_cfg["conditioned_layers"],
            use_improved_continuous_encoder=film_cfg["use_improved_continuous_encoder"],
            layer_specific_encoding=film_cfg["layer_specific_encoding"],
        )
    
    elif model_key == "tafilm":
        from ls.models.tafilm import ASTTAFiLM
        model_cfg = cfg[model_key]
        # AST backbone kwargs
        ast_kwargs = {
            "label_dim": model_cfg["label_dim"],
            "fstride": model_cfg.get("fstride", 10),
            "tstride": model_cfg.get("tstride", 10),
            "input_fdim": model_cfg.get("input_fdim", 128),
            "input_tdim": model_cfg.get("input_tdim", 1024),
            "imagenet_pretrain": model_cfg.get("imagenet_pretrain", True),
            "audioset_pretrain": model_cfg.get("audioset_pretrain", True),
            "audioset_ckpt_path": model_cfg['audioset_ckpt_path'],
            "model_size": model_cfg.get("model_size", "base384"),
        }

        # Conditioned layers
        conditioned_layers = model_cfg.get("conditioned_layers", [10, 11])
        if isinstance(conditioned_layers, list):
            conditioned_layers = tuple(conditioned_layers)

        model = ASTTAFiLM(
            ast_kwargs=ast_kwargs,
            num_devices=num_devices,
            num_sites=num_sites,
            rest_dim=rest_dim,
            dev_emb_dim=model_cfg.get("dev_emb_dim", 8),
            site_emb_dim=model_cfg.get("site_emb_dim", 14),
            conditioned_layers=conditioned_layers,
            metadata_hidden_dim=model_cfg.get("metadata_hidden_dim", 64),
            film_hidden_dim=model_cfg.get("film_hidden_dim", 64),
            dropout_p=model_cfg.get("dropout", 0.3),
            debug=model_cfg.get("debug", False),
            condition_on_device=model_cfg.get("use_device", True),
            condition_on_site=model_cfg.get("use_site", True),
            condition_on_rest=model_cfg.get("use_continuous", True),
            use_improved_continuous_encoder=model_cfg.get("use_improved_continuous_encoder", True),
        )

        return model
    
    elif model_key == "ast_film_soft":
            from ls.models.ast_filmpp_soft import ASTFiLMPlusPlusSoft
            model_cfg = cfg[model_key]

            ast_kwargs = {
                "label_dim": model_cfg["label_dim"],
                "fstride": model_cfg.get("fstride", 10),
                "tstride": model_cfg.get("tstride", 10),
                "input_fdim": model_cfg.get("input_fdim", 128),
                "input_tdim": model_cfg.get("input_tdim", 1024),
                "imagenet_pretrain": model_cfg.get("imagenet_pretrain", True),
                "audioset_pretrain": model_cfg.get("audioset_pretrain", True),
                "audioset_ckpt_path": model_cfg['audioset_ckpt_path'],
                "model_size": model_cfg.get("model_size", "base384"),
            }

            conditioned_layers = model_cfg.get("conditioned_layers", [10, 11])
            if isinstance(conditioned_layers, list):
                conditioned_layers = tuple(conditioned_layers)

            model = ASTFiLMPlusPlusSoft(
                ast_kwargs=ast_kwargs,
                num_devices=num_devices,
                num_sites=num_sites,
                rest_dim=rest_dim,
                num_labels=model_cfg["label_dim"],
                dev_emb_dim=model_cfg.get("dev_emb_dim", 8),
                site_emb_dim=model_cfg.get("site_emb_dim", 14),
                conditioned_layers=conditioned_layers,
                metadata_hidden_dim=model_cfg.get("metadata_hidden_dim", 64),
                film_hidden_dim=model_cfg.get("film_hidden_dim", 64),
                dropout_p=model_cfg.get("dropout", 0.3),
                # v2 mask parameters
                mask_init_scale=model_cfg.get("mask_init_scale", 0.5),
                mask_sparsity_lambda=model_cfg.get("mask_sparsity_lambda", 0.01),
                mask_coverage_lambda=model_cfg.get("mask_coverage_lambda", 0.005),
                per_layer_masks=model_cfg.get("per_layer_masks", False),
                mask_temperature=model_cfg.get("mask_temperature", 1.0),
                film_init_gain=model_cfg.get("film_init_gain", 0.1),
                debug_film=model_cfg.get("debug_film", False),
            )

            return model
    
    else:
        raise ValueError(f"Unknown model_key: {model_key}")
    
    return model
