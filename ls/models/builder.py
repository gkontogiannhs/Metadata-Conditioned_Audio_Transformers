import torch.nn as nn
from typing import Optional
from ls.config.dataclasses import ModelsConfig


def build_model(
    cfg: ModelsConfig,
    model_key: str = "cnn6",
    *,
    num_devices: Optional[int] = None,
    num_sites: Optional[int] = None,
    rest_dim: Optional[int] = None,
) -> nn.Module:
    """
    Factory function to build a model (CNN6, AST, AST+Metadata variants) from config.

    IMPORTANT:
      - For FiLM / FiLM++ you must pass num_devices, num_sites, rest_dim
        (because embeddings depend on vocab sizes).
    """
    model_cfg = getattr(cfg, model_key, None)
    print(f"Building model '{model_key}' with config: {model_cfg}")

    if model_cfg is None:
        raise ValueError(f"Model key '{model_key}' not found in ModelsConfig")

    # ---------- SimpleRespCNN ----------
    if model_key == "simplerespcnn":
        from ls.models.baseline import SimpleRespCNN
        return SimpleRespCNN(n_classes=getattr(model_cfg, "label_dim", 4))

    # ---------- CNN6 ----------
    if model_key == "cnn6":
        from ls.models.cnn import CNN6
        return CNN6(
            in_channels=1,
            num_classes=getattr(model_cfg, "label_dim", 4),
            do_dropout=getattr(model_cfg, "do_dropout", False),
            cpt_path=getattr(model_cfg, "cpt_path", None),
        )

    # ---------- AST baseline ----------
    if model_key == "ast":
        from ls.models.ast import ASTModel
        return ASTModel(
            label_dim=getattr(model_cfg, "label_dim", 4),
            fstride=getattr(model_cfg, "fstride", 10),
            tstride=getattr(model_cfg, "tstride", 10),
            input_fdim=getattr(model_cfg, "input_fdim", 128),
            input_tdim=getattr(model_cfg, "input_tdim", 1024),
            imagenet_pretrain=getattr(model_cfg, "imagenet_pretrain", True),
            audioset_pretrain=getattr(model_cfg, "audioset_pretrain", True),
            model_size=getattr(model_cfg, "model_size", "base384"),
            backbone_only=getattr(model_cfg, "backbone_only", False),
            audioset_ckpt_path=getattr(model_cfg, "audioset_ckpt_path", None),
            dropout_p=getattr(model_cfg, "dropout", 0.0),
        )

    # ---------- AST + Metadata Projection Fusion ----------
    if model_key == "ast_proj":
        # You should have saved this class under ls.models.ast_variants or similar
        from ls.models.ast_fus import ASTMetaProj

        ast_kwargs = dict(
            label_dim=getattr(model_cfg, "label_dim", 2),
            fstride=getattr(model_cfg, "fstride", 10),
            tstride=getattr(model_cfg, "tstride", 10),
            input_fdim=getattr(model_cfg, "input_fdim", 128),
            input_tdim=getattr(model_cfg, "input_tdim", 1024),
            imagenet_pretrain=getattr(model_cfg, "imagenet_pretrain", True),
            audioset_pretrain=getattr(model_cfg, "audioset_pretrain", True),
            model_size=getattr(model_cfg, "model_size", "base384"),
            audioset_ckpt_path=getattr(model_cfg, "audioset_ckpt_path", None),
            verbose=getattr(model_cfg, "verbose", False),
        )

        if num_devices is None or num_sites is None or rest_dim is None:
            raise ValueError("ast_proj requires num_devices, num_sites, rest_dim")

        return ASTMetaProj(
            ast_kwargs=ast_kwargs,
            num_devices=num_devices,
            num_sites=num_sites,
            dev_emb_dim=getattr(model_cfg, "dev_emb_dim", 4),
            site_emb_dim=getattr(model_cfg, "site_emb_dim", 4),
            rest_dim=rest_dim,
            hidden_dim=getattr(model_cfg, "hidden_dim", 64),
            dropout_p=getattr(model_cfg, "dropout", 0.3),
            num_labels=getattr(model_cfg, "label_dim", 2),
        )

    # ---------- AST + FiLM (single-stream) ----------
    if model_key == "ast_film":
        from ls.models.ast_film import ASTFiLM

        ast_kwargs = dict(
            label_dim=getattr(model_cfg, "label_dim", 2),
            fstride=getattr(model_cfg, "fstride", 10),
            tstride=getattr(model_cfg, "tstride", 10),
            input_fdim=getattr(model_cfg, "input_fdim", 128),
            input_tdim=getattr(model_cfg, "input_tdim", 1024),
            imagenet_pretrain=getattr(model_cfg, "imagenet_pretrain", True),
            audioset_pretrain=getattr(model_cfg, "audioset_pretrain", True),
            model_size=getattr(model_cfg, "model_size", "base384"),
            audioset_ckpt_path=getattr(model_cfg, "audioset_ckpt_path", None),
            verbose=getattr(model_cfg, "verbose", False),
        )

        if num_devices is None or num_sites is None or rest_dim is None:
            raise ValueError("ast_film requires num_devices, num_sites, rest_dim")

        return ASTFiLM(
            ast_kwargs=ast_kwargs,
            num_devices=num_devices,
            num_sites=num_sites,
            rest_dim=rest_dim,
            dev_emb_dim=getattr(model_cfg, "dev_emb_dim", 4),
            site_emb_dim=getattr(model_cfg, "site_emb_dim", 4),
            conditioned_layers=tuple(getattr(model_cfg, "conditioned_layers", (10, 11, 12))),
            metadata_hidden_dim=getattr(model_cfg, "metadata_hidden_dim", 64),
            film_hidden_dim=getattr(model_cfg, "film_hidden_dim", 64),
            dropout_p=getattr(model_cfg, "dropout", 0.3),
            num_labels=getattr(model_cfg, "label_dim", 2),
            debug_film=getattr(model_cfg, "debug_film", False),
        )

    # ---------- AST + FiLM++ (grouped) ----------
    if model_key == "ast_filmpp":
        from ls.models.ast_pp import ASTFiLMPlusPlus  # <- put FiLM++ class here

        ast_kwargs = dict(
            label_dim=getattr(model_cfg, "label_dim", 2),
            fstride=getattr(model_cfg, "fstride", 10),
            tstride=getattr(model_cfg, "tstride", 10),
            input_fdim=getattr(model_cfg, "input_fdim", 128),
            input_tdim=getattr(model_cfg, "input_tdim", 1024),
            imagenet_pretrain=getattr(model_cfg, "imagenet_pretrain", True),
            audioset_pretrain=getattr(model_cfg, "audioset_pretrain", True),
            model_size=getattr(model_cfg, "model_size", "base384"),
            audioset_ckpt_path=getattr(model_cfg, "audioset_ckpt_path", None),
            verbose=getattr(model_cfg, "verbose", False),
        )

        if num_devices is None or num_sites is None or rest_dim is None:
            raise ValueError("ast_filmpp requires num_devices, num_sites, rest_dim")

        return ASTFiLMPlusPlus(
            ast_kwargs=ast_kwargs,
            num_devices=num_devices,
            num_sites=num_sites,
            rest_dim=rest_dim,
            D_dev=getattr(model_cfg, "D_dev", 128),
            D_site=getattr(model_cfg, "D_site", 128),
            conditioned_layers=tuple(getattr(model_cfg, "conditioned_layers", (10, 11, 12))),
            dev_emb_dim=getattr(model_cfg, "dev_emb_dim", 4),
            site_emb_dim=getattr(model_cfg, "site_emb_dim", 4),
            metadata_hidden_dim=getattr(model_cfg, "metadata_hidden_dim", 64),
            film_hidden_dim=getattr(model_cfg, "film_hidden_dim", 64),
            dropout_p=getattr(model_cfg, "dropout", 0.3),
            num_labels=getattr(model_cfg, "label_dim", 2),
            debug_film=getattr(model_cfg, "debug_film", False),
        )

    raise ValueError(f"Unknown model name: {model_key}")