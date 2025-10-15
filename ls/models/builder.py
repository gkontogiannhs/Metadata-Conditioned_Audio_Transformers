import torch.nn as nn

# from ls.models.resnet import ResNet18, ResNet34, ResNet50
# from ls.models.ssast import SSASTModel
from ls.config.dataclasses import ModelsConfig


def build_model(cfg: ModelsConfig, model_key: str = "cnn6") -> nn.Module:
    """
    Factory function to build a model (CNN6, AST, etc.) from the configuration.

    Args:
        cfg (ModelsConfig): The models configuration container (contains model1, model2, etc.)
        model_key (str): Which model to build ("model1" or "model2", defaults to model1)

    Returns:
        model (nn.Module): Instantiated model.
    """
    # pick the specific model configuration
    model_cfg = getattr(cfg, model_key, None)
    print(f"Building model '{model_key}' with config: {model_cfg}")

    if model_cfg is None:
        raise ValueError(f"Model key '{model_key}' not found in ModelsConfig")

    # ---------- SimpleRespCNN ----------
    if model_key == "simplerespcnn":
        from ls.models.baseline import SimpleRespCNN

        return SimpleRespCNN(
            n_classes=model_cfg.label_dim if hasattr(model_cfg, "label_dim") else 4
        )

    # ---------- CNN6 ----------
    if model_key == "cnn6":
        from ls.models.cnn import CNN6

        return CNN6(
            in_channels=1,
            num_classes=model_cfg.label_dim if hasattr(model_cfg, "label_dim") else 4,
            do_dropout=getattr(model_cfg, "do_dropout", False),
            cpt_path=getattr(model_cfg, "cpt_path", None),
        )

    # ---------- AST ----------
    elif model_key == "ast":
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

    # elif name == "resnet18":
    #     return ResNet18(num_classes=model_cfg.label_dim)
    # elif name == "resnet34":
    #     return ResNet34(num_classes=model_cfg.label_dim)
    # elif name == "ssast":
    #     return SSASTModel(...)

    else:
        raise ValueError(f"Unknown model name: {model_key}")
