import torch
import torch.nn as nn
from ls.models.ast import ASTModel

class ASTFiLM(nn.Module):
    """
    Flexible FiLM conditioning with configurable metadata sources.
    
    Can condition on:
    - Device (stethoscope) embeddings
    - Site (recording location) embeddings
    - Continuous metadata (age, BMI, duration, etc.)
    - Any combination of the above
    
    Args:
        condition_on_device: bool, use device embeddings
        condition_on_site: bool, use site embeddings
        condition_on_rest: bool, use continuous metadata
    """

    def __init__(
        self,
        ast_kwargs: dict,
        num_devices: int,
        num_sites: int,
        rest_dim: int,
        dev_emb_dim: int = 4,
        site_emb_dim: int = 4,
        conditioned_layers=(10, 11, 12),
        metadata_hidden_dim: int = 64,
        film_hidden_dim: int = 64,
        dropout_p: float = 0.3,
        num_labels: int = 2,
        debug_film: bool = False,
        condition_on_device: bool = True,
        condition_on_site: bool = True,
        condition_on_rest: bool = True,
    ):
        super().__init__()

        # Backbone
        self.ast = ASTModel(backbone_only=True, **ast_kwargs)
        D = self.ast.original_embedding_dim
        self.D = D

        self.debug_film = debug_film
        self.conditioned_layers = sorted(list(conditioned_layers))
        self.conditioned_set = set(self.conditioned_layers)
        
        # Conditioning flags
        self.condition_on_device = condition_on_device
        self.condition_on_site = condition_on_site
        self.condition_on_rest = condition_on_rest
        
        # Validate: at least one conditioning source must be enabled
        if not any([condition_on_device, condition_on_site, condition_on_rest]):
            raise ValueError("At least one conditioning source must be enabled!")

        # Categorical embeddings (only create if needed)
        if condition_on_device:
            self.dev_emb = nn.Embedding(num_devices, dev_emb_dim)
        else:
            self.dev_emb = None
            dev_emb_dim = 0
        
        if condition_on_site:
            self.site_emb = nn.Embedding(num_sites, site_emb_dim)
        else:
            self.site_emb = None
            site_emb_dim = 0
        
        if not condition_on_rest:
            rest_dim = 0

        # Total metadata dimension
        meta_dim = dev_emb_dim + site_emb_dim + rest_dim
        
        if debug_film:
            print(f"[FiLM Config] Device: {condition_on_device} ({dev_emb_dim}d), "
                  f"Site: {condition_on_site} ({site_emb_dim}d), "
                  f"Rest: {condition_on_rest} ({rest_dim}d)")
            print(f"[FiLM Config] Total metadata dim: {meta_dim}")
            print(f"[FiLM Config] Conditioned layers: {self.conditioned_layers}")

        # Metadata encoder h_m = g(m)
        self.metadata_encoder = nn.Sequential(
            nn.LayerNorm(meta_dim),
            nn.Linear(meta_dim, metadata_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p * 0.5),  # Light dropout in encoder
            nn.Linear(metadata_hidden_dim, film_hidden_dim),
            nn.ReLU(),
        )

        # One FiLM generator per conditioned layer: h_m -> [Δγ || Δβ] in R^{2D}
        self.film_generators = nn.ModuleDict({
            str(l): nn.Linear(film_hidden_dim, 2 * D) for l in self.conditioned_layers
        })

        # Classifier head (multi-label logits)
        self.classifier = nn.Sequential(
            nn.LayerNorm(D),
            nn.Dropout(dropout_p),
            nn.Linear(D, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, num_labels),
        )

    def _prep_tokens(self, x):
        """
        Build token sequence like ASTModel.forward_features().
        x: (B,1,F,T)
        returns: (B, N+2, D)
        """
        B = x.shape[0]
        v = self.ast.v

        x = v.patch_embed(x)  # (B, N, D)
        cls_tokens = v.cls_token.expand(B, -1, -1)
        dist_token = v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)  # (B, N+2, D)

        x = x + v.pos_embed
        x = v.pos_drop(x)
        return x

    def forward_features(self, x, device_id=None, site_id=None, m_rest=None):
        """
        x: (B,1,F,T)
        device_id: (B,) or None
        site_id: (B,) or None
        m_rest: (B, rest_dim) or None
        returns: (B, D)
        """
        v = self.ast.v
        x = self._prep_tokens(x)

        # Build metadata vector m (only include enabled sources)
        metadata_parts = []
        
        if self.condition_on_device:
            if device_id is None:
                raise ValueError("device_id required when condition_on_device=True")
            dev = self.dev_emb(device_id)  # (B, dev_emb_dim)
            metadata_parts.append(dev)
        
        if self.condition_on_site:
            if site_id is None:
                raise ValueError("site_id required when condition_on_site=True")
            site = self.site_emb(site_id)  # (B, site_emb_dim)
            metadata_parts.append(site)
        
        if self.condition_on_rest:
            if m_rest is None:
                raise ValueError("m_rest required when condition_on_rest=True")
            metadata_parts.append(m_rest)  # (B, rest_dim)
        
        m = torch.cat(metadata_parts, dim=-1)  # (B, meta_dim)

        # Encode once
        h_m = self.metadata_encoder(m)  # (B, film_hidden_dim)

        # Precompute FiLM params for each conditioned layer
        gamma = {}
        beta = {}
        for l in self.conditioned_layers:
            film = self.film_generators[str(l)](h_m)  # (B, 2D)
            dg, db = film.chunk(2, dim=-1)            # (B,D), (B,D)
            gamma[l] = 1.0 + dg                       # identity-centered
            beta[l] = db

        # Unroll transformer blocks
        for layer_idx, blk in enumerate(v.blocks):
            # Attention block
            attn_out = blk.attn(blk.norm1(x))
            x = x + blk.drop_path(attn_out)  # MHSA residual

            # Apply FiLM if this layer is conditioned
            if layer_idx in self.conditioned_set:
                if self.debug_film:
                    print(f"[FiLM] Applying to layer {layer_idx}")
                g = gamma[layer_idx].unsqueeze(1)  # (B,1,D)
                b = beta[layer_idx].unsqueeze(1)   # (B,1,D)
                x = g * x + b

            # FFN block
            x = x + blk.drop_path(blk.mlp(blk.norm2(x)))  # FFN residual

        x = v.norm(x)
        h_cls = (x[:, 0] + x[:, 1]) / 2.0
        return h_cls

    def forward(self, x, device_id=None, site_id=None, m_rest=None):
        """
        Forward pass with flexible metadata.
        
        Args:
            x: (B,T,F) or (B,1,F,T) spectrogram
            device_id: (B,) device indices (required if condition_on_device=True)
            site_id: (B,) site indices (required if condition_on_site=True)
            m_rest: (B, rest_dim) continuous metadata (required if condition_on_rest=True)
        
        Returns:
            logits: (B, num_labels)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1).transpose(2, 3)  # (B,1,F,T)

        h = self.forward_features(x, device_id, site_id, m_rest)
        logits = self.classifier(h)
        return logits
    
    def freeze_backbone(self, until_block=None, freeze_film=False):
        """
        Freeze AST backbone (optionally keep FiLM trainable).
        
        Args:
            until_block: freeze up to this block (None = all)
            freeze_film: if True, also freeze FiLM generators
        """
        # Freeze patch embedding
        for p in self.ast.v.patch_embed.parameters():
            p.requires_grad = False
        
        # Freeze positional embeddings
        self.ast.v.pos_embed.requires_grad = False
        self.ast.v.cls_token.requires_grad = False
        self.ast.v.dist_token.requires_grad = False
        
        # Freeze transformer blocks
        for i, blk in enumerate(self.ast.v.blocks):
            if until_block is None or i <= until_block:
                for p in blk.parameters():
                    p.requires_grad = False
        
        # Optionally freeze FiLM components
        if freeze_film:
            if self.dev_emb is not None:
                for p in self.dev_emb.parameters():
                    p.requires_grad = False
            if self.site_emb is not None:
                for p in self.site_emb.parameters():
                    p.requires_grad = False
            for p in self.metadata_encoder.parameters():
                p.requires_grad = False
            for p in self.film_generators.parameters():
                p.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True