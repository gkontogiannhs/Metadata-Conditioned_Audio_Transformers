import torch
import torch.nn as nn
from ls.models.ast import ASTModel

import torch
import torch.nn as nn

class ASTFiLM(nn.Module):
    """
    Single-stream FiLM conditioning:
      m = [Emb_dev(device_id), Emb_site(site_id), m_rest]
      h_m = g(m)
      For each conditioned layer l: [Δγ_l, Δβ_l] = W_l h_m
      Apply: Z = (1+Δγ_l) ⊙ Z + Δβ_l   (broadcast over tokens)
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
    ):
        super().__init__()

        # Backbone
        self.ast = ASTModel(backbone_only=True, **ast_kwargs)
        D = self.ast.original_embedding_dim
        self.D = D

        self.debug_film = debug_film
        self.conditioned_layers = sorted(list(conditioned_layers))
        self.conditioned_set = set(self.conditioned_layers)

        # Categorical embeddings
        self.dev_emb = nn.Embedding(num_devices, dev_emb_dim)
        self.site_emb = nn.Embedding(num_sites, site_emb_dim)

        meta_dim = dev_emb_dim + site_emb_dim + rest_dim

        # Metadata encoder h_m = g(m)
        self.metadata_encoder = nn.Sequential(
            nn.LayerNorm(meta_dim),
            nn.Linear(meta_dim, metadata_hidden_dim),
            nn.ReLU(),
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
        Build token sequence like ASTModel.forward_features(), but we will unroll blocks ourselves.
        x: (B,1,F,T)
        returns token tensor (B, N+2, D)
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

    def forward_features(self, x, device_id, site_id, m_rest):
        """
        x: (B,1,F,T)
        device_id: (B,)
        site_id: (B,)
        m_rest: (B, rest_dim)
        returns: (B, D)
        """
        v = self.ast.v
        x = self._prep_tokens(x)

        # Build metadata vector m
        dev = self.dev_emb(device_id)      # (B, dev_emb_dim)
        site = self.site_emb(site_id)      # (B, site_emb_dim)
        m = torch.cat([dev, site, m_rest], dim=-1)

        # Encode once
        h_m = self.metadata_encoder(m)     # (B, film_hidden_dim)

        # Precompute FiLM params for each conditioned layer
        gamma = {}
        beta = {}
        for l in self.conditioned_layers:
            film = self.film_generators[str(l)](h_m)   # (B, 2D)
            dg, db = film.chunk(2, dim=-1)            # (B,D), (B,D)
            gamma[l] = 1.0 + dg                       # identity-centered
            beta[l]  = db

        # Unroll transformer blocks
        for layer_idx, blk in enumerate(v.blocks):
            attn_out = blk.attn(blk.norm1(x))
            x = x + blk.drop_path(attn_out)           # after MHSA residual

            if layer_idx in self.conditioned_set:
                if self.debug_film:
                    print(f"[FiLM] layer {layer_idx}")
                g = gamma[layer_idx].unsqueeze(1)     # (B,1,D)
                b = beta[layer_idx].unsqueeze(1)      # (B,1,D)
                x = g * x + b

            x = x + blk.drop_path(blk.mlp(blk.norm2(x)))  # FFN residual

        x = v.norm(x)
        h_cls = (x[:, 0] + x[:, 1]) / 2.0
        return h_cls

    def forward(self, x, device_id, site_id, m_rest):
        """
        x: (B,T,F) or (B,1,F,T)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1).transpose(2, 3)  # (B,1,F,T)

        h = self.forward_features(x, device_id, site_id, m_rest)
        logits = self.classifier(h)
        return logits