import torch
import torch.nn as nn
from ls.models.ast import ASTModel

class ASTFiLMPlusPlus(nn.Module):
    """
    FiLM++ grouped conditioning:

      Split hidden dim D = D_dev + D_site + D_rest

      - dev branch uses m_dev = Emb_dev(device_id) -> h_dev -> [Δγ_dev, Δβ_dev] in R^{2D_dev}
      - site branch uses m_site = Emb_site(site_id) -> h_site -> [Δγ_site, Δβ_site] in R^{2D_site}
      - rest branch uses m_rest (float) -> h_rest -> [Δγ_rest, Δβ_rest] in R^{2D_rest}

    Apply (after MHSA residual, before FFN):
      x_dev  = (1+Δγ_dev)  ⊙ x_dev  + Δβ_dev
      x_site = (1+Δγ_site) ⊙ x_site + Δβ_site
      x_rest = (1+Δγ_rest) ⊙ x_rest + Δβ_rest
      x = concat([x_dev, x_site, x_rest], dim=-1)
    """

    def __init__(
        self,
        ast_kwargs: dict,
        num_devices: int,
        num_sites: int,
        rest_dim: int,
        D_dev: int = 128,
        D_site: int = 128,
        conditioned_layers=(10, 11, 12),
        dev_emb_dim: int = 4,
        site_emb_dim: int = 4,
        metadata_hidden_dim: int = 64,
        film_hidden_dim: int = 64,
        dropout_p: float = 0.3,
        num_labels: int = 2,
        debug_film: bool = False,
    ):
        super().__init__()

        self.ast = ASTModel(backbone_only=True, **ast_kwargs)
        D = self.ast.original_embedding_dim
        self.D = D

        assert D_dev > 0 and D_site > 0 and (D_dev + D_site) < D, \
            "Need D_rest = D - D_dev - D_site > 0"

        self.D_dev = D_dev
        self.D_site = D_site
        self.D_rest = D - D_dev - D_site

        self.debug_film = debug_film
        self.conditioned_layers = sorted(list(conditioned_layers))
        self.conditioned_set = set(self.conditioned_layers)

        # Categorical embeddings
        self.dev_emb = nn.Embedding(num_devices, dev_emb_dim)
        self.site_emb = nn.Embedding(num_sites, site_emb_dim)

        # Encoders per branch
        self.dev_encoder = nn.Sequential(
            nn.LayerNorm(dev_emb_dim),
            nn.Linear(dev_emb_dim, metadata_hidden_dim),
            nn.ReLU(),
            nn.Linear(metadata_hidden_dim, film_hidden_dim),
            nn.ReLU(),
        )
        self.site_encoder = nn.Sequential(
            nn.LayerNorm(site_emb_dim),
            nn.Linear(site_emb_dim, metadata_hidden_dim),
            nn.ReLU(),
            nn.Linear(metadata_hidden_dim, film_hidden_dim),
            nn.ReLU(),
        )
        self.rest_encoder = nn.Sequential(
            nn.LayerNorm(rest_dim),
            nn.Linear(rest_dim, metadata_hidden_dim),
            nn.ReLU(),
            nn.Linear(metadata_hidden_dim, film_hidden_dim),
            nn.ReLU(),
        )

        # Per-layer generators per branch
        self.dev_generators = nn.ModuleDict({
            str(l): nn.Linear(film_hidden_dim, 2 * self.D_dev) for l in self.conditioned_layers
        })
        self.site_generators = nn.ModuleDict({
            str(l): nn.Linear(film_hidden_dim, 2 * self.D_site) for l in self.conditioned_layers
        })
        self.rest_generators = nn.ModuleDict({
            str(l): nn.Linear(film_hidden_dim, 2 * self.D_rest) for l in self.conditioned_layers
        })

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(D),
            nn.Dropout(dropout_p),
            nn.Linear(D, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, num_labels),
        )

    def _prep_tokens(self, x):
        B = x.shape[0]
        v = self.ast.v
        x = v.patch_embed(x)  # (B, N, D)

        cls_tokens = v.cls_token.expand(B, -1, -1)
        dist_token = v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + v.pos_embed
        x = v.pos_drop(x)
        return x

    def forward_features(self, x, device_id, site_id, m_rest):
        v = self.ast.v
        x = self._prep_tokens(x)

        # Branch embeddings / encodings
        m_dev = self.dev_emb(device_id)          # (B, dev_emb_dim)
        m_site = self.site_emb(site_id)          # (B, site_emb_dim)

        h_dev = self.dev_encoder(m_dev)          # (B, H)
        h_site = self.site_encoder(m_site)       # (B, H)
        h_rest = self.rest_encoder(m_rest)       # (B, H)

        # Precompute params for each conditioned layer
        params = {}
        for l in self.conditioned_layers:
            # dev
            film_d = self.dev_generators[str(l)](h_dev)     # (B, 2*D_dev)
            dg_d, db_d = film_d.chunk(2, dim=-1)
            g_dev = 1.0 + dg_d
            b_dev = db_d

            # site
            film_s = self.site_generators[str(l)](h_site)   # (B, 2*D_site)
            dg_s, db_s = film_s.chunk(2, dim=-1)
            g_site = 1.0 + dg_s
            b_site = db_s

            # rest
            film_r = self.rest_generators[str(l)](h_rest)   # (B, 2*D_rest)
            dg_r, db_r = film_r.chunk(2, dim=-1)
            g_rest = 1.0 + dg_r
            b_rest = db_r

            params[l] = (g_dev, b_dev, g_site, b_site, g_rest, b_rest)

        # Unroll transformer blocks
        for layer_idx, blk in enumerate(v.blocks):
            attn_out = blk.attn(blk.norm1(x))
            x = x + blk.drop_path(attn_out)

            if layer_idx in self.conditioned_set:
                if self.debug_film:
                    print(f"[FiLM++] layer {layer_idx}")

                g_dev, b_dev, g_site, b_site, g_rest, b_rest = params[layer_idx]

                # Split feature dimension
                x_dev = x[:, :, :self.D_dev]                                  # (B,T,D_dev)
                x_site = x[:, :, self.D_dev:self.D_dev + self.D_site]         # (B,T,D_site)
                x_rest = x[:, :, self.D_dev + self.D_site:]                   # (B,T,D_rest)

                # Broadcast over tokens
                x_dev  = g_dev.unsqueeze(1)  * x_dev  + b_dev.unsqueeze(1)
                x_site = g_site.unsqueeze(1) * x_site + b_site.unsqueeze(1)
                x_rest = g_rest.unsqueeze(1) * x_rest + b_rest.unsqueeze(1)

                x = torch.cat([x_dev, x_site, x_rest], dim=-1)

            x = x + blk.drop_path(blk.mlp(blk.norm2(x)))

        x = v.norm(x)
        h_cls = (x[:, 0] + x[:, 1]) / 2.0
        return h_cls

    def forward(self, x, device_id, site_id, m_rest):
        if x.dim() == 3:
            x = x.unsqueeze(1).transpose(2, 3)  # (B,1,F,T)

        h = self.forward_features(x, device_id, site_id, m_rest)
        logits = self.classifier(h)
        return logits
