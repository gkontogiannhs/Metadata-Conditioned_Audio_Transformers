import torch
import torch.nn as nn
from ls.models.ast import ASTModel

class ASTFiLMPlusPlus(ASTModel):
    """
    FiLM++: factor-aligned grouped FiLM with K=3 groups:
      - device group    (D_dev)
      - site group      (D_site)
      - rest group      (D_rest)

    Metadata inputs are passed separately as (m_dev, m_site, m_rest).
    """
    def __init__(
        self,
        dev_metadata_dim: int,
        site_metadata_dim: int,
        rest_metadata_dim: int,
        D_dev: int,
        D_site: int,
        conditioned_layers=(0, 1, 2, 3),
        metadata_hidden_dim: int = 64,
        film_hidden_dim: int = 64,
        num_labels: int = 2,
        dropout_p: float = 0.3,
        ast_kwargs: dict = None,
    ):
        ast_kwargs = ast_kwargs or {}
        super().__init__(backbone_only=True, **ast_kwargs)

        self.conditioned_layers = sorted(list(conditioned_layers))
        self.conditioned_layers_set = set(self.conditioned_layers)

        D_total = self.original_embedding_dim
        assert D_dev + D_site <= D_total, "D_dev + D_site must be <= total embedding dim"
        D_rest = D_total - D_dev - D_site

        self.D_dev = D_dev
        self.D_site = D_site
        self.D_rest = D_rest
        self.D_total = D_total

        # --- Metadata encoders for each factor ---
        self.dev_encoder = nn.Sequential(
            nn.LayerNorm(dev_metadata_dim),
            nn.Linear(dev_metadata_dim, metadata_hidden_dim),
            nn.ReLU(),
            nn.Linear(metadata_hidden_dim, film_hidden_dim),
            nn.ReLU(),
        )
        self.site_encoder = nn.Sequential(
            nn.LayerNorm(site_metadata_dim),
            nn.Linear(site_metadata_dim, metadata_hidden_dim),
            nn.ReLU(),
            nn.Linear(metadata_hidden_dim, film_hidden_dim),
            nn.ReLU(),
        )
        self.rest_encoder = nn.Sequential(
            nn.LayerNorm(rest_metadata_dim),
            nn.Linear(rest_metadata_dim, metadata_hidden_dim),
            nn.ReLU(),
            nn.Linear(metadata_hidden_dim, film_hidden_dim),
            nn.ReLU(),
        )

        # --- FiLM generators per group and per conditioned layer ---
        self.dev_film = nn.ModuleDict()
        self.site_film = nn.ModuleDict()
        self.rest_film = nn.ModuleDict()
        for l in self.conditioned_layers:
            self.dev_film[str(l)] = nn.Linear(film_hidden_dim, 2 * D_dev)
            self.site_film[str(l)] = nn.Linear(film_hidden_dim, 2 * D_site)
            self.rest_film[str(l)] = nn.Linear(film_hidden_dim, 2 * D_rest)

        # Classification head (same style as baseline)
        self.classifier = nn.Sequential(
            nn.LayerNorm(D_total),
            nn.Dropout(dropout_p),
            nn.Linear(D_total, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, num_labels),
        )

    def _apply_filmpp_grouped(self, x, gammas, betas):
        """
        x:      (B, T, D_total)
        gammas: dict with 'dev','site','rest' tensors (B, D_group)
        betas:  same as above

        Applies group-wise FiLM and returns (B, T, D_total).
        """
        B, T, D = x.shape
        D_dev, D_site, D_rest = self.D_dev, self.D_site, self.D_rest

        x_dev, x_site, x_rest = torch.split(
            x, [D_dev, D_site, D_rest], dim=-1
        )  # each (B, T, D_group)

        # Broadcast gammas/betas over tokens
        g_dev = gammas["dev"].unsqueeze(1)   # (B, 1, D_dev)
        b_dev = betas["dev"].unsqueeze(1)
        g_site = gammas["site"].unsqueeze(1) # (B, 1, D_site)
        b_site = betas["site"].unsqueeze(1)
        g_rest = gammas["rest"].unsqueeze(1) # (B, 1, D_rest)
        b_rest = betas["rest"].unsqueeze(1)

        x_dev_hat = g_dev * x_dev + b_dev
        x_site_hat = g_site * x_site + b_site
        x_rest_hat = g_rest * x_rest + b_rest

        x_hat = torch.cat([x_dev_hat, x_site_hat, x_rest_hat], dim=-1)  # (B, T, D_total)
        return x_hat

    def forward_features(self, x, m_dev, m_site, m_rest):
        """
        x:      (B, 1, F, T)
        m_dev:  (B, dev_metadata_dim)
        m_site: (B, site_metadata_dim)
        m_rest: (B, rest_metadata_dim)
        """
        B = x.shape[0]

        # Patch embedding
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        # Encode metadata for each factor
        h_dev = self.dev_encoder(m_dev)
        h_site = self.site_encoder(m_site)
        h_rest = self.rest_encoder(m_rest)

        # Precompute gamma,beta for each conditioned layer
        gamma_dev, beta_dev = {}, {}
        gamma_site, beta_site = {}, {}
        gamma_rest, beta_rest = {}, {}

        for l in self.conditioned_layers:
            dev_params = self.dev_film[str(l)](h_dev)   # (B, 2*D_dev)
            site_params = self.site_film[str(l)](h_site) # (B, 2*D_site)
            rest_params = self.rest_film[str(l)](h_rest) # (B, 2*D_rest)

            g_dev, b_dev = dev_params.chunk(2, dim=-1)
            g_site, b_site = site_params.chunk(2, dim=-1)
            g_rest, b_rest = rest_params.chunk(2, dim=-1)

            gamma_dev[l], beta_dev[l] = g_dev, b_dev
            gamma_site[l], beta_site[l] = g_site, b_site
            gamma_rest[l], beta_rest[l] = g_rest, b_rest

        # Unroll ViT blocks with FiLM++ after MHSA
        for layer_idx, blk in enumerate(self.v.blocks):
            attn_out = blk.attn(blk.norm1(x))
            x = x + blk.drop_path(attn_out)   # (B, T, D_total)

            if layer_idx in self.conditioned_layers_set:
                gammas = {
                    "dev": gamma_dev[layer_idx],
                    "site": gamma_site[layer_idx],
                    "rest": gamma_rest[layer_idx],
                }
                betas = {
                    "dev": beta_dev[layer_idx],
                    "site": beta_site[layer_idx],
                    "rest": beta_rest[layer_idx],
                }
                x = self._apply_filmpp_grouped(x, gammas, betas)

            x = x + blk.drop_path(blk.mlp(blk.norm2(x)))

        x = self.v.norm(x)
        h_cls = (x[:, 0] + x[:, 1]) / 2
        return h_cls

    def forward(self, x, m_dev, m_site, m_rest):
        """
        x:      (B, T, F) or (B, 1, F, T)
        m_dev:  (B, dev_metadata_dim)
        m_site: (B, site_metadata_dim)
        m_rest: (B, rest_metadata_dim)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1).transpose(2, 3)

        h_cls = self.forward_features(x, m_dev, m_site, m_rest)
        logits = self.classifier(h_cls)
        return logits
    

# ast_kwargs = dict(
#     label_dim=2,          # unused since backbone_only=True
#     fstride=10,
#     tstride=10,
#     input_fdim=128,
#     input_tdim=1024,
#     imagenet_pretrain=True,
#     audioset_pretrain=True,
#     audioset_ckpt_path='/Users/gkont/Documents/Code/pretrained_models/audioset_10_10_0.4593.pth',
#     model_size='base384',
#     verbose=True,
# )

# astfilmpp = ASTFiLMPlusPlus(
#     dev_metadata_dim=4,
#     site_metadata_dim=7,
#     rest_metadata_dim=3,
#     D_dev=128,
#     D_site=128,
#     ast_kwargs=ast_kwargs,
#     conditioned_layers=(10, 11, 12), # last 3 layers
#     metadata_hidden_dim=64, 
#     film_hidden_dim=64,
#     dropout_p=0.3, 
#     num_labels=2
# ).to(DEVICE)
# print(astfilmpp)

# dev_metadata = torch.randn(2, 4).to(DEVICE)
# site_metadata = torch.randn(2, 7).to(DEVICE)
# rest_metadata = torch.randn(2, 3).to(DEVICE)
# out = astfilmpp(dummy_input, dev_metadata, site_metadata, rest_metadata)  # (2, 2)
# print(out.shape)  # torch.Size([2, 2])