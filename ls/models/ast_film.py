import torch
import torch.nn as nn
from ls.models.ast import ASTModel

class ASTFiLM(ASTModel):
    """
    AST with FiLM conditioning on selected Transformer layers.

    - Adds a metadata encoder g(m).
    - For a set of layers B, generates layer-specific gamma_l, beta_l.
    - Applies FiLM after the MHSA residual, before the FFN (as per your equations).
    """
    def __init__(
        self,
        metadata_dim: int,
        conditioned_layers=(0, 1, 2, 3),   # indices into self.v.blocks
        metadata_hidden_dim: int = 64,
        film_hidden_dim: int = 64,
        num_labels: int = 2,
        dropout_p: float = 0.3,
        ast_kwargs: dict = None,
    ):
        ast_kwargs = ast_kwargs or {}
        super().__init__(backbone_only=True, **ast_kwargs)

        self.metadata_dim = metadata_dim
        self.conditioned_layers = sorted(list(conditioned_layers))
        self.conditioned_layers_set = set(self.conditioned_layers)

        D = self.original_embedding_dim
        self.num_layers = len(self.v.blocks)

        # Metadata encoder h_m = g(m)
        self.metadata_encoder = nn.Sequential(
            nn.LayerNorm(metadata_dim),
            nn.Linear(metadata_dim, metadata_hidden_dim),
            nn.ReLU(),
            nn.Linear(metadata_hidden_dim, film_hidden_dim),
            nn.ReLU(),
        )

        # One FiLM generator per conditioned layer:
        # f_l: h_m -> [gamma_l || beta_l] in R^{2D}
        self.film_generators = nn.ModuleDict()
        for l in self.conditioned_layers:
            self.film_generators[str(l)] = nn.Linear(film_hidden_dim, 2 * D)

        # Classification head (same as baseline)
        self.classifier = nn.Sequential(
            nn.LayerNorm(D),
            nn.Dropout(dropout_p),
            nn.Linear(D, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, num_labels)
        )

    def forward_features(self, x, m):
        """
        x: (B, 1, F, T)  (we'll handle 3D in forward())
        m: (B, M) metadata
        returns: h_CLS in R^D (FiLM-conditioned)
        """
        B = x.shape[0]

        # Patch embedding: (B, N, D)
        x = self.v.patch_embed(x)

        # CLS + dist tokens and positional embeddings
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        # Encode metadata once, then generate FiLM params per layer
        h_m = self.metadata_encoder(m)   # (B, film_hidden_dim)

        gamma = {}
        beta = {}
        for l in self.conditioned_layers:
            film = self.film_generators[str(l)](h_m)  # (B, 2D)
            g_l, b_l = film.chunk(2, dim=-1)          # (B, D), (B, D)
            gamma[l] = g_l
            beta[l] = b_l

        # Manually unroll each Transformer block
        for layer_idx, blk in enumerate(self.v.blocks):
            # MHSA sublayer with pre-norm
            attn_out = blk.attn(blk.norm1(x))
            x = x + blk.drop_path(attn_out)   # Z_tilde_l

            # Apply FiLM after MHSA for selected layers
            if layer_idx in self.conditioned_layers_set:
                print(f"Applying FiLM at layer {layer_idx}")
                g_l = gamma[layer_idx].unsqueeze(1)   # (B, 1, D) -> broadcast over tokens
                b_l = beta[layer_idx].unsqueeze(1)    # (B, 1, D)
                x = g_l * x + b_l                     # Eq. (14–15): hat{Z}_l

            # FFN + residual
            x = x + blk.drop_path(blk.mlp(blk.norm2(x)))

        x = self.v.norm(x)
        h_cls = (x[:, 0] + x[:, 1]) / 2  # pooled token (average CLS + dist)
        return h_cls

    def forward(self, x, m):
        """
        x: (B, T, F) or (B, 1, F, T)
        m: (B, M) metadata
        """
        if x.dim() == 3:
            x = x.unsqueeze(1).transpose(2, 3)  # (B, 1, F, T)

        h_cls = self.forward_features(x, m)
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

# astfilm = ASTFiLM(
#     ast_kwargs=ast_kwargs, 
#     metadata_dim=10,
#     conditioned_layers=range(12), # all layers
#     metadata_hidden_dim=64, 
#     film_hidden_dim=64,
#     dropout_p=0.3, 
#     num_labels=2
# ).to(DEVICE)
# print(astfilm)
# metadata  = torch.randn(2, 10).to(DEVICE)  # (B, M)
# out = astfilm(dummy_input, metadata)  # (2, 2)
# print(out.shape)  # torch.Size([2, 2])