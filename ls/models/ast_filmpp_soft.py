import torch
import torch.nn as nn
from ls.models.ast import ASTModel


class ASTFiLMPlusPlusSoft(nn.Module):
    """
    FiLM++ with soft learned factorization.
    
    Instead of hard-splitting the hidden dimension, we learn soft masks
    that determine which dimensions each metadata source should modulate.
    
    For each conditioned layer:
        w_dev, w_site, w_rest = sigmoid(mask_dev), sigmoid(mask_site), sigmoid(mask_rest)
        
        γ = w_dev ⊙ γ_dev + w_site ⊙ γ_site + w_rest ⊙ γ_rest
        β = w_dev ⊙ β_dev + w_site ⊙ β_site + w_rest ⊙ β_rest
        
        x_modulated = γ ⊙ x + β
    
    The masks are initialized to encourage separation but can learn overlap.
    This is more principled than hard splits since pretrained features aren't
    naturally factorized by metadata source.
    """

    def __init__(
        self,
        ast_kwargs: dict,
        num_devices: int,
        num_sites: int,
        rest_dim: int,
        conditioned_layers: tuple = (10, 11),
        dev_emb_dim: int = 4,
        site_emb_dim: int = 4,
        metadata_hidden_dim: int = 64,
        film_hidden_dim: int = 64,
        dropout_p: float = 0.3,
        num_labels: int = 2,
        mask_init_scale: float = 2.0,  # higher = sharper initial separation
        mask_sparsity_lambda: float = 0.01,  # regularization for mask overlap
        per_layer_masks: bool = False,  # if True, learn separate masks per layer
        debug_film: bool = False,
    ):
        super().__init__()

        self.ast = ASTModel(backbone_only=True, **ast_kwargs)
        D = self.ast.original_embedding_dim
        self.D = D

        self.debug_film = debug_film
        self.conditioned_layers = sorted(list(conditioned_layers))
        self.conditioned_set = set(self.conditioned_layers)
        self.mask_sparsity_lambda = mask_sparsity_lambda
        self.per_layer_masks = per_layer_masks

        # Categorical embeddings
        self.dev_emb = nn.Embedding(num_devices, dev_emb_dim)
        self.site_emb = nn.Embedding(num_sites, site_emb_dim)

        # Branch encoders
        self.dev_encoder = nn.Sequential(
            nn.LayerNorm(dev_emb_dim),
            nn.Linear(dev_emb_dim, metadata_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(metadata_hidden_dim, film_hidden_dim),
            nn.ReLU(),
        )
        self.site_encoder = nn.Sequential(
            nn.LayerNorm(site_emb_dim),
            nn.Linear(site_emb_dim, metadata_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(metadata_hidden_dim, film_hidden_dim),
            nn.ReLU(),
        )
        self.rest_encoder = nn.Sequential(
            nn.LayerNorm(rest_dim),
            nn.Linear(rest_dim, metadata_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(metadata_hidden_dim, film_hidden_dim),
            nn.ReLU(),
        )

        # FiLM generators: each outputs (γ, β) for full D dimensions
        self.dev_generators = nn.ModuleDict({
            str(l): nn.Linear(film_hidden_dim, 2 * D) for l in self.conditioned_layers
        })
        self.site_generators = nn.ModuleDict({
            str(l): nn.Linear(film_hidden_dim, 2 * D) for l in self.conditioned_layers
        })
        self.rest_generators = nn.ModuleDict({
            str(l): nn.Linear(film_hidden_dim, 2 * D) for l in self.conditioned_layers
        })

        # Initialize generators for identity (γ=1, β=0) at start
        self._init_film_generators()

        # Soft masks: learnable logits, sigmoid gives [0,1] weights
        if per_layer_masks:
            # Separate masks for each conditioned layer
            self.mask_dev = nn.ParameterDict({
                str(l): nn.Parameter(torch.zeros(D)) for l in self.conditioned_layers
            })
            self.mask_site = nn.ParameterDict({
                str(l): nn.Parameter(torch.zeros(D)) for l in self.conditioned_layers
            })
            self.mask_rest = nn.ParameterDict({
                str(l): nn.Parameter(torch.zeros(D)) for l in self.conditioned_layers
            })
            # Initialize each layer's masks
            for l in self.conditioned_layers:
                self._init_masks(
                    self.mask_dev[str(l)], 
                    self.mask_site[str(l)], 
                    self.mask_rest[str(l)],
                    mask_init_scale
                )
        else:
            # Shared masks across layers
            self.mask_dev = nn.Parameter(torch.zeros(D))
            self.mask_site = nn.Parameter(torch.zeros(D))
            self.mask_rest = nn.Parameter(torch.zeros(D))
            self._init_masks(self.mask_dev, self.mask_site, self.mask_rest, mask_init_scale)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(D),
            nn.Dropout(dropout_p),
            nn.Linear(D, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, num_labels),
        )

    def _init_masks(self, mask_dev, mask_site, mask_rest, scale):
        """
        Initialize masks to encourage initial separation into thirds.
        After sigmoid:
            - mask_dev high for dims [0, D/3)
            - mask_site high for dims [D/3, 2D/3)  
            - mask_rest high for dims [2D/3, D)
        """
        D = self.D
        third = D // 3

        # Set logits so sigmoid gives ~0.88 for "on" and ~0.12 for "off" at scale=2
        mask_dev.data[:third] = scale
        mask_dev.data[third:] = -scale

        mask_site.data[:third] = -scale
        mask_site.data[third:2*third] = scale
        mask_site.data[2*third:] = -scale

        mask_rest.data[:2*third] = -scale
        mask_rest.data[2*third:] = scale

    def _init_film_generators(self):
        """Initialize FiLM generators for near-identity at start."""
        for gen_dict in [self.dev_generators, self.site_generators, self.rest_generators]:
            for l in self.conditioned_layers:
                layer = gen_dict[str(l)]
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _get_masks(self, layer_idx):
        """Get sigmoid-activated masks for a given layer."""
        if self.per_layer_masks:
            w_dev = torch.sigmoid(self.mask_dev[str(layer_idx)])
            w_site = torch.sigmoid(self.mask_site[str(layer_idx)])
            w_rest = torch.sigmoid(self.mask_rest[str(layer_idx)])
        else:
            w_dev = torch.sigmoid(self.mask_dev)
            w_site = torch.sigmoid(self.mask_site)
            w_rest = torch.sigmoid(self.mask_rest)
        return w_dev, w_site, w_rest

    def mask_overlap_loss(self):
        """
        Regularization term encouraging masks to be disjoint.
        Penalizes cases where multiple masks are active for the same dimension.
        
        Returns:
            loss: scalar tensor
        """
        total_loss = 0.0
        num_terms = 0

        if self.per_layer_masks:
            for l in self.conditioned_layers:
                w_dev, w_site, w_rest = self._get_masks(l)
                # Pairwise overlap penalties
                total_loss += (w_dev * w_site).mean()
                total_loss += (w_dev * w_rest).mean()
                total_loss += (w_site * w_rest).mean()
                num_terms += 3
        else:
            w_dev, w_site, w_rest = self._get_masks(self.conditioned_layers[0])
            total_loss += (w_dev * w_site).mean()
            total_loss += (w_dev * w_rest).mean()
            total_loss += (w_site * w_rest).mean()
            num_terms = 3

        return total_loss / num_terms

    def _prep_tokens(self, x):
        B = x.shape[0]
        v = self.ast.v
        x = v.patch_embed(x)

        cls_tokens = v.cls_token.expand(B, -1, -1)
        dist_token = v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + v.pos_embed
        x = v.pos_drop(x)
        return x

    def forward_features(self, x, device_id, site_id, m_rest, return_film_info=False):
        v = self.ast.v
        x = self._prep_tokens(x)

        # Encode each metadata branch
        m_dev = self.dev_emb(device_id)
        m_site = self.site_emb(site_id)

        h_dev = self.dev_encoder(m_dev)
        h_site = self.site_encoder(m_site)
        h_rest = self.rest_encoder(m_rest)

        # Precompute FiLM params for all conditioned layers
        params = {}
        for l in self.conditioned_layers:
            # Device branch
            film_d = self.dev_generators[str(l)](h_dev)
            dg_d, db_d = film_d.chunk(2, dim=-1)
            g_dev, b_dev = 1.0 + dg_d, db_d

            # Site branch
            film_s = self.site_generators[str(l)](h_site)
            dg_s, db_s = film_s.chunk(2, dim=-1)
            g_site, b_site = 1.0 + dg_s, db_s

            # Rest branch
            film_r = self.rest_generators[str(l)](h_rest)
            dg_r, db_r = film_r.chunk(2, dim=-1)
            g_rest, b_rest = 1.0 + dg_r, db_r

            params[l] = (g_dev, b_dev, g_site, b_site, g_rest, b_rest)

        # For visualization
        film_info = {
            'masks': {},
            'params': {},
            'modulation_magnitude': {}
        } if return_film_info else None

        # Unroll transformer blocks
        for layer_idx, blk in enumerate(v.blocks):
            # Attention block
            attn_out = blk.attn(blk.norm1(x))
            x = x + blk.drop_path(attn_out)

            # FFN block with FiLM++ conditioning
            normed = blk.norm2(x)

            if layer_idx in self.conditioned_set:
                g_dev, b_dev, g_site, b_site, g_rest, b_rest = params[layer_idx]
                w_dev, w_site, w_rest = self._get_masks(layer_idx)

                # Combine with soft masks: weighted sum of modulations
                # gamma = w_dev * g_dev + w_site * g_site + w_rest * g_rest
                # But we want identity (gamma=1) where no mask is active, so:
                # gamma = 1 + w_dev*(g_dev-1) + w_site*(g_site-1) + w_rest*(g_rest-1)
                # Since g_dev = 1 + dg_d, this simplifies to:
                gamma = 1.0 + w_dev * (g_dev - 1.0) + w_site * (g_site - 1.0) + w_rest * (g_rest - 1.0)
                beta = w_dev * b_dev + w_site * b_site + w_rest * b_rest

                if return_film_info:
                    film_info['masks'][layer_idx] = {
                        'dev': w_dev.detach().cpu(),
                        'site': w_site.detach().cpu(),
                        'rest': w_rest.detach().cpu(),
                    }
                    film_info['params'][layer_idx] = {
                        'gamma': gamma.detach().cpu(),
                        'beta': beta.detach().cpu(),
                    }

                if self.debug_film:
                    print(f"[FiLM++Soft] layer {layer_idx}")
                    print(f"  mask_dev active dims: {(w_dev > 0.5).sum().item()}")
                    print(f"  mask_site active dims: {(w_site > 0.5).sum().item()}")
                    print(f"  mask_rest active dims: {(w_rest > 0.5).sum().item()}")

                pre_norm = normed.norm(dim=-1).mean().item() if return_film_info else None
                
                # Apply modulation (broadcast over tokens)
                normed = gamma.unsqueeze(1) * normed + beta.unsqueeze(1)

                if return_film_info:
                    post_norm = normed.norm(dim=-1).mean().item()
                    film_info['modulation_magnitude'][layer_idx] = post_norm - pre_norm

            x = x + blk.drop_path(blk.mlp(normed))

        x = v.norm(x)
        h_cls = (x[:, 0] + x[:, 1]) / 2.0

        if return_film_info:
            return h_cls, film_info
        return h_cls

    def forward(self, x, device_id, site_id, m_rest):
        if x.dim() == 3:
            x = x.unsqueeze(1).transpose(2, 3)

        h = self.forward_features(x, device_id, site_id, m_rest)
        logits = self.classifier(h)
        return logits

    def forward_with_film_info(self, x, device_id, site_id, m_rest):
        """Forward pass returning predictions and FiLM info for visualization."""
        if x.dim() == 3:
            x = x.unsqueeze(1).transpose(2, 3)

        h, film_info = self.forward_features(x, device_id, site_id, m_rest, return_film_info=True)
        logits = self.classifier(h)
        return logits, film_info

    def get_mask_stats(self):
        """Return statistics about learned masks for logging/visualization."""
        stats = {}
        
        with torch.no_grad():
            if self.per_layer_masks:
                for l in self.conditioned_layers:
                    w_dev, w_site, w_rest = self._get_masks(l)
                    stats[f'layer_{l}'] = {
                        'dev_active': (w_dev > 0.5).sum().item(),
                        'site_active': (w_site > 0.5).sum().item(),
                        'rest_active': (w_rest > 0.5).sum().item(),
                        'overlap_dev_site': (w_dev * w_site).mean().item(),
                        'overlap_dev_rest': (w_dev * w_rest).mean().item(),
                        'overlap_site_rest': (w_site * w_rest).mean().item(),
                    }
            else:
                w_dev, w_site, w_rest = self._get_masks(self.conditioned_layers[0])
                stats['shared'] = {
                    'dev_active': (w_dev > 0.5).sum().item(),
                    'site_active': (w_site > 0.5).sum().item(),
                    'rest_active': (w_rest > 0.5).sum().item(),
                    'overlap_dev_site': (w_dev * w_site).mean().item(),
                    'overlap_dev_rest': (w_dev * w_rest).mean().item(),
                    'overlap_site_rest': (w_site * w_rest).mean().item(),
                }
        
        return stats