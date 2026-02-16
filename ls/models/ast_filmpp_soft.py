import torch
import torch.nn as nn
import torch.nn.functional as F
from ls.models.ast import ASTModel


class ASTFiLMPlusPlusSoft(nn.Module):
    """
    FiLM++ with soft learned factorization (v2 — fixed mask training).
    
    Key fixes over v1:
        1. FiLM generators use small random init (not zeros), so gradients
           flow to masks from the start.
        2. Mask logits are initialized with moderate scale (not deep in
           sigmoid saturation), so they can move early in training.
        3. Optional temperature annealing sharpens masks over training,
           moving from soft blending → near-binary selection.
        4. Coverage regularization encourages all dimensions to be used
           (prevents mask collapse to all-off).
    
    For each conditioned layer:
        w_dev, w_site, w_rest = σ(mask_dev/τ), σ(mask_site/τ), σ(mask_rest/τ)
        
        γ = 1 + w_dev ⊙ (γ_dev − 1) + w_site ⊙ (γ_site − 1) + w_rest ⊙ (γ_rest − 1)
        β = w_dev ⊙ β_dev + w_site ⊙ β_site + w_rest ⊙ β_rest
        
        x_modulated = γ ⊙ x + β
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
        mask_init_scale: float = 0.5,       # moderate init: σ(0.5)≈0.62, σ(-0.5)≈0.38
        mask_sparsity_lambda: float = 0.01,  # weight for overlap loss
        mask_coverage_lambda: float = 0.005, # weight for coverage loss
        per_layer_masks: bool = False,
        mask_temperature: float = 1.0,       # initial temperature (annealed externally)
        film_init_gain: float = 0.1,         # small but nonzero generator init
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
        self.mask_coverage_lambda = mask_coverage_lambda
        self.per_layer_masks = per_layer_masks
        self.film_init_gain = film_init_gain

        # Register temperature as a buffer (not a parameter — controlled externally)
        self.register_buffer('mask_temperature', torch.tensor(mask_temperature))

        # Metadata encoders
        self.dev_emb = nn.Embedding(num_devices, dev_emb_dim)
        self.site_emb = nn.Embedding(num_sites, site_emb_dim)

        self.dev_encoder = self._make_encoder(dev_emb_dim, metadata_hidden_dim, film_hidden_dim, dropout_p)
        self.site_encoder = self._make_encoder(site_emb_dim, metadata_hidden_dim, film_hidden_dim, dropout_p)
        self.rest_encoder = self._make_encoder(rest_dim, metadata_hidden_dim, film_hidden_dim, dropout_p)

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

        # Small random init instead of zeros — breaks the zero-gradient trap
        self._init_film_generators()

        # Soft masks
        if per_layer_masks:
            self.mask_dev = nn.ParameterDict({
                str(l): nn.Parameter(torch.zeros(D)) for l in self.conditioned_layers
            })
            self.mask_site = nn.ParameterDict({
                str(l): nn.Parameter(torch.zeros(D)) for l in self.conditioned_layers
            })
            self.mask_rest = nn.ParameterDict({
                str(l): nn.Parameter(torch.zeros(D)) for l in self.conditioned_layers
            })
            for l in self.conditioned_layers:
                self._init_masks(
                    self.mask_dev[str(l)],
                    self.mask_site[str(l)],
                    self.mask_rest[str(l)],
                    mask_init_scale,
                )
        else:
            self.mask_dev = nn.Parameter(torch.zeros(D))
            self.mask_site = nn.Parameter(torch.zeros(D))
            self.mask_rest = nn.Parameter(torch.zeros(D))
            self._init_masks(self.mask_dev, self.mask_site, self.mask_rest, mask_init_scale)

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(D),
            nn.Dropout(dropout_p),
            nn.Linear(D, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, num_labels),
        )

    # Initialization helpers
    @staticmethod
    def _make_encoder(in_dim, hidden_dim, out_dim, dropout_p):
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def _init_film_generators(self):
        """
        Small Xavier init instead of all-zeros.
        
        With zero init, generators produce g=1, b=0 exactly, so the mask
        logits are multiplied by zero and receive no gradient. Small random
        init ensures non-zero outputs from step 1, allowing gradients to
        reach the masks immediately.
        
        The gamma bias is still set to 0 (so gamma starts near 1) and the
        beta bias is 0 (beta starts near 0), preserving the near-identity
        property while allowing gradient flow.
        """
        for gen_dict in [self.dev_generators, self.site_generators, self.rest_generators]:
            for l in self.conditioned_layers:
                layer = gen_dict[str(l)]
                nn.init.xavier_uniform_(layer.weight, gain=self.film_init_gain)
                # Bias: gamma part = 0 (so 1 + 0 = 1), beta part = 0
                nn.init.zeros_(layer.bias)

    def _init_masks(self, mask_dev, mask_site, mask_rest, scale):
        """
        Moderate initialization scale.
        
        With scale=2.0, sigmoid(±2)≈0.88/0.12 — already deep in the flat
        region where gradients vanish. With scale=0.5, sigmoid(±0.5)≈0.62/0.38,
        giving masks room to move in either direction during early training.
        
        The thirds-based partition is still used as a structural prior, but
        the model can easily override it.
        """
        D = self.D
        third = D // 3

        mask_dev.data[:third] = scale
        mask_dev.data[third:] = -scale

        mask_site.data[:third] = -scale
        mask_site.data[third:2 * third] = scale
        mask_site.data[2 * third:] = -scale

        mask_rest.data[:2 * third] = -scale
        mask_rest.data[2 * third:] = scale

    # Mask access with temperature
    def _get_masks(self, layer_idx):
        """
        Get sigmoid-activated masks with temperature scaling.
        
        Lower temperature → sharper (more binary) masks.
        τ=1.0 is standard sigmoid; τ→0 approaches hard selection.
        """
        tau = self.mask_temperature.clamp(min=0.1)  # safety floor

        if self.per_layer_masks:
            w_dev = torch.sigmoid(self.mask_dev[str(layer_idx)] / tau)
            w_site = torch.sigmoid(self.mask_site[str(layer_idx)] / tau)
            w_rest = torch.sigmoid(self.mask_rest[str(layer_idx)] / tau)
        else:
            w_dev = torch.sigmoid(self.mask_dev / tau)
            w_site = torch.sigmoid(self.mask_site / tau)
            w_rest = torch.sigmoid(self.mask_rest / tau)
        return w_dev, w_site, w_rest

    def set_mask_temperature(self, tau):
        """Set mask temperature (call during training for annealing)."""
        self.mask_temperature.fill_(tau)

    # Regularization losses
    def mask_overlap_loss(self):
        """
        Penalizes pairwise overlap between masks.
        Encourages each dimension to be owned by at most one factor.
        """
        total_loss = 0.0
        count = 0

        layers = self.conditioned_layers if self.per_layer_masks else [self.conditioned_layers[0]]
        for l in layers:
            w_dev, w_site, w_rest = self._get_masks(l)
            total_loss += (w_dev * w_site).mean()
            total_loss += (w_dev * w_rest).mean()
            total_loss += (w_site * w_rest).mean()
            count += 3

        return total_loss / count

    def mask_coverage_loss(self):
        """
        Encourages every dimension to be claimed by at least one mask.
        Penalizes dimensions where max(w_dev, w_site, w_rest) is low.
        """
        total_loss = 0.0
        count = 0

        layers = self.conditioned_layers if self.per_layer_masks else [self.conditioned_layers[0]]
        for l in layers:
            w_dev, w_site, w_rest = self._get_masks(l)
            max_activation = torch.stack([w_dev, w_site, w_rest], dim=0).max(dim=0).values
            # Penalize low max activation (want every dim used by someone)
            total_loss += (1.0 - max_activation).mean()
            count += 1

        return total_loss / count

    def mask_regularization_loss(self):
        """
        Combined mask regularization: overlap + coverage.
        Call this and add to your main loss:
        
            loss = ce_loss + model.mask_regularization_loss()
        """
        loss = 0.0
        if self.mask_sparsity_lambda > 0:
            loss += self.mask_sparsity_lambda * self.mask_overlap_loss()
        if self.mask_coverage_lambda > 0:
            loss += self.mask_coverage_lambda * self.mask_coverage_loss()
        return loss

    # Forward pass
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

        # Encode metadata branches
        h_dev = self.dev_encoder(self.dev_emb(device_id))
        h_site = self.site_encoder(self.site_emb(site_id))
        h_rest = self.rest_encoder(m_rest)

        # Precompute FiLM params for conditioned layers
        params = {}
        for l in self.conditioned_layers:
            film_d = self.dev_generators[str(l)](h_dev)
            dg_d, db_d = film_d.chunk(2, dim=-1)

            film_s = self.site_generators[str(l)](h_site)
            dg_s, db_s = film_s.chunk(2, dim=-1)

            film_r = self.rest_generators[str(l)](h_rest)
            dg_r, db_r = film_r.chunk(2, dim=-1)

            # Store raw deltas (not 1+delta) to make the mask interaction clearer
            params[l] = (dg_d, db_d, dg_s, db_s, dg_r, db_r)

        film_info = {
            'masks': {},
            'params': {},
            'modulation_magnitude': {},
        } if return_film_info else None

        # Unroll transformer blocks
        for layer_idx, blk in enumerate(v.blocks):
            attn_out = blk.attn(blk.norm1(x))
            x = x + blk.drop_path(attn_out)

            normed = blk.norm2(x)

            if layer_idx in self.conditioned_set:
                dg_d, db_d, dg_s, db_s, dg_r, db_r = params[layer_idx]
                w_dev, w_site, w_rest = self._get_masks(layer_idx)

                # Masked combination: identity where no mask is active
                gamma = 1.0 + w_dev * dg_d + w_site * dg_s + w_rest * dg_r
                beta = w_dev * db_d + w_site * db_s + w_rest * db_r

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
                    print(f"  mask_dev active: {(w_dev > 0.5).sum().item()}/{self.D}")
                    print(f"  mask_site active: {(w_site > 0.5).sum().item()}/{self.D}")
                    print(f"  mask_rest active: {(w_rest > 0.5).sum().item()}/{self.D}")
                    print(f"  |dg_dev|={dg_d.abs().mean():.4f}  |dg_site|={dg_s.abs().mean():.4f}  |dg_rest|={dg_r.abs().mean():.4f}")

                pre_norm = normed.norm(dim=-1).mean().item() if return_film_info else None
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
        return self.classifier(h)

    def forward_with_film_info(self, x, device_id, site_id, m_rest):
        if x.dim() == 3:
            x = x.unsqueeze(1).transpose(2, 3)
        h, film_info = self.forward_features(x, device_id, site_id, m_rest, return_film_info=True)
        return self.classifier(h), film_info

    # Utilities
    def get_mask_stats(self):
        """Return mask statistics for logging."""
        stats = {}
        with torch.no_grad():
            layers = (
                self.conditioned_layers if self.per_layer_masks
                else [('shared', self.conditioned_layers[0])]
            )
            if self.per_layer_masks:
                layers = [(f'layer_{l}', l) for l in self.conditioned_layers]
            else:
                layers = [('shared', self.conditioned_layers[0])]

            for name, l in layers:
                w_dev, w_site, w_rest = self._get_masks(l)
                dominant = torch.stack([w_dev, w_site, w_rest]).argmax(dim=0)
                stats[name] = {
                    'dev_active': (w_dev > 0.5).sum().item(),
                    'site_active': (w_site > 0.5).sum().item(),
                    'rest_active': (w_rest > 0.5).sum().item(),
                    'overlap_dev_site': (w_dev * w_site).mean().item(),
                    'overlap_dev_rest': (w_dev * w_rest).mean().item(),
                    'overlap_site_rest': (w_site * w_rest).mean().item(),
                    'dev_dominant': (dominant == 0).sum().item(),
                    'site_dominant': (dominant == 1).sum().item(),
                    'rest_dominant': (dominant == 2).sum().item(),
                    'temperature': self.mask_temperature.item(),
                }
        return stats

    def get_optimizer_param_groups(self, lr=1e-4, mask_lr_multiplier=10.0):
        """
        Convenience method to create parameter groups with higher LR for masks.
        
        Usage:
            param_groups = model.get_optimizer_param_groups(lr=1e-4, mask_lr_multiplier=10)
            optimizer = torch.optim.AdamW(param_groups)
        """
        mask_params = []
        other_params = []

        for name, param in self.named_parameters():
            if 'mask_dev' in name or 'mask_site' in name or 'mask_rest' in name:
                mask_params.append(param)
            else:
                other_params.append(param)

        return [
            {'params': other_params, 'lr': lr},
            {'params': mask_params, 'lr': lr * mask_lr_multiplier, 'weight_decay': 0.0},
        ]


# Temperature annealing schedulet
class MaskTemperatureScheduler:
    """
    Anneals mask temperature from τ_start → τ_end over the training run.
    
    - Early training (high τ): soft masks, easy gradient flow, model learns
      which dimensions each factor needs.
    - Late training (low τ): masks sharpen toward binary, giving clean
      factorization at inference time.
    
    Usage:
        scheduler = MaskTemperatureScheduler(model, tau_start=1.0, tau_end=0.2, total_epochs=50)
        for epoch in range(50):
            scheduler.step(epoch)
            train(...)
    """

    def __init__(self, model, tau_start=1.0, tau_end=0.2, total_epochs=50, warmup_epochs=5):
        self.model = model
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Keep temperature high during warmup
            tau = self.tau_start
        else:
            # Cosine annealing from tau_start to tau_end
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)
            tau = self.tau_end + 0.5 * (self.tau_start - self.tau_end) * (1 + torch.cos(torch.tensor(progress * torch.pi)).item())

        self.model.set_mask_temperature(tau)
        return tau