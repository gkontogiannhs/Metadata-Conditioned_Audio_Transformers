import torch
import torch.nn as nn
from ls.models.ast import ASTModel


class ImprovedContinuousEncoder(nn.Module):
    """
    Enhanced encoder for continuous metadata (age, BMI, duration).
    Uses per-feature learned transformations instead of simple concatenation.
    """
    def __init__(self, num_features=3, hidden_dim=16, dropout_p=0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
        per_feature_dim = max(1, hidden_dim // num_features)
        
        self.feature_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, per_feature_dim),
                nn.LayerNorm(per_feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p)
            ) for _ in range(num_features)
        ])
        
        concat_dim = per_feature_dim * num_features
        
        self.combiner = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        features = x.chunk(self.num_features, dim=-1)
        encoded = [enc(f) for enc, f in zip(self.feature_encoders, features)]
        combined = torch.cat(encoded, dim=-1)
        return self.combiner(combined)


class ASTTAFiLM(nn.Module):
    """
    Token-Aware FiLM (TAFiLM) conditioned AST.
    
    Unlike standard FiLM which applies uniform modulation across all tokens,
    TAFiLM generates separate FiLM parameters for:
    - CLS/distillation tokens (global representation for classification)
    - Patch tokens (local spectro-temporal features)
    
    Args:
        ast_kwargs: Arguments for base AST model
        num_devices: Number of recording devices
        num_sites: Number of auscultation sites
        rest_dim: Dimension of continuous metadata (age, bmi, duration)
        dev_emb_dim: Device embedding dimension
        site_emb_dim: Site embedding dimension
        conditioned_layers: Which transformer layers to condition
        metadata_hidden_dim: Hidden dim for metadata encoder
        film_hidden_dim: Hidden dim for FiLM generators
        dropout_p: Dropout probability
        debug: Print debug information
        condition_on_device: Use device metadata
        condition_on_site: Use site metadata
        condition_on_rest: Use continuous metadata
        use_improved_continuous_encoder: Use per-feature encoding for continuous
    """

    def __init__(
        self,
        ast_kwargs: dict,
        num_devices: int,
        num_sites: int,
        rest_dim: int,
        dev_emb_dim: int = 8,
        site_emb_dim: int = 8,
        conditioned_layers: tuple = (10, 11),
        metadata_hidden_dim: int = 64,
        film_hidden_dim: int = 64,
        dropout_p: float = 0.3,
        debug: bool = False,
        condition_on_device: bool = True,
        condition_on_site: bool = True,
        condition_on_rest: bool = True,
        use_improved_continuous_encoder: bool = True,
    ):
        super().__init__()

        # Build AST backbone (without classification head)
        self.ast = ASTModel(backbone_only=True, **ast_kwargs)
        self.num_labels = ast_kwargs["label_dim"]
        D = self.ast.original_embedding_dim  # 768 for base
        self.D = D
        self.debug = debug

        # Store conditioning config
        self.conditioned_layers = sorted(list(conditioned_layers))
        self.conditioned_set = set(self.conditioned_layers)
        self.condition_on_device = condition_on_device
        self.condition_on_site = condition_on_site
        self.condition_on_rest = condition_on_rest

        # Validate: at least one conditioning source
        if not any([condition_on_device, condition_on_site, condition_on_rest]):
            raise ValueError("At least one conditioning source must be enabled!")
        
        # Device embedding
        if condition_on_device:
            self.dev_emb = nn.Embedding(num_devices, dev_emb_dim)
        else:
            self.dev_emb = None
            dev_emb_dim = 0

        # Site embedding
        if condition_on_site:
            self.site_emb = nn.Embedding(num_sites, site_emb_dim)
        else:
            self.site_emb = None
            site_emb_dim = 0

        # Continuous metadata encoder
        if condition_on_rest:
            if use_improved_continuous_encoder:
                self.rest_encoder = ImprovedContinuousEncoder(
                    num_features=rest_dim,
                    hidden_dim=film_hidden_dim // 2,
                    dropout_p=dropout_p * 0.5
                )
                rest_encoded_dim = film_hidden_dim // 2
            else:
                self.rest_encoder = None
                rest_encoded_dim = rest_dim
        else:
            self.rest_encoder = None
            rest_encoded_dim = 0

        # Total metadata dimension
        meta_dim = dev_emb_dim + site_emb_dim + rest_encoded_dim
        
        self.metadata_encoder = nn.Sequential(
            nn.LayerNorm(meta_dim),
            nn.Linear(meta_dim, metadata_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(metadata_hidden_dim, film_hidden_dim),
            nn.GELU(),
        )
        
        # Separate generators for CLS tokens and patch tokens
        self.film_generators_cls = nn.ModuleDict({
            str(l): nn.Linear(film_hidden_dim, 2 * D)
            for l in self.conditioned_layers
        })
        
        self.film_generators_patch = nn.ModuleDict({
            str(l): nn.Linear(film_hidden_dim, 2 * D)
            for l in self.conditioned_layers
        })
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(D),
            nn.Dropout(dropout_p),
            nn.Linear(D, 64),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, self.num_labels),
        )

        # INITIALIZATION: FiLM as identity
        self._init_film_as_identity()

        # Print config
        if debug:
            print(f"\n[ASTTAFiLM] Configuration:")
            print(f"  Conditioned layers: {self.conditioned_layers}")
            print(f"  Device conditioning: {condition_on_device} (dim={dev_emb_dim})")
            print(f"  Site conditioning: {condition_on_site} (dim={site_emb_dim})")
            print(f"  Continuous conditioning: {condition_on_rest} (dim={rest_encoded_dim})")
            print(f"  Total metadata dim: {meta_dim}")
            print(f"  AST hidden dim: {D}")
            print(f"  FiLM params per layer: 2 × 2 × {D} = {4 * D} (CLS + Patch)")

    def _init_film_as_identity(self):
        """
        Initialize FiLM generators to output near-zero values.
        This ensures γ ≈ 1 and β ≈ 0 at initialization, making TAFiLM
        act as identity and preserving pretrained AST features.
        """
        for l in self.conditioned_layers:
            # CLS generators
            nn.init.zeros_(self.film_generators_cls[str(l)].weight)
            nn.init.zeros_(self.film_generators_cls[str(l)].bias)
            
            # Patch generators
            nn.init.zeros_(self.film_generators_patch[str(l)].weight)
            nn.init.zeros_(self.film_generators_patch[str(l)].bias)
        
        if self.debug:
            print("[ASTTAFiLM] FiLM generators initialized to identity (γ≈1, β≈0)")

    def _build_metadata_embedding(self, device_id, site_id, m_rest):
        """
        Build and encode metadata vector.
        
        Args:
            device_id: (B,) device indices
            site_id: (B,) site indices
            m_rest: (B, rest_dim) normalized continuous metadata
        
        Returns:
            h_m: (B, film_hidden_dim) encoded metadata
        """
        parts = []

        if self.condition_on_device:
            if device_id is None:
                raise ValueError("device_id required when condition_on_device=True")
            parts.append(self.dev_emb(device_id))

        if self.condition_on_site:
            if site_id is None:
                raise ValueError("site_id required when condition_on_site=True")
            parts.append(self.site_emb(site_id))

        if self.condition_on_rest:
            if m_rest is None:
                raise ValueError("m_rest required when condition_on_rest=True")
            if self.rest_encoder is not None:
                parts.append(self.rest_encoder(m_rest))
            else:
                parts.append(m_rest)

        m = torch.cat(parts, dim=-1)  # (B, meta_dim)
        h_m = self.metadata_encoder(m)  # (B, film_hidden_dim)
        
        return h_m

    def _prep_tokens(self, x):
        """
        Build token sequence (same as ASTModel.forward_features).
        
        Args:
            x: (B, 1, F, T) input spectrogram
        
        Returns:
            x: (B, N+2, D) token sequence [CLS, DIST, patch_1, ..., patch_N]
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

    def forward_features(self, x, device_id, site_id, m_rest, return_film_info=False):
        """
        Forward through transformer with Token-Aware FiLM conditioning.
        
        Args:
            x: (B, 1, F, T) spectrogram
            device_id: (B,) device indices
            site_id: (B,) site indices
            m_rest: (B, rest_dim) continuous metadata
            return_film_info: If True, return FiLM parameters for analysis
        
        Returns:
            h_cls: (B, D) class token representation
            film_info: dict (only if return_film_info=True)
        """
        v = self.ast.v
        x = self._prep_tokens(x)  # (B, T, D) where T = N+2

        # Encode metadata
        h_m = self._build_metadata_embedding(device_id, site_id, m_rest)

        # Precompute FiLM parameters for each conditioned layer
        gamma_cls, beta_cls = {}, {}
        gamma_patch, beta_patch = {}, {}

        for l in self.conditioned_layers:
            # CLS/dist token FiLM params
            film_cls = self.film_generators_cls[str(l)](h_m)
            dg_cls, db_cls = film_cls.chunk(2, dim=-1)
            gamma_cls[l] = 1.0 + dg_cls  # Identity-centered
            beta_cls[l] = db_cls

            # Patch token FiLM params
            film_patch = self.film_generators_patch[str(l)](h_m)
            dg_patch, db_patch = film_patch.chunk(2, dim=-1)
            gamma_patch[l] = 1.0 + dg_patch
            beta_patch[l] = db_patch

        # Store for visualization
        film_info = None
        if return_film_info:
            film_info = {
                'gamma_cls': gamma_cls,
                'beta_cls': beta_cls,
                'gamma_patch': gamma_patch,
                'beta_patch': beta_patch,
            }

        # Forward through transformer blocks
        for layer_idx, blk in enumerate(v.blocks):
            # Self-attention
            attn_out = blk.attn(blk.norm1(x))
            x = x + blk.drop_path(attn_out)

            # FFN with Token-Aware FiLM
            normed = blk.norm2(x)

            if layer_idx in self.conditioned_set:
                # Get FiLM params for this layer
                g_cls = gamma_cls[layer_idx].unsqueeze(1)      # (B, 1, D)
                b_cls = beta_cls[layer_idx].unsqueeze(1)       # (B, 1, D)
                g_patch = gamma_patch[layer_idx].unsqueeze(1)  # (B, 1, D)
                b_patch = beta_patch[layer_idx].unsqueeze(1)   # (B, 1, D)

                # Token-aware modulation:
                # - Indices 0, 1: CLS and distillation tokens
                # - Indices 2+: Patch tokens
                normed_cls = g_cls * normed[:, :2, :] + b_cls
                normed_patch = g_patch * normed[:, 2:, :] + b_patch
                normed = torch.cat([normed_cls, normed_patch], dim=1)

            x = x + blk.drop_path(blk.mlp(normed))

        # Final norm and CLS extraction
        x = v.norm(x)
        h_cls = (x[:, 0] + x[:, 1]) / 2.0  # Average CLS and dist tokens

        if return_film_info:
            return h_cls, film_info
        return h_cls

    def forward(self, x, device_id=None, site_id=None, m_rest=None):
        """
        Standard forward pass for training.
        
        Args:
            x: (B, T, F) or (B, 1, F, T) spectrogram
            device_id: (B,) device indices
            site_id: (B,) site indices
            m_rest: (B, rest_dim) continuous metadata
        
        Returns:
            logits: (B, num_labels)
        """
        # Handle input format
        if x.dim() == 3:
            x = x.unsqueeze(1).transpose(2, 3)  # (B, 1, F, T)

        h = self.forward_features(x, device_id, site_id, m_rest)
        logits = self.classifier(h)
        
        return logits

    def forward_with_film_info(self, x, device_id=None, site_id=None, m_rest=None):
        """
        Forward pass that returns FiLM parameters for analysis/visualization.
        
        Returns:
            logits: (B, num_labels)
            film_info: dict with gamma/beta for CLS and patch tokens per layer
        """
        if x.dim() == 3:
            x = x.unsqueeze(1).transpose(2, 3)

        h, film_info = self.forward_features(
            x, device_id, site_id, m_rest, return_film_info=True
        )
        logits = self.classifier(h)
        
        return logits, film_info

    def freeze_backbone(self, until=None, freeze_tafilm=False):
        """
        Freeze AST backbone layers.
        
        Args:
            until: Freeze blocks 0..until (None = all blocks)
            freeze_tafilm: If True, also freeze TAFiLM components
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
            if until is None or i <= until:
                for p in blk.parameters():
                    p.requires_grad = False

        # Optionally freeze TAFiLM components
        if freeze_tafilm:
            if self.dev_emb is not None:
                for p in self.dev_emb.parameters():
                    p.requires_grad = False
            if self.site_emb is not None:
                for p in self.site_emb.parameters():
                    p.requires_grad = False
            if self.rest_encoder is not None:
                for p in self.rest_encoder.parameters():
                    p.requires_grad = False
            for p in self.metadata_encoder.parameters():
                p.requires_grad = False
            for p in self.film_generators_cls.parameters():
                p.requires_grad = False
            for p in self.film_generators_patch.parameters():
                p.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True

    def get_film_stats(self):
        """
        Get statistics about current FiLM parameters for logging.
        
        Returns:
            dict with mean/std of gamma and beta for each layer
        """
        stats = {}
        
        with torch.no_grad():
            for l in self.conditioned_layers:
                # CLS generators
                w_cls = self.film_generators_cls[str(l)].weight
                stats[f'L{l}_cls_weight_norm'] = w_cls.norm().item()
                
                # Patch generators
                w_patch = self.film_generators_patch[str(l)].weight
                stats[f'L{l}_patch_weight_norm'] = w_patch.norm().item()
        
        return stats