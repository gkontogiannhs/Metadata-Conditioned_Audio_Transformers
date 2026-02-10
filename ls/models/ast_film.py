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
        
        # Calculate per-feature dimension (ensure it divides evenly)
        per_feature_dim = max(1, hidden_dim // num_features)
        
        # Learnable per-feature transformations
        self.feature_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, per_feature_dim),
                nn.LayerNorm(per_feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p)
            ) for _ in range(num_features)
        ])
        
        # Calculate actual concatenated dimension
        concat_dim = per_feature_dim * num_features
         
        # Combine encoded features (project to target hidden_dim)
        self.combiner = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, num_features) continuous metadata
        Returns:
            encoded: (B, hidden_dim) encoded representation
        """
        # Split features and encode each separately
        features = x.chunk(self.num_features, dim=-1)  # [(B,1), (B,1), (B,1)]
        encoded = [enc(f) for enc, f in zip(self.feature_encoders, features)]
        combined = torch.cat(encoded, dim=-1)  # (B, per_feature_dim * num_features)
        return self.combiner(combined)  # (B, hidden_dim)


class ASTFiLM(nn.Module):
    """
    FiLM-conditioned AST with configurable metadata sources.
    
    Can condition on:
    - Device (stethoscope) embeddings
    - Site (recording location) embeddings  
    - Continuous metadata (age, BMI, duration, etc.)
    - Any combination of the above
    """

    def __init__(
        self,
        ast_kwargs: dict,
        num_devices: int,
        num_sites: int,
        rest_dim: int,
        dev_emb_dim: int = 4,
        site_emb_dim: int = 4,
        conditioned_layers=(10, 11),
        metadata_hidden_dim: int = 64,
        film_hidden_dim: int = 64,
        dropout_p: float = 0.3,
        debug_film: bool = False,
        condition_on_device: bool = True,
        condition_on_site: bool = True,
        condition_on_rest: bool = True,
        use_improved_continuous_encoder: bool = True,
        layer_specific_encoding: bool = False,
    ):
        super().__init__()

        # Backbone
        self.ast = ASTModel(backbone_only=True, **ast_kwargs)
        self.num_labels = ast_kwargs["label_dim"]
        D = self.ast.original_embedding_dim
        self.D = D

        self.debug_film = debug_film
        self.conditioned_layers = sorted(list(conditioned_layers))
        self.conditioned_set = set(self.conditioned_layers)
        self.layer_specific_encoding = layer_specific_encoding
        
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
        
        # Continuous metadata encoder
        if condition_on_rest:
            if use_improved_continuous_encoder:
                # Use improved encoder with per-feature transformations
                self.rest_encoder = ImprovedContinuousEncoder(
                    num_features=rest_dim,
                    hidden_dim=film_hidden_dim // 2,
                    dropout_p=dropout_p * 0.5
                )
                rest_encoded_dim = film_hidden_dim // 2
            else:
                # Simple pass-through (original approach)
                self.rest_encoder = None
                rest_encoded_dim = rest_dim
        else:
            self.rest_encoder = None
            rest_encoded_dim = 0

        # Total metadata dimension
        meta_dim = dev_emb_dim + site_emb_dim + rest_encoded_dim
        
        if debug_film:
            print(f"[FiLM Config] Device: {condition_on_device} ({dev_emb_dim}d), "
                  f"Site: {condition_on_site} ({site_emb_dim}d), "
                  f"Rest: {condition_on_rest} ({rest_encoded_dim}d)")
            print(f"[FiLM Config] Total metadata dim: {meta_dim}")
            print(f"[FiLM Config] Conditioned layers: {self.conditioned_layers}")
            print(f"[FiLM Config] Improved encoder: {use_improved_continuous_encoder}")
            print(f"[FiLM Config] Layer-specific encoding: {layer_specific_encoding}")

        # Metadata encoder h_m = g(m)
        self.metadata_encoder = nn.Sequential(
            nn.LayerNorm(meta_dim),
            nn.Linear(meta_dim, metadata_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(metadata_hidden_dim, film_hidden_dim),
            nn.ReLU(),
        )

        # Optional layer-specific encoders
        if layer_specific_encoding:
            self.layer_encoders = nn.ModuleDict({
                str(l): nn.Linear(film_hidden_dim, film_hidden_dim)
                for l in self.conditioned_layers
            })
        else:
            self.layer_encoders = None

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
            nn.Linear(64, self.num_labels),
        )

    def _prep_tokens(self, x):
        """
        Build token sequence like ASTModel.forward_features().
        
        Args:
            x: (B, 1, F, T) input spectrogram
        
        Returns:
            x: (B, N+2, D) token sequence with CLS and dist tokens
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

    def _build_metadata(self, device_id, site_id, m_rest):
        """
        Build and encode metadata vector.
        
        Args:
            device_id, site_id, m_rest: metadata inputs
        
        Returns:
            h_m: (B, film_hidden_dim) encoded metadata
        """
        metadata_parts = []
        
        if self.condition_on_device:
            if device_id is None:
                raise ValueError("device_id required when condition_on_device=True")
            dev = self.dev_emb(device_id)
            metadata_parts.append(dev)
        
        if self.condition_on_site:
            if site_id is None:
                raise ValueError("site_id required when condition_on_site=True")
            site = self.site_emb(site_id)
            metadata_parts.append(site)
        
        if self.condition_on_rest:
            if m_rest is None:
                raise ValueError("m_rest required when condition_on_rest=True")
            if self.rest_encoder is not None:
                # Use improved encoder
                rest_encoded = self.rest_encoder(m_rest)
                metadata_parts.append(rest_encoded)
            else:
                # Simple pass-through
                metadata_parts.append(m_rest)
        
        m = torch.cat(metadata_parts, dim=-1)
        h_m = self.metadata_encoder(m)
        return h_m

    def forward_features(self, x, device_id=None, site_id=None, m_rest=None, return_film_info=False):
        """
        Forward through transformer with FiLM conditioning.
        
        Args:
            x: (B, 1, F, T) spectrogram
            device_id: (B,) device indices
            site_id: (B,) site indices
            m_rest: (B, rest_dim) continuous metadata
            return_film_info: if True, return dict with FiLM parameters for visualization
        
        Returns:
            h_cls: (B, D) class token features
            film_info: dict (only if return_film_info=True)
        """
        v = self.ast.v
        x = self._prep_tokens(x)

        # Build and encode metadata
        h_m = self._build_metadata(device_id, site_id, m_rest)

        # Precompute FiLM params for each conditioned layer
        gamma = {}
        beta = {}
        for l in self.conditioned_layers:
            # Optional layer-specific encoding
            h_m_l = h_m
            if self.layer_encoders is not None:
                h_m_l = self.layer_encoders[str(l)](h_m)
            
            film = self.film_generators[str(l)](h_m_l)
            dg, db = film.chunk(2, dim=-1)
            gamma[l] = 1.0 + dg  # identity-centered
            beta[l] = db

        # Store for visualization
        film_info = {
            'gamma': gamma,
            'beta': beta,
            'pre_film': {},
            'post_film': {},
            'modulation_magnitude': {}
        } if return_film_info else None

        # Unroll transformer blocks
        for layer_idx, blk in enumerate(v.blocks):
            # Attention block
            attn_out = blk.attn(blk.norm1(x))
            x = x + blk.drop_path(attn_out)

            # FFN block with FiLM conditioning
            normed = blk.norm2(x)
            
            # Apply FiLM after norm2, before MLP
            if layer_idx in self.conditioned_set:
                if return_film_info:
                    film_info['pre_film'][layer_idx] = normed.clone()
                
                g = gamma[layer_idx].unsqueeze(1)  # (B, 1, D)
                b = beta[layer_idx].unsqueeze(1)   # (B, 1, D)
                normed = g * normed + b
                
                if return_film_info:
                    film_info['post_film'][layer_idx] = normed.clone()

            x = x + blk.drop_path(blk.mlp(normed))

        x = v.norm(x)
        h_cls = (x[:, 0] + x[:, 1]) / 2.0
        
        if return_film_info:
            return h_cls, film_info
        return h_cls

    def forward(self, x, device_id=None, site_id=None, m_rest=None):
        """
        Standard forward pass for training.
        
        Args:
            x: (B, T, F) or (B, 1, F, T) spectrogram
            device_id: (B,) device indices (required if condition_on_device=True)
            site_id: (B,) site indices (required if condition_on_site=True)
            m_rest: (B, rest_dim) continuous metadata (required if condition_on_rest=True)
        
        Returns:
            logits: (B, num_labels)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1).transpose(2, 3)  # (B, 1, F, T)

        h = self.forward_features(x, device_id, site_id, m_rest, return_film_info=False)
        logits = self.classifier(h)
        return logits
    
    def forward_with_film_info(self, x, device_id=None, site_id=None, m_rest=None):
        """
        Forward pass that returns both predictions and FiLM information.
        Use this for visualization/analysis.
        
        Args:
            x: (B, T, F) or (B, 1, F, T) spectrogram
            device_id, site_id, m_rest: metadata
        
        Returns:
            logits: (B, num_labels)
            film_info: dict with FiLM parameters and feature modulation info
        """
        if x.dim() == 3:
            x = x.unsqueeze(1).transpose(2, 3)

        h, film_info = self.forward_features(x, device_id, site_id, m_rest, return_film_info=True)
        logits = self.classifier(h)
        
        return logits, film_info
    
    def freeze_backbone(self, until=None, freeze_film=False):
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
            if until is None or i <= until:
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
            if self.rest_encoder is not None:
                for p in self.rest_encoder.parameters():
                    p.requires_grad = False
            for p in self.metadata_encoder.parameters():
                p.requires_grad = False
            for p in self.film_generators.parameters():
                p.requires_grad = False
            if self.layer_encoders is not None:
                for p in self.layer_encoders.parameters():
                    p.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True