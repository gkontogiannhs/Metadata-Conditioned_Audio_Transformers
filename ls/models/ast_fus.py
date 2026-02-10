import torch
import torch.nn as nn
from ls.models.ast import ASTModel

class ASTMetaProj(nn.Module):
    """
    AST with metadata projection fusion.
    
    Supports ablation via use_device, use_site, use_continuous flags.
    
    Fusion: h_tilde = h_CLS + gate * proj(metadata)
    """
    def __init__(
        self,
        ast_kwargs: dict,
        num_devices: int = 4,
        num_sites: int = 7,
        dev_emb_dim: int = 8,
        site_emb_dim: int = 8,
        rest_dim: int = 3,  # age, bmi, duration
        hidden_dim: int = 64,
        dropout_p: float = 0.3,
        num_labels: int = 2,
        init_gate: float = 0.5,
        # Ablation flags
        use_device: bool = True,
        use_site: bool = True,
        use_continuous: bool = True,
        use_missing_flags: bool = False,
    ):
        super().__init__()
        
        # Store ablation flags
        self.use_device = use_device
        self.use_site = use_site
        self.use_continuous = use_continuous
        self.use_missing_flags = use_missing_flags
        
        # Build AST backbone
        from ls.models.ast import ASTModel
        self.ast = ASTModel(backbone_only=True, **ast_kwargs)
        D = self.ast.original_embedding_dim  # 768 for base
        
        # Categorical embeddings (only if used)
        self.dev_emb = nn.Embedding(num_devices, dev_emb_dim) if use_device else None
        self.site_emb = nn.Embedding(num_sites, site_emb_dim) if use_site else None
        
        # Continuous feature normalization
        actual_rest_dim = rest_dim + (2 if use_missing_flags else 0)  # +2 for missing flags
        self.rest_norm = nn.LayerNorm(actual_rest_dim) if use_continuous else None
        
        # Compute total metadata dimension
        meta_dim = 0
        if use_device:
            meta_dim += dev_emb_dim
        if use_site:
            meta_dim += site_emb_dim
        if use_continuous:
            meta_dim += actual_rest_dim
        
        self.meta_dim = meta_dim
        self.use_metadata = meta_dim > 0
        
        # Metadata projection (only if any metadata is used)
        if self.use_metadata:
            self.metadata_proj = nn.Sequential(
                nn.Linear(meta_dim, D),
                nn.LayerNorm(D),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(D, D),
                nn.LayerNorm(D),
            )
            # Learnable gate
            self.gate = nn.Parameter(torch.tensor(init_gate))
        else:
            self.metadata_proj = None
            self.gate = None
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(D),
            nn.Dropout(dropout_p),
            nn.Linear(D, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, num_labels),
        )
        
        # Print configuration
        print(f"\n[ASTMetaProj] Configuration:")
        print(f"  use_device: {use_device} (dim={dev_emb_dim if use_device else 0})")
        print(f"  use_site: {use_site} (dim={site_emb_dim if use_site else 0})")
        print(f"  use_continuous: {use_continuous} (dim={actual_rest_dim if use_continuous else 0})")
        print(f"  use_missing_flags: {use_missing_flags}")
        print(f"  Total metadata dim: {meta_dim}")
        print(f"  AST hidden dim: {D}")
        print(f"  Init gate: {init_gate}")

    def forward(self, x, device_id=None, site_id=None, m_rest=None, 
                age_missing=None, bmi_missing=None):
        """
        Args:
            x: (B, 1, F, T) spectrogram
            device_id: (B,) device indices
            site_id: (B,) site indices
            m_rest: (B, 3) normalized [age, bmi, duration]
            age_missing: (B,) missing flags (optional)
            bmi_missing: (B,) missing flags (optional)
        """
        # Extract AST features
        h_cls = self.ast.forward_features(x)  # (B, D)
        
        if not self.use_metadata:
            # No metadata - just classify
            return self.classifier(h_cls)
        
        # Build metadata vector
        meta_parts = []
        
        if self.use_device and device_id is not None:
            dev = self.dev_emb(device_id)  # (B, dev_emb_dim)
            meta_parts.append(dev)
        
        if self.use_site and site_id is not None:
            site = self.site_emb(site_id)  # (B, site_emb_dim)
            meta_parts.append(site)
        
        if self.use_continuous and m_rest is not None:
            if self.use_missing_flags and age_missing is not None and bmi_missing is not None:
                # Concatenate missing flags
                m_rest_full = torch.cat([
                    m_rest, 
                    age_missing.unsqueeze(-1), 
                    bmi_missing.unsqueeze(-1)
                ], dim=-1)
            else:
                m_rest_full = m_rest
            
            m_rest_normed = self.rest_norm(m_rest_full)
            meta_parts.append(m_rest_normed)
        
        # Concatenate all metadata
        m = torch.cat(meta_parts, dim=-1)  # (B, meta_dim)
        
        # Project to AST dimension
        m_prime = self.metadata_proj(m)  # (B, D)
        
        # Gated fusion
        h_tilde = h_cls + self.gate * m_prime
        
        # Classify
        logits = self.classifier(h_tilde)
        
        return logits

    def freeze_backbone(self, until_block=None):
        """Freeze AST backbone layers."""
        self.ast.freeze_backbone(until_block=until_block)
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True

    def get_gate_value(self):
        """Return current gate value for logging."""
        return self.gate.item() if self.gate is not None else 0.0