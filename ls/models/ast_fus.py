import torch
import torch.nn as nn
from ls.models.ast import ASTModel

class ASTWithMetadataProjection(nn.Module):
    """
    Metadata projection fusion:
        m = [E_dev(device_id), E_site(site_id), m_rest]
        m' = W m
        h_tilde = h_CLS + m'
    """
    def __init__(
        self,
        ast_kwargs: dict,
        num_devices: int,
        num_sites: int,
        dev_emb_dim: int = 4,
        site_emb_dim: int = 4,
        rest_dim: int = 5,
        hidden_dim: int = 64,
        dropout_p: float = 0.3,
        num_labels: int = 2,
    ):
        super().__init__()

        self.ast = ASTModel(backbone_only=True, **ast_kwargs)
        D = self.ast.original_embedding_dim

        # categorical encoders
        self.dev_emb  = nn.Embedding(num_devices, dev_emb_dim)
        self.site_emb = nn.Embedding(num_sites, site_emb_dim)

        meta_dim = dev_emb_dim + site_emb_dim + rest_dim
        self.metadata_proj = nn.Linear(meta_dim, D)

        self.classifier = nn.Sequential(
            nn.LayerNorm(D),
            nn.Dropout(dropout_p),
            nn.Linear(D, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, x, device_id, site_id, m_rest):
        """
        x:          (B, 1, F, T)
        device_id:  (B,) long
        site_id:    (B,) long
        m_rest:     (B, rest_dim) float
        """
        h_cls = self.ast(x)                    # (B, D)

        dev  = self.dev_emb(device_id)         # (B, d_dev)
        site = self.site_emb(site_id)          # (B, d_site)

        m = torch.cat([dev, site, m_rest], dim=-1)  # (B, meta_dim)
        m_prime = self.metadata_proj(m)              # (B, D)

        h_tilde = h_cls + m_prime
        logits = self.classifier(h_tilde)
        return logits