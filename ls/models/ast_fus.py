import torch
import torch.nn as nn
from ls.models.ast import ASTModel

class ASTMetaProj(nn.Module):
    """
    Metadata projection fusion:
        m = [E_dev(device_id), E_site(site_id), m_rest]
        m' = W m
        h_tilde = h_CLS + α · m'
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
        init_gate: float = 0.1,
    ):
        super().__init__()

        self.ast = ASTModel(backbone_only=True, **ast_kwargs)
        D = self.ast.original_embedding_dim

        # Categorical encoders
        self.dev_emb = nn.Embedding(num_devices, dev_emb_dim)
        self.site_emb = nn.Embedding(num_sites, site_emb_dim)

        meta_dim = dev_emb_dim + site_emb_dim + rest_dim
        
        self.metadata_proj = nn.Sequential(
            nn.Linear(meta_dim, D),
            nn.LayerNorm(D),
        )
        
        # Learnable gate initialized small
        self.gate = nn.Parameter(torch.tensor(init_gate))

        self.classifier = nn.Sequential(
            nn.LayerNorm(D),
            nn.Dropout(dropout_p),
            nn.Linear(D, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x, device_id, site_id, m_rest):
        h_cls = self.ast.forward_features(x)

        dev = self.dev_emb(device_id)
        site = self.site_emb(site_id)

        m = torch.cat([dev, site, m_rest], dim=-1)
        m_prime = self.metadata_proj(m)

        h_tilde = h_cls + self.gate * m_prime
        
        logits = self.classifier(h_tilde)
        return logits