import torch
import torch.nn as nn
from ls.models.ast import ASTModel
    

class ASTWithNaiveMetadataConcat(nn.Module):
    """
    Early fusion: [h_CLS; m] -> MLP -> 2 logits.

    m is assumed to be a normalized, preprocessed metadata vector
    (e.g. z-scored continuous + one-hot categorical).
    """
    def __init__(
        self,
        ast_kwargs: dict,
        metadata_dim: int,
        hidden_dim: int = 128,
        dropout_p: float = 0.3,
        num_labels: int = 2,
    ):
        super().__init__()
        self.ast = ASTModel(
            backbone_only=True,
            **ast_kwargs
        )
        D = self.ast.original_embedding_dim
        self.metadata_dim = metadata_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(D + metadata_dim),
            nn.Dropout(dropout_p),
            nn.Linear(D + metadata_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, x, m):
        """
        x: (B, T, F) or (B, 1, F, T)
        m: (B, M) metadata vector
        """
        h_cls = self.ast(x)              # (B, D)
        z = torch.cat([h_cls, m], dim=-1)  # (B, D+M)
        logits = self.classifier(z)
        return logits