import torch
import torch.nn as nn
from ls.models.ast import ASTModel

class ASTWithMetadataProjection(nn.Module):
    """
    Metadata projection fusion:
        m -> m' in R^D
        h_tilde = h_CLS + m'
        h_tilde -> MLP -> logits
    """
    def __init__(
        self,
        ast_kwargs: dict,
        metadata_dim: int,
        hidden_dim: int = 64,
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

        # Linear projection m -> m' in R^D
        self.metadata_proj = nn.Linear(metadata_dim, D)

        # Same style classifier as baseline
        self.classifier = nn.Sequential(
            nn.LayerNorm(D),
            nn.Dropout(dropout_p),
            nn.Linear(D, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, x, m):
        """
        x: (B, T, F) or (B, 1, F, T)
        m: (B, M)
        """
        h_cls = self.ast(x)               # (B, D)
        m_prime = self.metadata_proj(m)   # (B, D)
        h_tilde = h_cls + m_prime         # (B, D)  -- Eq. (11) of paper
        logits = self.classifier(h_tilde)
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

# astproj = ASTWithMetadataProjection(ast_kwargs, metadata_dim=10, hidden_dim=64, dropout_p=0.3, num_labels=2).to(DEVICE)
# print(astproj)
# out = astproj(dummy_input, metadata)  # (2, 2)
# print(out.shape)  # torch.Size([2, 2])