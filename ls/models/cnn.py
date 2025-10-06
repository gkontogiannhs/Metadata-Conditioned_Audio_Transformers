import torch
import torch.nn as nn
import torch.nn.functional as F


def init_layer(layer: nn.Module):
    """Initialize a Linear or Convolutional layer using Xavier uniform initialization."""
    if hasattr(layer, 'weight') and layer.weight is not None:
        nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn: nn.BatchNorm2d):
    """Initialize a BatchNorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock5x5(nn.Module):
    """
    A convolutional block with a 5x5 kernel, followed by BatchNorm, ReLU activation,
    and optional pooling.
    """
    def __init__(self, in_channels: int, out_channels: int, stride=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(5, 5),
            stride=stride,
            padding=(2, 2),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, x, pool_size=(2, 2), pool_type='avg'):
        """Forward pass with configurable pooling."""
        x = F.relu_(self.bn1(self.conv1(x)))

        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x = F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type is None:
            pass  # no pooling
        else:
            raise ValueError(f"Invalid pool_type: {pool_type}. Choose from ['avg', 'max', 'avg+max', None].")

        return x


class CNN6(nn.Module):
    """
    CNN6 architecture used in AudioSet pretraining.
    Reference: Kong et al., PANNs: Large-Scale Pretrained Audio Neural Networks (2020).

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for log-mel spectrograms).
    num_classes : int or None
        If None, the model acts as a feature extractor (returns embeddings).
        If set, a classification head is added.
    do_dropout : bool
        Whether to apply dropout between convolutional blocks.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = None, do_dropout: bool = False):
        super().__init__()
        self.final_feat_dim = 512
        self.do_dropout = do_dropout
        self.is_backbone = num_classes is None  # True → feature extractor mode

        # Convolutional backbone
        self.conv_block1 = ConvBlock5x5(in_channels, 64)
        self.conv_block2 = ConvBlock5x5(64, 128)
        self.conv_block3 = ConvBlock5x5(128, 256)
        self.conv_block4 = ConvBlock5x5(256, 512)

        self.dropout = nn.Dropout(0.2)

        # Optional classifier head
        if not self.is_backbone:
            self.classifier = nn.Linear(self.final_feat_dim, num_classes)
            init_layer(self.classifier)
        else:
            self.classifier = None

    def load_sl_official_weights(self, path: str = 'pretrained_models/Cnn6_mAP=0.343.pth'):
        """
        Load pretrained CNN6 weights (AudioSet version).
        Available at: https://zenodo.org/record/3960586
        """
        checkpoint = torch.load(path, map_location='cpu')
        weights = checkpoint.get('model', checkpoint)
        state_dict = {k: v for k, v in weights.items() if k in self.state_dict()}
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights. Missing: {missing}, Unexpected: {unexpected}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN6 backbone (and classifier if specified)."""
        for conv_block in [self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4]:
            x = conv_block(x, pool_size=(2, 2), pool_type='avg')
            if self.do_dropout:
                x = self.dropout(x)

        # Global pooling: mean over time, then combine mean+max over frequency
        x = torch.mean(x, dim=3)          # mean over time dimension
        x_max, _ = torch.max(x, dim=2)    # max over frequency
        x_mean = torch.mean(x, dim=2)     # mean over frequency
        x = x_max + x_mean                # combine both statistics

        if self.is_backbone:
            return x  # return embeddings for downstream tasks
        else:
            return self.classifier(x)  # return logits