# Implementation of AST model for Multi-Label Binary Classification
# Adapted from original for 2-output sigmoid classification

import torch
import torch.nn as nn
import os
from timm.models.layers import to_2tuple, trunc_normal_
import timm

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ASTModelBinaryMultiLabel(nn.Module):
    """
    The AST model for Multi-Label Binary Classification.
    
    Outputs: 2 binary classifiers
    - Output 0: Crackles presence (0 or 1)
    - Output 1: Wheezes presence (0 or 1)
    
    Use with BCEWithLogitsLoss for training.
    """
    def __init__(
            self, 
            label_dim=2,  # Changed: 2 outputs for multi-label binary
            fstride=10, 
            tstride=10, 
            input_fdim=128, 
            input_tdim=1024,
            imagenet_pretrain=True,
            audioset_pretrain=False,
            audioset_ckpt_path='',
            model_size='base384', 
            verbose=True, 
            backbone_only=False,
            dropout_p=0.3,
        ):

        super(ASTModelBinaryMultiLabel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))
            print('Multi-Label Binary: 2 outputs (Crackles, Wheezes)')
        
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        self.label_dim = label_dim  # Changed: 2 outputs for multi-label binary
        self.reg_dropout = nn.Dropout(dropout_p)
        
        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            
            if verbose:
                print('Vision transformer model size {:s} created.'.format(model_size))
            
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]

            self.mlp_head = None

            if not backbone_only:
                # Changed: 2 outputs for multi-label binary
                self.mlp_head = nn.Sequential(
                    nn.LayerNorm(self.original_embedding_dim), 
                    self.reg_dropout,
                    nn.Linear(self.original_embedding_dim, 64),  # Hidden layer
                    nn.ReLU(),
                    self.reg_dropout,
                    nn.Linear(64, self.label_dim)  # 2 outputs (no sigmoid here, apply in loss)
                )

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequency stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            if imagenet_pretrain == True:
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        elif audioset_pretrain == True:
            if audioset_pretrain == True and imagenet_pretrain == False:
                raise ValueError('currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model.')
            if model_size != 'base384':
                raise ValueError('currently only has base384 AudioSet pretrained model.')
            
            if not os.path.exists(audioset_ckpt_path):
                raise FileNotFoundError(f"Pretrained AudioSet model not found at '{audioset_ckpt_path}'.")
            
            if verbose:
                print(f"Loading AudioSet pretrained model from {audioset_ckpt_path}")
            
            # Load checkpoint
            sd = torch.load(audioset_ckpt_path, map_location=DEVICE)
            
            # Create temporary model to load AudioSet weights
            audio_model = ASTModelBinaryMultiLabel(
                label_dim=527, 
                fstride=10, 
                tstride=10, 
                input_fdim=128, 
                input_tdim=1024, 
                imagenet_pretrain=False, 
                audioset_pretrain=False, 
                model_size='base384', 
                verbose=False
            )
            
            # Handle different checkpoint formats (with or without DataParallel)
            model_dict = audio_model.state_dict()
            pretrained_dict = {}
            
            for k, v in sd.items():
                # Remove various prefixes
                if k.startswith('module.'):
                    key = k[7:]  # Remove 'module.' prefix (DataParallel)
                else:
                    key = k
                
                # Keep trained weights only for the vision transformer backbone
                # Skip mlp_head weights since dimensions differ
                if key in model_dict:
                    if model_dict[key].shape == v.shape:
                        pretrained_dict[key] = v
                    elif verbose:
                        print(f"Shape mismatch for {key}: model={model_dict[key].shape}, ckpt={v.shape}")
            
            if len(pretrained_dict) > 0:
                model_dict.update(pretrained_dict)
                audio_model.load_state_dict(model_dict)
                if verbose:
                    print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} tensors from AudioSet checkpoint")
                    print(f"The remaining {len(model_dict) - len(pretrained_dict)} tensors are randomly initialized and will be trained from scratch.")
            else:
                raise RuntimeError("No matching weights found in checkpoint!")
            
            # Extract the vision transformer backbone
            self.v = audio_model.v
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            
            # Create new classification head for multi-label binary
            self.mlp_head = None
            if not backbone_only:
                # Changed: 2 outputs for multi-label binary
                self.mlp_head = nn.Sequential(
                    nn.LayerNorm(self.original_embedding_dim), 
                    self.reg_dropout,
                    nn.Linear(self.original_embedding_dim, 64),  # Hidden layer
                    nn.ReLU(),
                    self.reg_dropout,
                    nn.Linear(64, self.label_dim)  # 2 outputs (no sigmoid here, apply in loss)
                )

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequency stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # Adapt positional embeddings from AudioSet (12x101) to your dimensions
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim/2): 50 - int(t_dim/2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim/2): 6 - int(f_dim/2) + f_dim, :]
            elif f_dim > 12:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def forward_features(self, x):
        """
        Extract features from the backbone.
        :param x: input spectrogram, shape (B, 1, F, T)
        :return: feature embeddings (B, embedding_dim)
        """
        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for blk in self.v.blocks:
            x = blk(x)

        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2  # average CLS and distillation tokens
        return x

    @torch.amp.autocast(device_type=DEVICE.type)
    def forward(self, x):
        """
        Forward pass for multi-label binary classification.
        
        :param x: input spectrogram
                  Can be either (B, T, F) [original format] or (B, 1, F, T) [preprocessed]
        :return: logits (B, 2) - raw outputs for sigmoid/BCEWithLogitsLoss
                 Output 0: Crackles logit
                 Output 1: Wheezes logit
        """
        # Handle both input formats
        if x.dim() == 3:  # (B, T, F) - original format
            x = x.unsqueeze(1)  # (B, 1, T, F)
            x = x.transpose(2, 3)  # (B, 1, F, T)
        
        # Now x is (B, 1, F, T)
        x = self.forward_features(x)

        if self.mlp_head is not None:
            x = self.mlp_head(x)  # (B, 2) logits

        return x  # Return logits, NOT sigmoid
    
    def freeze_backbone(self, until_block=None):
        """
        Freeze transformer blocks for fine-tuning.
        :param until_block: freeze blocks up to and including this index (None = freeze all)
        """
        for p in self.v.patch_embed.parameters():
            p.requires_grad = False
        
        self.v.pos_embed.requires_grad = False
        self.v.cls_token.requires_grad = False
        self.v.dist_token.requires_grad = False
        
        for i, blk in enumerate(self.v.blocks):
            if until_block is None or i <= until_block:
                for p in blk.parameters():
                    p.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True