import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SegFormer.mix_transformer import *
from models.SegFormer.segformer_head import SegFormerHead
from models.SegFormer.wrappers import resize

norm_cfg = dict(type='SyncBN', requires_grad=True)


class SegFormerNet(nn.Module):
    def __init__(self, num_classes, shape, **kwargs):
        super(SegFormerNet, self).__init__()
        self.shape = shape
        self.backbone = mit_b1(**kwargs)
        
        self.decoder = SegFormerHead(in_channels=[64, 128, 320, 512],
                        in_index=[0, 1, 2, 3],
                        feature_strides=[4, 8, 16, 32],
                        channels=128,
                        dropout_ratio=0.1,
                        num_classes=num_classes,
                        norm_cfg=norm_cfg,
                        align_corners=False,
                        decoder_params=dict(embed_dim=768))

    def forward(self, x, return_attn = False):
        if return_attn:
            x, attn = self.backbone(x, return_attn=True)
            x = self.decoder(x)
            x = F.interpolate(x, size = self.shape, mode='trilinear')
            return x, attn
        x = self.backbone(x)

        x = self.decoder(x)
        x = F.interpolate(x, size = self.shape, mode='trilinear')
        return x
