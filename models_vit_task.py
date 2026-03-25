from channel import Channel
from functools import partial
import torch
import torch.nn as nn
import timm.models.vision_transformer
from random import choice
from SNRModule import SNRModulation
import time


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, args, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.global_pool = args.global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm
        self.channel = None
        self.snr_modulation = None
        self.noise_std = None
        if args.pass_channel:
            self.channel = Channel(args.channel_type)
            self.snr = args.snr_set
            self.given_snr = args.given_snr
            self.snr_modulation = SNRModulation(
                embed_dim=kwargs['embed_dim'],
                num_layers=args.modulation_layers
            )

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        if self.channel is not None:
            # [B, embed_dim] -> [B, 1, embed_dim]
            outcome = outcome.unsqueeze(1)
            snr = self.given_snr if self.given_snr is not None else choice(self.snr)
            outcome = self.snr_modulation(outcome, snr)
            self.noise_std, outcome = self.channel(outcome, snr)
            # [B, 1, embed_dim] -> [B, embed_dim]
            outcome = outcome.squeeze(1)

        return outcome

    def forward(self, x):
        feature = self.forward_features(x)
        x = self.head(feature)

        return self.noise_std, feature, x


def vit_tiny_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
