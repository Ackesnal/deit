# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models._registry import register_model
from timm.models.layers import trunc_normal_, PatchEmbed, DropPath
import math
from typing import Optional
import timm
from timm.layers.helpers import to_2tuple
    

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, norm_layer=None, bias=True, drop=0.):
        
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
        self.hidden_features = hidden_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        
        # Shuffle
        B, H, N, C = x.shape
        # x = x.reshape(B, H, N, H, C//H).permute(0,2,1,4,3).reshape(B,N,C,H).permute(0,3,1,2)
        x = x.reshape(B, H, N, H, C//H).transpose(1,3).reshape(B,H,N,C)
        # x = x.transpose(-1,-2).reshape(B, C, H, N).permute(0, 2, 3, 1)
        
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.scale = qk_scale or dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, H, N, C = x.shape
        
        qkv = self.qkv(x) # B,H,N,3*C
        q, k, v = qkv.split([C, C, C], dim=3) # B,H,N,C
        
        attn = (q @ k.transpose(-2, -1) * self.scale) # B,H,N,N
        attn = attn.softmax(dim=-1)
        
        x = attn @ v # B, H, N, C
        # Shuffle
        # x = x.reshape(B, H, N, H, C//H).permute(0,2,1,4,3).reshape(B,N,C,H).permute(0,3,1,2)
        x = x.reshape(B, H, N, H, C//H).transpose(1,3).reshape(B,H,N,C)
        # x = x.transpose(-1,-2).reshape(B, C, H, N).permute(0, 2, 3, 1)
        return x



class Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 init_values=None):
                 
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        if init_values is not None:
            self.layer_scale = True
            self.ls1 = nn.parameter.Parameter(init_values * torch.ones((dim)))
            self.ls2 = nn.parameter.Parameter(init_values * torch.ones((dim)))
        else:
            self.layer_scale = False
    
    def forward(self, x):
        if self.layer_scale:
            x = x + self.drop_path(self.ls1.view(1,1,-1) * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.ls2.view(1,1,-1) * self.mlp(self.norm2(x)))
        
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
        
        

class ShuffleTransformer(VisionTransformer):
    """
    Modifications:
    - Initialize r, token size, and token sources.
    - For MAE: make global average pooling proportional to token size
    """
    def __init__(self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            global_pool='token',
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            init_values=None,
            class_token=True,
            no_embed_class=False,
            pre_norm=False,
            fc_norm=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            weight_init='',
            embed_layer=PatchEmbed,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            block_fn=Block):
        
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            class_token=class_token,
            no_embed_class=no_embed_class,
            pre_norm=pre_norm,
            fc_norm=fc_norm,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate)
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule    
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=self.head_dim,
                num_heads=self.num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)])
    
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        B, N, C = x.shape
        x = x.reshape(B, N, self.num_heads, C//self.num_heads).transpose(1,2) # B, H, N, C
        
        x = self.blocks(x)
        
        x = x.transpose(1,2).reshape(B, N, C) # B, N, C
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
        

@register_model
def shufformer_tiny_224(pretrained=False, 
                        pretrained_cfg=None, 
                        pretrained_cfg_overlay=None, 
                        **kwargs):
    model = ShuffleTransformer(patch_size=16, embed_dim=288, depth=12,
                               num_heads=3, mlp_ratio=4, qkv_bias=True,
                               norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    
    
@register_model
def shufformer_extra_tiny_224(pretrained=False, 
                              pretrained_cfg=None, 
                              pretrained_cfg_overlay=None, 
                              **kwargs):
    model = ShuffleTransformer(patch_size=16, embed_dim=216, depth=12,
                               num_heads=3, mlp_ratio=4, qkv_bias=True,
                               norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    
    
@register_model
def shufformer_extreme_tiny_224(pretrained=False, 
                                pretrained_cfg=None, 
                                pretrained_cfg_overlay=None, 
                                **kwargs):
    model = ShuffleTransformer(patch_size=16, embed_dim=144, depth=12,
                               num_heads=3, mlp_ratio=4, qkv_bias=True,
                               norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
