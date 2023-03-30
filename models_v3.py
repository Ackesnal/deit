# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models._registry import register_model
from timm.models.layers import trunc_normal_, PatchEmbed, Mlp, DropPath
import math


class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, sparsity=0.2, reduction_num=0, alpha=0.5):
        original_x = x
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        H = self.num_heads
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1) # B, H, N, N
        
        attn_diag = attn.reshape(B, H, N*N)[:, :, N+1::N+1].mean(1) # B, N-1
        token_rank = torch.argsort(attn_diag, dim=1, descending=True) # B, N-1
        
        num_kept = N - reduction_num
        index_cls = torch.zeros((B, 1), device=token_rank.device, dtype=token_rank.dtype)
        index_kept = torch.cat((index_cls, token_rank[:, :-reduction_num]+1), dim=1) # B, N-K
        index_kept, _ = torch.sort(index_kept) # B, N-K
        index_B = torch.arange(B, dtype=index_kept.dtype, device=index_kept.device).reshape(B, 1).expand(B, num_kept).reshape(-1)*N
        index_kept = index_kept.reshape(B*num_kept) + index_B
        
        weight = attn.transpose(0, 1) # H, B, N, N
        weight = weight.reshape(H, B*N, N) # H, B*N, N
        weight = weight.index_select(dim=1, index=index_kept) # H, B*(N-K), N
        weight = weight.reshape(H, B, num_kept, N) # H, B, (N-K), N
        weight = weight.transpose(0, 1) # B, H, (N-K), N
        
        weight = self.attn_drop(weight)
        x = (weight @ v).transpose(1, 2).reshape(B, num_kept, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        original_x = original_x.reshape(B*N, C).index_select(dim=0, index=index_kept).reshape(B, num_kept, C)
        return x + original_x


class GraphPropagationBlock(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=None,
                 selection="DiagAttn", propagation="None", reduction_num=0, sparsity=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        
        self.propagation = propagation
        self.selection = selection
        self.reduction_num = reduction_num
        self.sparsity = sparsity
    
    def forward(self, x):
        x = self.drop_path(self.ls1(self.attn(self.norm1(x), sparsity=self.sparsity, reduction_num=self.reduction_num, alpha=0.5)))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x 


class GraphPropagationTransformer(VisionTransformer):
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
            block_fn=GraphPropagationBlock,
            selection="None",
            propagation="None",
            reduction_num=0,
            sparsity=1):
        
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
            
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule    
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                selection=selection,
                propagation=propagation,
                reduction_num=reduction_num,
                sparsity=sparsity
            )
            for i in range(depth)])

        
@register_model
def graph_propagation_deit_small_patch16_224(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = GraphPropagationTransformer(patch_size=16, embed_dim=384, depth=12,
                                        num_heads=6, mlp_ratio=4, qkv_bias=True,
                                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
            