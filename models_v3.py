# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models._registry import register_model
from timm.models.layers import trunc_normal_, PatchEmbed, DropPath
import math
from typing import Optional
import timm
from timm.layers.helpers import to_2tuple
from timm.layers import use_fused_attn
import torch.utils.checkpoint as checkpoint
from torch.jit import Final
    

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            num_heads=8,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
            use_conv=False,
            
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        in_features = in_features 
        out_features = out_features 
        hidden_features = hidden_features 
        
        self.fc1 = nn.Conv1d(in_features, hidden_features, 1, groups=num_heads)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, 1, groups=num_heads)
        
    def forward(self, x):
        x = x.transpose(1,2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.transpose(1,2)
        return x



class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Conv1d(dim, dim*3, 1, groups=num_heads)
        self.proj = nn.Conv1d(dim, dim, 1, groups=num_heads)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x.transpose(1,2)).reshape(B, 3, self.num_heads, C//self.num_heads, N).permute(1,0,2,4,3) # B, C*3, N
        q, k, v = torch.chunk(qkv, chunks=3, dim=0) # B, H, N, C/H
        x = F.scaled_dot_product_attention(q, k, v) # B, H, N, C/H
        x = self.proj(x.transpose(2,3).reshape(B, C, N)).transpose(1,2)
        return x
        


class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), 
                             num_heads=num_heads, act_layer=act_layer)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.num_heads = num_heads

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        B, N, C = x.shape
        x = x.reshape(B, N, self.num_heads, C//self.num_heads).transpose(-1,-2).reshape(B, N, C)
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x

        
        

class ShuffleTransformer(VisionTransformer):
    """
    Modifications:
    - Initialize r, token size, and token sources.
    - For MAE: make global average pooling proportional to token size
    """
    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: str = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[Callable] = None,
            act_layer: Optional[Callable] = None,
            block_fn: Callable = Block,
            mlp_layer: Callable = Mlp,
            distillation: bool = False,
            use_checkpoint: bool = False):
        
        super().__init__(
            img_size = img_size,
            patch_size = patch_size,
            in_chans = in_chans,
            num_classes = num_classes,
            global_pool = global_pool,
            embed_dim = embed_dim,
            depth = depth,
            num_heads = num_heads,
            mlp_ratio = mlp_ratio,
            qkv_bias = qkv_bias,
            qk_norm = qk_norm,
            init_values = init_values,
            class_token = class_token,
            no_embed_class = no_embed_class,
            pre_norm = pre_norm,
            fc_norm = fc_norm,
            drop_rate = drop_rate,
            pos_drop_rate = pos_drop_rate,
            patch_drop_rate = patch_drop_rate,
            proj_drop_rate = proj_drop_rate,
            attn_drop_rate = attn_drop_rate,
            drop_path_rate = drop_path_rate,
            weight_init = weight_init,
            embed_layer = embed_layer,
            norm_layer = norm_layer,
            act_layer = act_layer,
            block_fn = block_fn,
            mlp_layer = mlp_layer)
        
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule    
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(depth)])
        
        self.distillation = distillation
        # if self.distillation:
        #    self.distill_head = nn.Linear(num_classes, num_classes)
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
    
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        if self.training and self.distillation:
            return x, x #self.distill_head(x)
        else:
            return x
        

@register_model
def shufformer_tiny_224(pretrained=False, 
                        pretrained_cfg=None, 
                        pretrained_cfg_overlay=None, 
                        **kwargs):
    """
    model = VisionTransformer(patch_size=16, embed_dim=192, depth=12, num_heads=3, 
                              mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    """
    model = ShuffleTransformer(patch_size=16, embed_dim=256, depth=12,
                               num_heads=2, mlp_ratio=4, qkv_bias=True,
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
def shufformer_extra_extra_tiny_224(pretrained=False, 
                                pretrained_cfg=None, 
                                pretrained_cfg_overlay=None, 
                                **kwargs):
    model = ShuffleTransformer(patch_size=16, embed_dim=144, depth=12,
                               num_heads=3, mlp_ratio=4, qkv_bias=True,
                               norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
