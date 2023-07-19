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
from timm.models.vision_transformer import init_weights_vit_timm
import math
from typing import Optional
import timm
from timm.layers.helpers import to_2tuple
from timm.layers import use_fused_attn
import torch.utils.checkpoint as checkpoint
from torch.jit import Final
    

class Downsampling(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)
        self.norm = norm_layer(4*dim)

    def forward(self, x):
        B, N, C = x.shape
        
        x = x.reshape(B, int(N**0.5), int(N**0.5), C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4*C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        
        return x



class Attention(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_mid,
            dim_out,
            num_heads,
            num_reps=8,
            attn_drop=0.,
            proj_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
    ):
        super().__init__()
        assert dim_in % num_heads == 0, 'dim should be divisible by num_heads'
        assert dim_mid % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.dim_mid = dim_mid
        self.scale = dim_in ** (-0.5)
        
        self.pre_norm = norm_layer(dim_in)
        
        self.qk = nn.Conv1d(in_channels=dim_in, 
                            out_channels=dim_in*2, 
                            kernel_size=1,
                            groups=num_heads)
                            
        self.v = nn.Conv1d(in_channels=dim_in, 
                           out_channels=dim_mid, 
                           kernel_size=1,
                           groups=num_heads)
                           
        self.rep_k = torch.nn.parameter.Parameter(torch.empty((1, num_heads, dim_in//num_heads, num_reps)))
        self.rep_q = torch.nn.parameter.Parameter(torch.empty((1, num_heads, num_reps, dim_in//num_heads)))
        
        self.post_norm = norm_layer(dim_mid)
        self.proj_1 = nn.Linear(dim_mid, dim_out)
        self.proj_2 = nn.Linear(dim_out, dim_out)
        self.act = act_layer()
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        # shortcut
        B, N, C = x.shape
        shortcut = x
        
        # pre-layer normalization
        x = self.pre_norm(x)
        
        # calculate query, key
        qk = self.qk(x.transpose(1, 2)).reshape(B, 2, self.num_heads, C//self.num_heads, N).transpose(-1, -2)
        q, k = qk[:, 0], qk[:, 1] # B, num_heads, N, C//num_heads
        
        # calculate value
        v = self.v(x.transpose(1, 2)).reshape(B, self.num_heads, self.dim_mid//self.num_heads, N).transpose(1, 2).reshape(B, self.num_heads, self.dim_mid//self.num_heads, N).transpose(-1, -2) # B, H, N, C*exp
        
        # calculate attention
        #x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop) # nB, num_heads, nN, nC*exp//num_heads
        q = q @ self.rep_k * self.scale # B, H, N, K
        k = self.rep_q @ k.transpose(-1,-2) * self.scale # B, H, K, N
        attn = q @ k # B, H, N, N
        attn = attn.softmax(-1) # B, H, N, N
        
        # calculate output
        x = attn @ v # B, H, N, C*exp
        
        # transpose back
        x = x.transpose(1, 2).reshape(B, N, self.dim_mid)
        
        # output linear
        x = self.proj_2(self.act(self.proj_1(self.post_norm(x))))
        x = shortcut + self.drop_path(x)
        return x
        


class Block(nn.Module):
    def __init__(
            self,
            dim,
            dim_next,
            num_rep,
            num_heads,
            num_layers,
            expansion=4.,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            init_values=None,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        
        self.downsampling = Downsampling(dim = dim_pre)
        
        self.attns = nn.Sequential([Attention(dim_in = dim,
                                              dim_mid = dim*expansion,
                                              dim_out = dim,
                                              num_rep = num_rep,
                                              num_heads = num_heads,
                                              attn_drop = attn_drop,
                                              proj_drop = proj_drop,
                                              drop_path = drop_path,
                                              norm_layer = norm_layer,
                                              act_layer = act_layer)
                                    for i in range(num_layers)])
        
    def forward(self, x):
        x = self.downsampling(x)
        x = self.attns(x)
        return x

        
        

class ShuffleTransformer(nn.Module):
    """
    Modifications:
    - Initialize r, token size, and token sources.
    - For MAE: make global average pooling proportional to token size
    """
    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 4,
            in_chans: int = 3,
            num_classes: int = 1000,
            num_reps: int = 8,
            global_pool: str = 'avg',
            embed_dims: int = [48, 96, 192, 384],
            num_layers: int = [2,2,6,2],
            num_heads: int = [1, 2, 3, 4],
            expansions: float = [3, 3, 3, 3],
            init_value: Optional[float] = None,
            fc_norm: Optional[bool] = None,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: str = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[Callable] = None,
            act_layer: Optional[Callable] = None,
            block_fn: Callable = Block,
            distillation: bool = False,
            use_checkpoint: bool = False):
        
        super().__init__()
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.use_checkpoint = use_checkpoint
        
        # Patch embedding
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dims[i],
            bias=True
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches, embed_dims[i]) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        
        # Stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule    
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dims[i],
                dim_next=embed_dims[i+1] if i < len(depth)-1 else None,
                num_rep=num_reps[i],
                num_heads=num_heads[i],
                num_layers=num_layers[i],
                expansion=expansions[i],
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                init_value=init_value,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
            )
            for i in range(depth)])
            
        self.norm = norm_layer(embed_dims[-1]) if not use_fc_norm else nn.Identity()
        
        # Classifier Head
        self.fc_norm = norm_layer(embed_dims[-1]) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)
            
    
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
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
    """
    model = VisionTransformer(patch_size=16, embed_dim=192, depth=12, num_heads=3, 
                              mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    """
    model = ShuffleTransformer(patch_size=16, embed_dim=192, depth=[2,2,6,2],
                               num_heads=3, mlp_ratio=3, qkv_bias=True,
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
