# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models._registry import register_model
from timm.models.layers import trunc_normal_, PatchEmbed, Mlp, DropPath
import math
import timm, tome



class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sparsity=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sparsity = sparsity

    def forward(self, x, x_original, num_prop=0):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
            
        if self.sparsity < 1:
            attn_rank = torch.sort(attn.reshape(B,self.num_heads,-1), dim=-1, descending=True)[0]
            attn_sigma = attn_rank[:,:,int(N*N*self.sparsity)].reshape(B,self.num_heads,1,1).expand(B,self.num_heads,N,N)
            attn = torch.where(attn>=attn_sigma, attn, 0.0)
        
        if num_prop > 0:
            token_rank = torch.diagonal(attn, dim1=-2, dim2=-1)[:,:,1:].max(1)[0]
            token_rank = torch.argsort(token_rank, dim=1, descending=True) # B, N-1
            index_kept = token_rank[:, :-num_prop] + 1 # B, N-K
            index_prop = token_rank[:, -num_prop:] + 1 # B, K
            index_cls = torch.zeros((B, 1), device=index_kept.device, dtype=index_kept.dtype) # B, 1
            index_kept = torch.cat((index_cls, index_kept), dim=1)
            # index_kept = torch.sort(index_kept, dim=1)[0]
            # index_prop = torch.sort(index_prop, dim=1)[0]
            
            attn_all2kept = attn.gather(dim=2, index=index_kept.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, N))
            # attn_all2prop = attn.gather(dim=2, index=index_prop.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, N))
            
            #attn_all2kept_normed = attn_all2kept / attn_all2kept.norm(dim=-1, keepdim=True)
            #attn_all2prop_normed = attn_all2prop / attn_all2prop.norm(dim=-1, keepdim=True)
            #attn_sim = attn_all2prop_normed @ attn_all2kept_normed.transpose(-1,-2)
            #attn_sim = attn_sim.max(1)[0] 
            #attn_sim_index = torch.argmax(attn_sim, dim=-1) # B, num_prop
            
            #attn_all2kept = attn_all2kept.scatter_reduce(dim=2, src=attn_all2prop, index=attn_sim_index.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, N), reduce="mean")
            
            x_original = x_original.gather(dim=1, index=index_kept.unsqueeze(-1).expand(-1, -1, C))
            x = (attn_all2kept @ v).transpose(1, 2).reshape(B, index_kept.shape[1], C)
        else:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, x_original



class GraphPropagationBlock(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 init_values=None, num_prop=0, sparsity=1):
                 
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                              attn_drop=attn_drop, proj_drop=drop, sparsity=sparsity)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        
        self.num_prop = num_prop
        self.sparsity = sparsity
    
    def forward(self, x):
        x, original_x = self.attn(self.norm1(x), x, self.num_prop)
        x = original_x + self.drop_path(self.ls1(x))
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
            num_prop=0,
            sparsity=1,
            alpha=0.1,
            prop_start_layer=0,
            reconstruct_layer=None,
            attention_scale=False,
            pretrained_cfg_overlay=None):
        
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
        
        self.attention_scale = attention_scale
        self.reconstruct_layer = reconstruct_layer
        self.prop_start_layer = prop_start_layer
        self.alpha = alpha
        self.num_heads = num_heads
        self.reconstruct = True if self.reconstruct_layer < depth and self.reconstruct_layer >= 0 else False
        
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
                num_prop=num_prop,
                sparsity=sparsity
            )
            for i in range(depth)])
    
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        
        for blk in self.blocks:
            x = blk(x)
            
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
def graph_propagation_deit_small_patch16_224(pretrained=False, pretrained_cfg=None, **kwargs):
    model = GraphPropagationTransformer(patch_size=16, embed_dim=384, depth=12,
                                        num_heads=6, mlp_ratio=4, qkv_bias=True,
                                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    
@register_model
def token_merge_deit_small_patch16_224(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = timm.create_model("deit_small_patch16_224", pretrained=True)
    tome.patch.timm(model)
    model.r = kwargs["num_prop"]
    return model