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

    def forward(self, x, origin, sparsity=1.0, num_prop=0):
        B, N, C = x.shape
        H = self.num_heads
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale * (1-0.1*math.log(197/N))

        attn = (q @ k.transpose(-2, -1)) # B, H, N, N
        attn = attn.softmax(dim=-1) # B, H, N, N
        
        attn_sort, _ = torch.sort(attn, dim=-1, descending=True)
        attn_threshold = attn_sort[:, :, :, min(N-1, int(N*sparsity))].unsqueeze(-1).expand(B, H, N, N)
        pad = torch.zeros((B, H, N, N), device = attn.device) # B, H, N, N
        attn = torch.where(attn>=attn_threshold, attn, pad)
        
        # attn = attn * torch.bernoulli(torch.ones(attn.shape, device=attn.device) * sparsity) 
        
        """
        attn_sort, _ = torch.sort(attn.reshape(B, H*N*N), dim = -1, descending=True)
        attn_threshold = attn_sort[:, int(H*N*N*sparsity)-1]
        attn_threshold = attn_threshold.reshape(B, 1, 1, 1).expand(B, H, N, N)
        # attn_threshold = torch.ones((B, H, N, N), device = attn.device) / N
        pad = torch.zeros((B, H, N, N), device = attn.device) # B, H, (N-K), K
        attn = torch.where(attn>=attn_threshold, attn, pad)
        """
        
        
        attn_diag = torch.diagonal(attn, dim1=-2, dim2=-1).mean(1) # B, N
        attn_rank = torch.argsort(attn_diag[:, 1:], dim=-1, descending=True) # B, N-1
        
        num_kept = N - num_prop
        
        index_cls = torch.zeros((B, 1), device=attn_rank.device)
        index_kept = torch.cat((index_cls, attn_rank[:, :-num_prop]+1), dim=1) # B, N-K
        index_kept, _ = torch.sort(index_kept, dim=-1) # B, N-K
        index_B = torch.arange(B, device=index_kept.device).reshape(B, 1).expand(B, num_kept).reshape(-1)*N
        index_kept = index_kept.reshape(B*num_kept) + index_B
        
        index_prop = attn_rank[:, -num_prop:]+1 # B, N-K
        index_prop, _ = torch.sort(index_prop, dim=-1) # B, N-K
        index_B = torch.arange(B, device=index_prop.device).reshape(B, 1).expand(B, num_prop).reshape(-1)*N
        index_prop = index_prop.reshape(B*num_prop) + index_B
        
        attn_kept = attn.permute(1,0,2,3).reshape(H, B*N, N).index_select(dim=1, index=index_kept.int())
        attn_kept = attn_kept.reshape(H, B, num_kept, N).permute(1,0,2,3) # B, H, N-K, N
        attn_prop = attn.permute(1,0,2,3).reshape(H, B*N, N).index_select(dim=1, index=index_prop.int())
        attn_prop = attn_prop.reshape(H, B, num_prop, N).permute(1,0,2,3) # B, H, K, N
        
        attn_sim = 1 - torch.cdist(attn_kept, attn_prop)
        
        attn_kept = attn_kept.scatter_reduce_(dim=-2, index=attn_sim.max(-2)[1].unsqueeze(-1).expand(B,H,num_prop,N), src=attn_prop*0.1, reduce="sum")
        """
        attn_sim = attn_prop @ attn_kept.transpose(-1,-2) # B, H, K, N-K
        attn_sim = attn_sim.argmax(dim=-1).unsqueeze(-1).expand(B, H, num_prop, N) # B, H, K, 1
        attn_kept = attn_kept.scatter_reduce_(dim=-2, index=attn_sim, src=attn_prop, reduce="mean")
        """
        
        """
        attn_kept_normed = 1 / attn_kept.norm(p=2, dim=-1, keepdim=True)
        attn_elim_normed = 1 / attn_elim.norm(p=2, dim=-1, keepdim=True)
        attn_denominator = attn_elim_normed @ attn_kept_normed.transpose(2, 3)
        attn_similarity = attn_elim @ attn_kept.transpose(2, 3) * attn_denominator
        attn_similarity = attn_similarity.mean(1)
        attn_similarity, attn_similarity_rank = torch.sort(attn_similarity, dim=-1, descending=True) 
        attn_similarity_rank = attn_similarity_rank[:, :, 0:1].reshape(B, 1, num_elim, 1).expand(B, H, num_elim, N) # B, K
        # attn_kept = attn_kept.scatter_reduce_(dim=2, index=attn_similarity_rank, src=attn_elim, reduce="mean")
        """
        """
        attn_kept_normed = 1 / attn_kept.norm(p=2, dim=-1, keepdim=True) 
        attn_prop_normed = 1 / attn_prop.norm(p=2, dim=-1, keepdim=True)
        attn_denominator = attn_kept_normed @ attn_prop_normed.transpose(2, 3)
        attn_similarity = (attn_kept @ attn_prop.transpose(2, 3)) * attn_denominator # B, H, N-K, K
        attn_threshold = attn_similarity.max(-2)[0].reshape(B, H, 1, num_prop).expand(B, H, num_kept, num_prop)
        pad = torch.zeros((B, H, num_kept, num_prop), device = attn.device) # B, H, (N-K), K
        attn_similarity = torch.where(attn_similarity >= attn_threshold, attn_similarity, pad) # B, H, (N-K), K
        #print(attn_similarity)
        attn_kept = attn_similarity @ attn_prop * 0.05 + attn_kept # B, H, N-K, K
        """
        attn_kept = self.attn_drop(attn_kept)
        x = (attn_kept @ v).transpose(1, 2).reshape(B, num_kept, C)
        """
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        """
        
        x = self.proj(x)
        x = self.proj_drop(x)
        
        origin = origin.reshape(B*N, C).index_select(dim=0, index=index_kept.int()).reshape(B, num_kept, C)
        return x + origin


class GraphPropagationBlock(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=None,
                 selection="None", propagation="None", num_prop=0, sparsity=1):
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
        
        self.num_prop = num_prop
        self.sparsity = sparsity
    
    def forward(self, x):
        x = self.drop_path(self.ls1(self.attn(self.norm1(x), x, sparsity=self.sparsity, num_prop=self.num_prop)))
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
                num_prop=num_prop,
                sparsity=sparsity
            )
            for i in range(depth)])

        
@register_model
def graph_propagation_deit_small_patch16_224(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = GraphPropagationTransformer(patch_size=16, embed_dim=384, depth=12,
                                        num_heads=6, mlp_ratio=4, qkv_bias=True,
                                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def graph_propagation_deit_base_patch16_224(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = GraphPropagationTransformer(patch_size=16, embed_dim=768, depth=12,
                                        num_heads=12, mlp_ratio=4, qkv_bias=True,
                                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    
