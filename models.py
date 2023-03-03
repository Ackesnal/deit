# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg, _create_vision_transformer
import timm

from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import math
from typing import Callable, Tuple
import time
from typing import List, Union

import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer



__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]

class GraphPropagationBlock(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)
    
    def propagation(self, x, weight, index_kept, index_elim):
        B, N, C = x.shape
        
        num_kept = index_kept.shape[1]
        num_elim = index_elim.shape[1]
        
        index_B = torch.arange(B, dtype=index_kept.dtype, device=index_kept.device).reshape(B, 1).expand(B, num_kept).reshape(-1)*N
        index_kept = index_kept.reshape(B*num_kept) + index_B
        index_B = torch.arange(B, dtype=index_elim.dtype, device=index_elim.device).reshape(B, 1).expand(B, num_elim).reshape(-1)*N
        index_elim = index_elim.reshape(B*num_elim) + index_B
        
        # divide tokens
        x_kept = x.reshape(B*N, C).index_select(dim=0, index=index_kept).reshape(B, num_kept, C)
        x_elim = x.reshape(B*N, C).index_select(dim=0, index=index_elim).reshape(B, num_elim, C)
        
        # propagate based on graph
        B, H, N, _ = weight.shape
        
        # Filter weight based on threshold
        weight_mean = weight.mean((-1,-2)).reshape(B,H,1,1).expand(B,H,N,N) # B,H,N,N
        pad = torch.zeros((B,H,N,N), device = weight.device) # B,H,N,N
        weight = torch.where(weight>=weight_mean, weight, pad) # B,H,N,N
        
        # Select weights
        weight = weight.transpose(0,1).reshape(H, B*N, N) # H,B*N,N
        weight_kept = weight.index_select(dim=1, index=index_kept) # H,B*(N-K),N
        weight_kept = weight_kept.reshape(H, B, num_kept, N) # H,B,(N-K),N
        weight_kept = weight_kept.transpose(-1, -2).reshape(H, B*N, num_kept) # H,B*N,(N-K)
        weight_elim2kept = weight_kept.index_select(dim=1, index=index_elim) # H,B*K,(N-K)
        weight_elim2kept = weight_elim2kept.reshape(H, B, num_elim, num_kept).permute(1,0,3,2) # B,H,(N-K),K
            
        # test only
        if False:
            print(torch.count_nonzero(weight_elim2kept, dim=(1,2))/(num_elim*num_kept))
            assert False
        
        prop_matrix = torch.nn.functional.normalize(weight_elim2kept.mean(1), dim=-2, p=1) # B, N-K, K
        
        x_elim = torch.nn.functional.layer_norm(x_elim, normalized_shape=[x_elim.shape[-1]])
        x_prop = weight_elim2kept @ x_elim.reshape(B, num_elim, H, C//H).transpose(1,2) # B, H, N-K, C//H
        x_prop = x_prop.transpose(1,2).reshape(B, num_kept, C) # B,N-K,C
        x_kept = x_kept + x_prop * 0.2 # B,N-K,C
        
        return x_kept, prop_matrix
        
    def forward(self, x: torch.Tensor, num_prop=13) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        
        if self._tome_info["size"] is None:
            x_attn = self.attn(self.norm1(x), attn_size)
            B, N, C = x.shape
            self._tome_info["size"] = torch.ones([B, N], device=x.device, dtype=x.dtype)
        else:
            x_attn, prop_matrix, index_kept, index_elim = self.attn(self.norm1(x), attn_size)
            B, N, C = x.shape
            size_kept = self._tome_info["size"].reshape(-1).index_select(dim=0, index=index_kept).reshape(B, index_kept.shape[0]//B)
            size_elim = self._tome_info["size"].reshape(-1).index_select(dim=0, index=index_elim).reshape(B, index_elim.shape[0]//B, 1)
            self._tome_info["size"] = size_kept + (prop_matrix @ size_elim).squeeze()
            x = x.reshape(-1, C).index_select(dim=0, index=index_kept).reshape(B, -1, C)
        
        x = x + self._drop_path1(x_attn)
        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        
        return x


class ModifiedAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None, num_prop = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        H = self.num_heads
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        if size is not None:
            attn = attn #+ size.log().reshape(B, 1, 1, N) # torch.nn.functional.normalize(size.reshape(B, 1, 1, N), p=1, dim=-1).expand(B,1,N,N) # 
            attn = attn.softmax(dim=-1)
            # select tokens
            token_rank = attn.mean(1).reshape(B, N*N)[:, 0::N+1] # B, N
            token_rank = torch.argsort(token_rank[:, 1:], dim=1, descending=True) # B, N-1
            index_kept, _ = torch.sort(token_rank[:, :-num_prop]) # B, N-K-1
            index_elim, _ = torch.sort(token_rank[:, -num_prop:]) # B, K
            
            index_cls = torch.zeros([index_kept.shape[0], 1], device=index_kept.device, dtype=index_kept.dtype)
            index_kept = torch.cat((index_cls, index_kept + 1), dim=1) # B, N-K
            index_elim = index_elim + 1 # B, K
            
            num_kept = index_kept.shape[1]
            num_elim = index_elim.shape[1]
            
            index_B = torch.arange(B, dtype=index_kept.dtype, device=index_kept.device).reshape(B, 1).expand(B, num_kept).reshape(-1)*N
            index_kept = index_kept.reshape(B*num_kept) + index_B # B*(N-K)
            index_B = torch.arange(B, dtype=index_elim.dtype, device=index_elim.device).reshape(B, 1).expand(B, num_elim).reshape(-1)*N
            index_elim = index_elim.reshape(B*num_elim) + index_B # B*K
            
            # select attention weights
            attn_kept = attn.transpose(0,1).reshape(H, B*N, N) # H,B*N,N
            attn_kept = attn_kept.index_select(dim=1, index=index_kept) # H,B*(N-K),N
            attn_kept = attn_kept.reshape(H, B, num_kept, N) # H,B,(N-K),N
            attn_kept = attn_kept.transpose(0, 1) # B, H, N-K, N 
            
            # filter attention weights
            attn_sorted, _ = torch.sort(attn_kept.reshape(B, self.num_heads, num_kept*N), descending=True, dim=-1) # B, H, N*N
            attn_threshold = attn_sorted[:, :, int(num_kept*N*0.9)] # B, H
            attn_threshold = attn_threshold.reshape(B, self.num_heads, 1, 1).expand(B, self.num_heads, num_kept, N) # B, H, N, N
            pad = torch.zeros((B, self.num_heads, num_kept, N), dtype = attn.dtype, device = attn.device)
            attn_kept = torch.where(attn_kept >= attn_threshold, attn_kept, pad) # B, H, N, N
            
            # update feature
            attn_kept = self.attn_drop(attn_kept)
            x = (attn_kept @ v).transpose(1, 2).reshape(B, num_kept, C)
            
            x = self.proj(x)
            x = self.proj_drop(x)
            
            # update size
            attn_elim2kept = attn_kept.permute(1, 0, 3, 2).reshape(H, B*N, num_kept) # H,B*N,(N-K)
            attn_elim2kept = attn_elim2kept.index_select(dim=1, index=index_elim) # H,B*K,(N-K)
            attn_elim2kept = attn_elim2kept.reshape(H, B, num_elim, num_kept).permute(1,0,3,2) # B,H,(N-K),K
            prop_matrix = torch.nn.functional.normalize(attn_elim2kept.mean(1), dim=-2, p=1) # B, N-K, K
            
            # Return 
            return x, prop_matrix, index_kept, index_elim
        else:
            attn = attn.softmax(dim=-1) # B, H, N, N
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["size"] = None

            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.
    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.
    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = GraphPropagationBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = ModifiedAttention



@register_model
def deit_tiny_patch16_224(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
    

@register_model
def deit_tiny_shuffle_patch16_224(pretrained=False, pretrained_cfg=None, **kwargs):
    model = ShuffleVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model



@register_model
def deit_small_patch16_224(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
    

@register_model
def tome_deit_small_patch16_224(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    apply_patch(model)
    model.r = 14
    
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        ) 
        model.load_state_dict(checkpoint["model"])
    return model
    
@register_model
def tome_deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        ) 
        model.load_state_dict(checkpoint["model"])
    apply_patch(model)
    model.r = 14
    
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs): 
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
    
    
@register_model
def vit_small_patch16_224(pretrained=True, **kwargs):
    """ ViT-Small (ViT-S/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch16_224_in21k', pretrained=pretrained, **dict(model_kwargs, **kwargs))
    print(pretrained)
    return model

