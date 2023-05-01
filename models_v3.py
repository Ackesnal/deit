# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models._registry import register_model
from timm.models.layers import trunc_normal_, PatchEmbed, Mlp, DropPath
import math
from typing import Optional


def propagate(x: torch.Tensor, weight: torch.Tensor, index_kept: torch.Tensor, index_prop: torch.Tensor, 
              standard: str = "None", alpha: Optional[float] = 0, token_scales: Optional[torch.Tensor] = None):
    """
    Propagate tokens based on the selection results.
    ================================================
    Args:
        - x: Tensor([B, N, C]): the feature map of N tokens, including the [CLS] token.
        - weight: Tensor([B, N-1, N-1]): the weight of each token propagated to the other tokens, excluding the [CLS] token.
                                         weight could be a pre-defined graph of the current feature map (by default) 
                                         or the attention map (need to manually modify the Block Module).
        - index_kept: Tensor([B, N-1-num_prop]): the index of kept image tokens in the feature map X
        - index_prop: Tensor([B, num_prop]): the index of propagated image tokens in the feature map X
        - standard: str: the method applied to propagate the tokens
        - alpha: float: the coefficient of propagated features
        - token_scales: Tensor([B, N]): the scale of tokens, including the [CLS] token. None by default.
                                        token_scales represents the scales of each token and sum up to N.
    Return:
        - x: Tensor([B, N-1-num_prop, C]): the feature map after propagation
        - weight: Tensor([B, N-1-num_prop, N-1-num_prop]): the graph of feature map after propagation
        - token_scales: Tensor([B, N-1-num_prop]): the scale of tokens after propagation
    """
    
    B, N, C = x.shape
    
    # Step 1: divide tokens
    x_cls = x[:, 0:1] # B, 1, C
    x_kept = x.gather(dim=1, index=index_kept.unsqueeze(-1).expand(-1,-1,C)) # B, N-1-num_prop, C
    x_prop = x.gather(dim=1, index=index_prop.unsqueeze(-1).expand(-1,-1,C)) # B, num_prop, C
    
    # Step 2: divide token_scales if it is not None
    if token_scales is not None:
        token_scales_cls = token_scales[:, 0:1] # B, 1
        token_scales_kept = token_scales.gather(dim=1, index=index_kept) # B, N-1-num_prop
        token_scales_prop = token_scales.gather(dim=1, index=index_prop) # B, num_prop
    
    # Step 3: propagate tokens
    if standard == "None":
        """
        No further propagation
        """
        pass
        
    elif standard == "Mean":
        """
        Calculate the mean of all the propagated tokens,
        and concatenate the result token back to kept tokens.
        """
        if token_scales is not None:
            # weighted average
            x_prop = x_prop * token_scales_prop.unsqueeze(-1) # B, num_prop, C
            x_prop = x_prop.sum(1, keepdim=True) # B, 1, C
            token_scales_prop = token_scales_prop.sum(1, keepdim=True) # B, 1
            x_prop = x_prop / token_scales_prop.unsqueeze(-1) # B, 1, C
            
            # Concatenate the average token and its scale
            x_kept = torch.cat((x_kept, x_prop), dim=1) # B, N-num_prop, C
            token_scales_kept = torch.cat((token_scales_kept, token_scales_prop), dim=1) # B, N-num_prop
        else:
            # naive average
            x_prop = x_prop.mean(1, keepdim=True) # B, 1, C
            # Concatenate the average token 
            x_kept = torch.cat((x_kept, x_prop), dim=1) # B, N-num_prop, C
            
    elif standard == "GraphProp":
        """
        Propagate all the propagated token to kept token
        with respect to the weights and token scales.
        """
        assert weight is not None, "The graph weight is needed for graph propagation"
        
        # Step 3.1: divide propagation weights.
        index_kept = index_kept - 1 # since weights do not include the [CLS] token
        index_prop = index_prop - 1 # since weights do not include the [CLS] token
        
        weight = weight.gather(dim=1, index=index_kept.unsqueeze(-1).expand(-1,-1,N-1)) # B, N-1-num_prop, N-1
        weight_kept = weight.gather(dim=2, index=index_kept.unsqueeze(1).expand(-1,weight.shape[1],-1)) # B, N-1-num_prop, num_prop
        weight_prop = weight.gather(dim=2, index=index_prop.unsqueeze(1).expand(-1,weight.shape[1],-1)) # B, N-1-num_prop, num_prop
        weight = weight.gather(dim=2, index=index_kept.unsqueeze(1).expand(-1,weight.shape[1],-1)) # B, N-1-num_prop, N-1-num_prop
        
        # Step 3.2: normalize the propagation weights to ensure each propagated token is fully broadcast
        weight_prop = weight_prop / (weight_prop.sum(-2, keepdim=True) + 1e-12)
        
        # Step 3.3: scale the tokens if token_scales is not None
        if token_scales is not None:
            x_kept = x_kept * token_scales_kept.unsqueeze(-1) # B, N-1-num_prop, C
            x_prop = x_prop * token_scales_prop.unsqueeze(-1) # B, num_prop, C
            
        # Step 3.4: generate the broadcast message and propagate the message to corresponding kept tokens
        x_prop = weight_prop @ x_prop # B, N-1-num_prop, C
        x_kept = x_kept + alpha * x_prop # B, N-1-num_prop, C
        
        # Step 3.5: re-scale the tokens if token_scales is not None
        if token_scales is not None:
            token_scales_prop = weight_prop @ token_scales_prop.unsqueeze(-1) # B, N-1-num_prop, 1
            token_scales_kept = token_scales_kept + alpha * token_scales_prop.squeeze(-1) # B, N-1-num_prop
            x_kept = x_kept / token_scales_kept.unsqueeze(-1) # B, N-1-num_prop, C
            
    else:
        assert False, "Propagation method \'%f\' has not been supported yet." % standard
    
    # Step 4： concatenate the [CLS] token and generate returned value
    x = torch.cat((x_cls, x_kept), dim=1) # B, N-num_prop, C
    if token_scales is not None:
        token_scales = torch.cat((token_scales_cls, token_scales_kept), dim=1) # B, N-num_prop
    return x, weight, token_scales



def select(weight: torch.Tensor, standard: str = "None", num_prop: int = 0):
    """
    Select image tokens to be propagated. The [CLS] token will be ignored. 
    =============================
    Args:
        - weight: Tensor([B, H, N, N]): only support the attention map of tokens in the current layer
        - standard: str: the method applied to select the tokens
        - num_prop: int: the number of tokens to be propagated
    Return:
        - index_kept: Tensor([B, N-1-num_prop]): the index of kept image tokens in the original feature map X
        - index_prop: Tensor([B, num_prop]): the index of propagated image tokens in the original feature map X
    """
    
    assert len(weight.shape) == 4, "Selection methods on tensors other than the attention map haven't been supported yet."
    B, H, N1, N2 = weight.shape
    assert N1 == N2, "Selection methods on tensors other than the attention map haven't been supported yet."
    N = N1
    assert num_prop >= 0, "The number of propagated/pruned tokens must be non-negative."
            
    if standard == "CLSAttnMean":
        token_rank = weight[:,:,0,1:].mean(1)
        
    elif standard == "CLSAttnMax":
        token_rank = weight[:,:,0,1:].max(1)[0]
            
    elif standard == "IMGAttnMean":
        token_rank = weight[:,:,:,1:].sum(-2).mean(1)
    
    elif standard == "IMGAttnMax":
        token_rank = weight[:,:,:,1:].sum(-2).max(1)[0]
            
    elif standard == "DiagAttnMean":
        token_rank = torch.diagonal(weight, dim1=-2, dim2=-1)[:,:,1:].mean(1)
        
    elif standard == "DiagAttnMax":
        token_rank = torch.diagonal(weight, dim1=-2, dim2=-1)[:,:,1:].max(1)[0]
        
    elif standard == "MixedAttnMean":
        token_rank_1 = torch.diagonal(weight, dim1=-2, dim2=-1)[:,:,1:].mean(1)
        token_rank_2 = weight[:,:,:,1:].sum(-2).mean(1)
        token_rank = token_rank_1 * token_rank_2
        
    elif standard == "MixedAttnMax":
        token_rank_1 = torch.diagonal(weight, dim1=-2, dim2=-1)[:,:,1:].max(1)[0]
        token_rank_2 = weight[:,:,:,1:].sum(-2).max(1)[0]
        token_rank = token_rank_1 * token_rank_2
        
    elif standard == "CosSimMean":
        weight = weight[:,:,1:,:].mean(1)
        weight = weight / weight.norm(dim=-1, keepdim=True)
        token_rank = -(weight @ weight.transpose(-1, -2)).sum(-1)
    
    elif standard == "CosSimMax":
        weight = weight[:,:,1:,:].max(1)[0]
        weight = weight / weight.norm(dim=-1, keepdim=True)
        token_rank = -(weight @ weight.transpose(-1, -2)).sum(-1)
        
    elif standard == "Random":
        token_rank = torch.randn((B, N-1), device=weight.device)
            
    else:
        print("Type\'", standard, "\' selection not supported.")
        assert False
        
    token_rank = torch.argsort(token_rank, dim=1, descending=True) # B, N-1
    index_kept = token_rank[:, :-num_prop]+1 # B, N-1-num_prop
    index_prop = token_rank[:, -num_prop:]+1 # B, num_prop
    return index_kept, index_prop
            
            

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

    def forward(self, x, token_scales=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        if token_scales is not None:
            attn = attn + token_scales.log().reshape(B, 1, 1, N)
            
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        if self.sparsity < 1:
            attn_rank = torch.sort(attn.reshape(B,self.num_heads,-1), dim=-1, descending=True)[0]
            attn_sigma = attn_rank[:,:,int(N*N*self.sparsity)].reshape(B,self.num_heads,1,1).expand(B,self.num_heads,N,N)
            attn = torch.where(attn>=attn_sigma, attn, 0.0)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn



class GraphPropagationBlock(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 init_values=None, selection="None", propagation="None", num_prop=0, sparsity=1,
                 alpha=0, token_scale=False):
                 
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
        
        self.propagation = propagation
        self.selection = selection
        self.num_prop = num_prop
        self.sparsity = sparsity
        self.token_scale = token_scale
        self.alpha = alpha
    
    def forward(self, x, weight, token_scales=None):
        tmp, attn = self.attn(self.norm1(x), token_scales)
        x = x + self.drop_path(self.ls1(tmp))
        
        if self.selection != "None":
            index_kept, index_prop = select(attn, standard=self.selection, num_prop=self.num_prop) # B, N
            x, weight, token_scales = propagate(x, weight, index_kept, index_prop, standard=self.propagation,
                                               alpha=self.alpha, token_scales=token_scales) 
                                                
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x, weight, token_scales
        
        

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
            token_scale=False,
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
        
        self.token_scale = token_scale
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
                selection=selection if i >= prop_start_layer and i < reconstruct_layer else "None",
                propagation=propagation if i >= prop_start_layer and i < reconstruct_layer else "None",
                num_prop=num_prop if i >= prop_start_layer and i < reconstruct_layer else 0,
                sparsity=sparsity if i >= prop_start_layer and i < reconstruct_layer else 1,
                alpha=alpha if i >= prop_start_layer and i < reconstruct_layer else 0,
                token_scale=self.token_scale
            )
            for i in range(depth)])
        
        if propagation == "GraphProp":
            N = (img_size // patch_size)**2
            # Create a range tensor of node indices
            indices = torch.arange(N)
            # Reshape the indices tensor to create a grid of row and column indices
            row_indices = indices.view(-1, 1).expand(-1, N)
            col_indices = indices.view(1, -1).expand(N, -1)
            # Compute the adjacency matrix
            row1, col1 = row_indices // int(math.sqrt(N)), row_indices % int(math.sqrt(N))
            row2, col2 = col_indices // int(math.sqrt(N)), col_indices % int(math.sqrt(N))
            graph = ((abs(row1 - row2) <= 1).float() * (abs(col1 - col2) <= 1).float())
            graph = graph - torch.eye(N)
            self.spatial_graph = graph
        
        if self.token_scale:
            self.token_scales = torch.ones([N+1])
    
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        B, N, C = x.shape
        
        spatial_graph = self.spatial_graph.unsqueeze(0).expand(B,-1,-1).to(x.device)
        
        semantic_graph = x[:, 1:] @ x[:, 1:].transpose(-1, -2)
        semantic_graph_rank = torch.sort(semantic_graph, dim=-1, descending=True)[0]
        semantic_graph_sigma = semantic_graph_rank[:, :, 3].unsqueeze(-1).expand(-1, -1, N-1)
        semantic_graph = torch.where(semantic_graph>=semantic_graph_sigma, 1, 0)
        semantic_graph = semantic_graph - torch.eye(N-1, device=semantic_graph.device).unsqueeze(0)
        
        graph = torch.bitwise_or(semantic_graph.int(), spatial_graph.int()).float()
        
        # normalize
        # degree = graph.sum(-1) # B, N
        # degree = torch.diag_embed(degree**(-1/2))
        # graph = degree @ graph @ degree
        
        token_scales = self.token_scales.unsqueeze(0).expand(B,-1).to(x.device) if self.token_scale else None
        
        for blk in self.blocks:
            x, graph, token_scales = blk(x, graph, token_scales)
        
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