# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models._registry import register_model
from timm.models.layers import trunc_normal_, PatchEmbed, Mlp, DropPath
import math


def graph_propagation(x_kept, x_elim, weight, index_kept, index_elim,
                      multihead=True, threshold=True, sparsity=0.05, 
                      alpha=0.1, token_scales=None):
    """
    Perform graph propagation to combine the eliminated tokens into kept tokens
    x_kept -> [B, N-K, C] : The input feature map
    x_elim -> [B, K, C]
    weight -> [B, H, N, N] : The graph edge weights
    index_kept -> [B*(N-K)] : The index of the kept tokens
    index_elim -> [B*K] : The index of the eliminated tokens
    multihead : Whether propagate on multiple heads
    threshold : Whether convert the dense-connected graph into sparse graph
    sparsity : If threshold is True, how sparsity the graph should be
    """
    
    # Step 1: update the token scale
    if token_scales is not None:
        token_scales_cls = token_scales[:, 0:1]
        token_scales = token_scales[:, 1:]
        token_scales_kept = token_scales.gather(dim=1, index=index_kept).unsqueeze(-1) # B, N-K, 1
        token_scales_elim = token_scales.gather(dim=1, index=index_elim).unsqueeze(-1) # B, K, 1
        
        x_elim = x_elim * token_scales_elim
        x_kept = x_kept * token_scales_kept
        
        token_scales_kept = token_scales_kept + weight @ token_scales_elim
        token_scales = torch.cat((token_scales_cls, token_scales_kept.squeeze(-1)), dim=1)
        
    weight = weight.to_sparse()
    # Step 3: propagate tokens
    x_prop = weight @ x_elim # B, N-K, C
    if token_scales is not None:
        x_kept = (x_kept + x_prop) / token_scales_kept # B, (N-K), C
    else:
        x_kept = x_kept + alpha*x_prop # B, (N-K), C
    
    return x_kept, token_scales



def propagate(x, weight, index_kept, index_elim, standard="None", sparsity=0.2, alpha=0.1, token_scales=None):
    B, N, C = x.shape
    
    # index_kept, _ = torch.sort(index_kept) # B, N-K
    # index_elim, _ = torch.sort(index_elim) # B, K
    
    # Step 1: divide tokens
    x_cls = x[:, 0:1] # B, 1, C
    x = x[:, 1:] # B, N-1, C
    x_kept = x.gather(dim=1, index=index_kept.unsqueeze(-1).expand(-1,-1,C)) # B, N-K-1, C
    x_elim = x.gather(dim=1, index=index_elim.unsqueeze(-1).expand(-1,-1,C)) # B, K, C
    
    # Step 2: divide weight
    weight = weight.gather(dim=1, index=index_kept.unsqueeze(2).expand(-1,-1,N-1)) # B, N-K, N
    weight_prop = weight.gather(dim=2, index=index_elim.unsqueeze(1).expand(-1,weight.shape[1],-1)) # B, N-K, K
    weight_kept = weight.gather(dim=2, index=index_kept.unsqueeze(1).expand(-1,weight.shape[1],-1)) # B, N-K, N-K
    
    # Step 3: propagate tokens
    if standard == "None" or sparsity == 0:
        # No further propagation
        pass
        
    elif standard == "Mean":
        # Only add the average
        x_kept = x_kept + alpha * x_elim.mean(1, keepdim=True)
            
    elif standard == "Graph":
        x_kept, token_scales = graph_propagation(x_kept, x_elim, weight_prop, index_kept, index_elim,
                                                 multihead=True, threshold=False, sparsity=sparsity, 
                                                 alpha=alpha, token_scales=token_scales)
        
    elif standard == "ThresholdGraph":
        x_kept, token_scales = graph_propagation(x_kept, x_elim, weight_prop, index_kept, index_elim,
                                                 multihead=True, threshold=True, sparsity=sparsity, 
                                                 alpha=alpha, token_scales=token_scales)
            
    elif standard == "SingleHeadThresholdGraph":
        x_kept, token_scales = graph_propagation(x_kept, x_elim, weight_prop, index_kept, index_elim,
                                                 multihead=False, threshold=True, sparsity=sparsity, 
                                                 alpha=alpha, token_scales=token_scales)
    
    elif standard == "SingleHeadGraph":
        x_kept, token_scales = graph_propagation(x_kept, x_elim, weight_prop, index_kept, index_elim,
                                                 multihead=False, threshold=False, sparsity=sparsity, 
                                                 alpha=alpha, token_scales=token_scales)
    
    else:
        print("Type\'", standard, "\' propagation not supported.")
        assert False
    
    x = torch.cat((x_cls, x_kept), dim=1) # B, N-K, C
    return x, token_scales, weight_kept



def select(weight, standard, num_prop, descending=True):
    """
    standard: "PageRank", "ThresholdPageRank", "CLSAttn" or "Predictor"
    weight: could be attention map (B*H*N*N) or original feature map (B*N*C)
    """
    if len(weight.shape) == 4:
        # attention map
        B, H, N, _ = weight.shape
    else:
        print("Select criterion without attention map hasn't been supported yet.")
        assert False
            
    if standard == "CLSAttn":
        token_rank = weight[:,:,0,1:].mean(1) # B, N-1
        
    elif standard == "CLSAttnMax":
        token_rank = weight[:,:,0,1:].max(1)[0] # B, N-1
            
    elif standard == "IMGAttnMean":
        token_rank = weight[:,:,:,1:].sum(-2).mean(1) # B, N-1
    
    elif standard == "IMGAttnMax":
        token_rank = weight[:,:,:,1:].sum(-2).max(1)[0] # B, N-1
            
    elif standard == "DiagAttnMean":
        # token_rank = weight.reshape(B, H, N*N)[:, :, N+1::N+1].mean(1)
        token_rank = torch.diagonal(weight, dim1=-2, dim2=-1)[:,:,1:].mean(1)
        
    elif standard == "DiagAttnMax":
        # token_rank = weight.reshape(B, H, N*N)[:, :, N+1::N+1].max(1)[0]
        token_rank = torch.diagonal(weight, dim1=-2, dim2=-1)[:,:,1:].max(1)[0]
        
    elif standard == "MixedAttnMax":
        # token_rank = weight.reshape(B, H, N*N)[:, :, N+1::N+1].max(1)[0]
        token_rank_1 = torch.diagonal(weight, dim1=-2, dim2=-1)[:,:,1:].max(1)[0]
        token_rank_2 = weight[:,:,:,1:].sum(-2).max(1)[0] # B, N-1
        token_rank = token_rank_1 * token_rank_2
        
    elif standard == "MixedAttnMean":
        # token_rank = weight.reshape(B, H, N*N)[:, :, N+1::N+1].max(1)[0]
        token_rank_1 = torch.diagonal(weight, dim1=-2, dim2=-1)[:,:,1:].mean(1)
        token_rank_2 = weight[:,:,:,1:].sum(-2).mean(1) # B, N-1
        token_rank = token_rank_1 * token_rank_2
                
    elif standard == "Random":
        token_rank = torch.randn((B, N-1), device=weight.device)
            
    else:
        print("Type\'", standard, "\' selection not supported.")
        assert False
        
    token_rank = torch.argsort(token_rank, dim=1, descending=descending) # B, N-1
    # index_cls = torch.zeros((B, 1), device=token_rank.device, dtype=token_rank.dtype) # B, 1
    # index_kept = torch.cat((index_cls, token_rank[:, :-num_prop]+1), dim=1) # B, N-K
    index_kept = token_rank[:, :-num_prop] # B, N-K
    index_elim = token_rank[:, -num_prop:] # B, K
    return index_kept, index_elim
            
            

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
        
        """
        attn_rank = torch.sort(attn.reshape(B,self.num_heads,-1), dim=-1, descending=True)[0]
        attn_sigma = attn_rank[:,:,int(N*N*0.7)].reshape(B,self.num_heads,1,1).expand(B,self.num_heads,N,N)
        attn = torch.where(attn>=attn_sigma, attn, 0.0)
        """
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn



class GraphPropagationBlock(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 init_values=None, selection="None", propagation="None", num_prop=0, sparsity=1,
                 alpha=0, attention_scale=False):
                 
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
        self.num_prop = num_prop
        self.sparsity = sparsity
        self.attention_scale = attention_scale
        self.alpha = alpha
    
    def forward(self, x, graph, token_scales=None):
        tmp, attn = self.attn(self.norm1(x), token_scales)
        x = x + self.drop_path(self.ls1(tmp))
            
        if self.selection != "None":
            index_kept, index_elim = select(attn, num_prop=self.num_prop, standard=self.selection) # B, N
            x, token_scales, graph = propagate(x, graph, index_kept, index_elim, 
                                        standard=self.propagation, sparsity=self.sparsity,
                                        alpha=self.alpha, token_scales=token_scales)
                                        
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        
        return x, graph, token_scales
        
        

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
                selection=selection if i >= prop_start_layer and i < reconstruct_layer else "None",
                propagation=propagation if i >= prop_start_layer and i < reconstruct_layer else "None",
                num_prop=num_prop if i >= prop_start_layer and i < reconstruct_layer else 0,
                sparsity=sparsity if i >= prop_start_layer and i < reconstruct_layer else 1,
                alpha=alpha if i >= prop_start_layer and i < reconstruct_layer else 0,
                attention_scale=self.attention_scale
            )
            for i in range(depth)])
        
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
        # Normalize the graph
        degree = graph.sum(-1) # B, N
        degree = torch.diag(degree**(-1/2))
        graph = degree @ graph @ degree
        # Batchify the adjacency matrix
        # graph = graph.unsqueeze(0).expand(B, N, N)
        # Put on GPU
        # graph = graph.to(x.device)
        self.graph = graph
        
        if self.attention_scale:
            self.token_scales = torch.ones([N+1])
    
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        B, N, C = x.shape
        
        graph = self.graph.unsqueeze(0).expand(B,-1,-1).to(x.device)
        token_scales = self.token_scales.unsqueeze(0).expand(B,-1).to(x.device) if self.attention_scale else None
        
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