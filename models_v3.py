# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, PatchEmbed, Mlp, DropPath
import math


def graph_propagation(x_kept, x_elim, weight, index_kept, index_elim,
                      multihead=True, threshold=True, sparsity=0.2, alpha=0.1):
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
    
    B, num_kept, C = x_kept.shape
    num_elim = x_elim.shape[1]
    B, H, N, _ = weight.shape
    
    # Step 1: select weights that propagate from eliminated tokens to kept tokens.
    weight = weight.transpose(0, 1) # H, B, N, N
    weight = weight.reshape(H, B*N, N) # H, B*N, N
    weight = weight.index_select(dim=1, index=index_kept) # H, B*(N-K), N
    weight = weight.reshape(H, B, num_kept, N) # H, B, (N-K), N
    weight = weight.transpose(2, 3) # H, B, N, (N-K)
    weight = weight.reshape(H, B*N, num_kept) # H, B*N, (N-K)
    weight = weight.index_select(dim=1, index=index_elim) # H, B*K, (N-K)
    weight = weight.reshape(H, B, num_elim, num_kept) # H, B, K, (N-K)
    weight = weight.transpose(2, 3) # H, B, (N-K), K
    
    # Step 2: filter out insignificant edges, depending on the sparsity
    if threshold:
        weight_rank, _ = torch.sort(weight.reshape(H, B, -1), dim=-1, descending=True) # H, B, (N-K)*K
        weight_threshold = weight_rank[:, :, int(num_elim * num_kept * sparsity)] # H, B, 1
        weight_threshold = weight_threshold.reshape(H, B, 1, 1).expand(H, B, num_kept, num_elim) # H, B, (N-K), K
        pad = torch.zeros((H, B, num_kept, num_elim), device = weight.device) # H, B, (N-K), K
        weight = torch.where(weight>=weight_threshold, weight, pad) # H, B, (N-K), K
        """ 
        # test only
        print(torch.count_nonzero(weight, dim=(1,2))/(num_elim*num_kept))
        assert False
        """
    
    # Step 3: propagate tokens
    if multihead:
        x_prop = weight @ x_elim.reshape(B, num_elim, H, C//H).permute(2, 0, 1, 3) # H, B, (N-K), C//H
        x_prop = x_prop.permute(1, 2, 0, 3).reshape(B, num_kept, C) # B, (N-K), C
        x_kept = x_kept + alpha * x_prop # B, (N-K), C
        """
        # sparse matrixm multiplication
        weight = weight.reshape(H*B, num_kept, num_elim)
        weight = weight.to_sparse()
        x_elim = x_elim.reshape(B, num_elim, H, C//H).permute(2, 0, 1, 3).reshape(H*B, num_elim, C//H)
        x_prop = torch.bmm(weight, x_elim) # H, B, (N-K), C//H
        x_prop = x_prop.reshape(H, B, num_kept, C//H).permute(1, 2, 0, 3).reshape(B, num_kept, C) # B, (N-K), C
        x_kept = x_kept + alpha * x_prop # B, (N-K), C
        """
    else:
        weight = weight.mean(0) # B, (N-K), K
        x_prop = weight @ x_elim # B, N-K, C
        x_kept = x_kept + alpha * x_prop # B, (N-K), C
    
    return x_kept


def propagate(x, weight, index_kept, index_elim, standard=None, alpha=1):
    B, N, C = x.shape
    num_kept = index_kept.shape[1]
    num_elim = index_elim.shape[1]
        
    index_kept, _ = torch.sort(index_kept) # B, N-K
    index_elim, _ = torch.sort(index_elim) # B, K
        
    index_B = torch.arange(B, dtype=index_kept.dtype, device=index_kept.device).reshape(B, 1).expand(B, num_kept).reshape(-1)*N
    index_kept = index_kept.reshape(B*num_kept) + index_B
    index_B = torch.arange(B, dtype=index_elim.dtype, device=index_elim.device).reshape(B, 1).expand(B, num_elim).reshape(-1)*N
    index_elim = index_elim.reshape(B*num_elim) + index_B
    
    # divide tokens
    x_kept = x.reshape(B*N, C).index_select(dim=0, index=index_kept).reshape(B, num_kept, C)
    x_elim = x.reshape(B*N, C).index_select(dim=0, index=index_elim).reshape(B, num_elim, C)
    
    if standard is None or standard == "none" or standard == "None":
        # No further propagation
        pass
        
    elif standard == "Mean":
        # Only add the average
        x_kept = alpha * x_kept + (1-alpha) * x_elim.mean(1, keepdim=True)
            
    elif standard == "Graph":
        x_kept = graph_propagation(x_kept, x_elim, weight, index_kept, index_elim,
                                   multihead=True, threshold=False, alpha=alpha)
        
    elif standard == "ThresholdGraph":
        x_kept = graph_propagation(x_kept, x_elim, weight, index_kept, index_elim,
                                   multihead=True, threshold=True, alpha=alpha)
            
    elif standard == "SingleHeadThresholdGraph":
       x_kept = graph_propagation(x_kept, x_elim, weight, index_kept, index_elim,
                                  multihead=False, threshold=True, alpha=alpha)
    
    elif standard == "SingleHeadGraph":
       x_kept = graph_propagation(x_kept, x_elim, weight, index_kept, index_elim,
                                  multihead=False, threshold=False, alpha=alpha)
    
    else:
        print("Type\'", standard, "\' propagation not supported.")
        assert False
            
    return x_kept


def select(weight, standard, descending=True):
    """
    standard: "PageRank", "ThresholdPageRank", "CLSAttn" or "Predictor"
    weight: could be attention map (B*H*N*N) or original feature map (B*N*C)
    """
    if len(weight.shape) == 3:
        # feature map
        B, N, C = weight.shape
    elif len(weight.shape) == 4:
        # attention map
        B, H, N, _ = weight.shape
    
    if standard == "PageRank":
        token_rank = pagerank(weight) # B, N-1
            
    elif standard == "ThresholdPageRank":
        token_rank = pagerank(weight, threshold=0.3) # B, N-1
            
    elif standard == "CLSAttn":
        token_rank = weight.mean(1)[:,0,1:] # B, N-1
            
    elif standard == "IMGAttn":
        token_rank = weight[:,:,1:,1:].mean(1).sum(-2) # B, N-1
            
    elif standard == "DiagAttn":
        token_rank = weight.mean(1).reshape(B, N*N)[:, 0::N+1][:,1:]
            
    elif standard == "Predictor":
        print("Haven't implemented")
        assert False
            
    elif standard == "Random":
        token_rank = torch.randn((B, N-1), device=weight.device)
            
    else:
        print("Type\'", standard, "\' selection not supported.")
        assert False
        
    token_rank = torch.argsort(token_rank, dim=1, descending=descending) # B, N-1
    return token_rank # B, N-1


def pagerank(weight, max_iter = 20, d = 0.95, min_dist = 1e-3, threshold = False):
    assert weight.shape[-1] == weight.shape[-2] # ensure weight is an N*N matrix
    B = weight.shape[0]
    N = weight.shape[-1]
        
    # aggregate multi-heads and detach
    if weight.shape[1] != N:
        new_weight = weight.mean(1).clone().detach() # B, N, N
    else:
        new_weight = weight.clone().detach() # B, N, N
            
    # deal with threshold
    if type(threshold) == bool and not threshold:
        pass
            
    elif type(threshold) == bool and threshold:
        # filter out values less than the mean by default
        new_weight_mean = new_weight.mean((1,2)) # B
        new_weight_mean = new_weight_mean.reshape(B,1,1).expand(B,N,N)
        pad = torch.zeros((B,N,N), dtype = new_weight.dtype, device = new_weight.device)
        new_weight = torch.where(new_weight >= new_weight_mean, new_weight, pad)
        
    elif type(threshold) == float:
        # filter out values less than the percentage
        new_weight_sorted, _ = torch.sort(new_weight.reshape(B,-1), dim=1, descending=True) # B, N*N
        new_weight_threshold = new_weight_sorted[:, int(N*N*threshold)] # B
        new_weight_threshold = new_weight_threshold.reshape(B,1,1).expand(B,N,N) # B,N,N
        pad = torch.zeros((B,N,N), dtype = new_weight.dtype, device = new_weight.device)
        new_weight = torch.where(new_weight >= new_weight_threshold, new_weight, pad)
        
        """
        # test only
        print(torch.count_nonzero(new_weight, dim=(1,2))/(N*N))
        assert False
        """
        
    # PageRank
    pagerank = torch.ones((B, N-1, 1), device=new_weight.device) / (N-1) # B, N-1, 1
    trans_matrix = new_weight[:,1:,1:].transpose(-1, -2) # transition matrix: B, N-1, N-1
    trans_matrix = trans_matrix / trans_matrix.sum(-2, keepdim=True) # B, N-1, N-1
    """
    # PageRank
    pagerank = torch.ones((B, N, 1), device=new_weight.device) / N # B, N-1, 1
    trans_matrix = new_weight.transpose(-1, -2) # transition matrix: B, N-1, N-1
    # trans_matrix = trans_matrix / trans_matrix.sum(-2, keepdim=True) # B, N-1, N-1
    """
        
    for i in range(max_iter):
        new_pagerank = d * trans_matrix @ pagerank + (1-d) / (N-1) # page rank update with dumping
        dist = torch.linalg.norm((new_pagerank-pagerank).squeeze())
        pagerank = new_pagerank
        if dist < min_dist:
            break
                
    return pagerank.squeeze() # B, N-1
        

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

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class GraphPropagationBlock(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=None,
                 selection="DiagAttn", propagation="None", reduction_num=0):
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
    
    def forward(self, x):
        tmp, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(self.ls1(tmp))
        
        if self.selection != "None" and self.reduction_num > 0:
            # select tokens and propagate
            token_rank = select(attn, standard=self.selection)
            index_cls = torch.zeros((x.shape[0], 1), device=token_rank.device, dtype=token_rank.dtype)
            index_kept = torch.cat((index_cls, token_rank[:, :-self.reduction_num]+1), dim=1) # B, N-K
            index_elim = token_rank[:, -self.reduction_num:]+1 # B, K
            x = propagate(x, attn, index_kept, index_elim, standard=self.propagation, alpha=0.5)
        
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
            reduction_num=0):
        
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
                reduction_num=reduction_num
            )
            for i in range(depth)])
    
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        x_cls = x[:,0:1]
        x_img = x[:,1:]
        B, N, C = x_img.shape
        x_img = x_img.reshape(B, int(math.sqrt(N)), int(math.sqrt(N)), C)
        x_part1 = x_img[:, 0::2, 0::2].reshape(B, -1, C) # B, N/4, C
        x_part2 = x_img[:, 0::2, 1::2].reshape(B, -1, C) # B, N/4, C
        x_part3 = x_img[:, 1::2, 0::2].reshape(B, -1, C) # B, N/4, C
        x_part4 = x_img[:, 1::2, 1::2].reshape(B, -1, C) # B, N/4, C
        
        x_1 = torch.cat((x_cls, x_part1), dim=1)
        x_2 = torch.cat((x_cls, x_part2), dim=1)
        x_3 = torch.cat((x_cls, x_part3), dim=1)
        x_4 = torch.cat((x_cls, x_part4), dim=1)
        
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x_1 = checkpoint_seq(self.blocks, x_1)
        else:
            x_1 = self.blocks(x_1)
            
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x_2 = checkpoint_seq(self.blocks, x_2)
        else:
            x_2 = self.blocks(x_2)
        
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x_3 = checkpoint_seq(self.blocks, x_3)
        else:
            x_3 = self.blocks(x_3)
            
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x_4 = checkpoint_seq(self.blocks, x_4)
        else:
            x_4 = self.blocks(x_4)
            
        x_1 = self.norm(x_1)
        x_2 = self.norm(x_2)
        x_3 = self.norm(x_3)
        x_4 = self.norm(x_4)
        
        return x_1, x_2, x_3, x_4
        
        
    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x #if pre_logits else self.head(x)

    def forward(self, x):
        x_1, x_2, x_3, x_4 = self.forward_features(x)
        x_1 = self.forward_head(x_1)
        x_2 = self.forward_head(x_2)
        x_3 = self.forward_head(x_3)
        x_4 = self.forward_head(x_4)
        x = self.head(x_1) + self.head(x_2) + self.head(x_3) + self.head(x_4)
        return x
        
@register_model
def graph_propagation_deit_small_patch16_224(pretrained=False, pretrained_cfg=None, **kwargs):
    model = GraphPropagationTransformer(patch_size=16, embed_dim=384, depth=12,
                                        num_heads=6, mlp_ratio=4, qkv_bias=True,
                                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
            