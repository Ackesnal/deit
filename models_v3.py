# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from einops import rearrange, reduce

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models._registry import register_model
from timm.layers import trunc_normal_, PatchEmbed, DropPath
from timm.layers.helpers import to_2tuple
import math
import torch.autograd.profiler as profiler
import torch.utils.checkpoint as ckpt


def make_print_grad(parameter):
    def print_grad(grad):
        print(f"Gradient of ({parameter}):", grad.max(), grad.min())
    return print_grad

    

def stadardization(W, dim_in, num_head, dim_head):
    num_dim = len(W.shape)
    if num_dim == 0 or W.requires_grad == False:
        return W
    if num_dim == 1:
        #if (W.abs()<=1e-9).all():
        #    return W
        #print(W.max(), W.min())
        W = W.reshape(num_head, dim_head)
    elif num_dim == 2:
        # Linear weights
        W = W.reshape(dim_in, num_head, dim_head)
    elif num_dim == 4:
        # Conv weights
        k = W.shape[-1] # kernel size
        W = W.reshape(dim_in, -1)
    
    #W.register_hook(make_print_grad("W before standardization")) 
        
    mean = W.mean(dim=-1, keepdim=True)
    std = W.std(dim=-1, keepdim=True)
    scale = torch.maximum(std*(dim_head**0.5), torch.ones(std.shape, device=std.device)*1e-2)
    W = (W - mean) / scale
    
    #W.register_hook(make_print_grad("W after standardization"))
        
    if num_dim == 1:
        W = W.reshape(num_head * dim_head)
    elif num_dim == 2:
        W = W.reshape(dim_in, num_head * dim_head).T
    elif num_dim == 4:
        W = W.reshape(dim_in, 1, k, k)
        
    assert not W.isnan().any() # prevent nan value after standardization
    #print(W.var(-1).mean(), W.mean(), W.max(), W.min())
    return W


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            dim_in,
            dim_hidden=None,
            dim_out=None,
            num_head=8,
            act_layer=nn.Sigmoid,
            bias=True,
            use_conv=False,
            drop=0.,
            drop_path=0.,
            layer=0
    ):
        super().__init__()
        # Hyperparameters
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden or dim_in
        self.dim_out = dim_out or dim_in
        self.num_head = num_head
        
        ############################ ↓↓↓ 2-layer MLP ↓↓↓ ###########################
        # Take place
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.fc1 = linear_layer(self.dim_in, self.dim_hidden, bias=bias)
        self.fc1.weight.requires_grad_(False)
        self.fc1.bias.requires_grad_(False)
        self.fc2 = linear_layer(self.dim_hidden, self.dim_out, bias=bias)
        self.fc2.weight.requires_grad_(False)
        self.fc2.bias.requires_grad_(False)
        
        # Trainable parameters
        self.fc1_weight = nn.Parameter(torch.rand((self.dim_in, self.dim_hidden)))
        self.fc2_weight = nn.Parameter(torch.rand((self.dim_hidden, self.dim_out)))
        if bias:
            self.fc1_bias = nn.Parameter(torch.zeros((self.dim_hidden)))
            self.fc2_bias = nn.Parameter(torch.zeros((self.dim_out)))
        else:
            self.fc1_bias = torch.zeros((self.dim_hidden))
            self.fc2_bias = torch.zeros((self.dim_out))
        
        # Activation, Sigmoid by default
        self.act = act_layer()
        ############################ ↑↑↑ 2-layer MLP ↑↑↑ ###########################
        
        ########################## ↓↓↓ Shortcut scale ↓↓↓ ##########################
        #self.shortcut_gain1 = nn.Parameter(torch.ones((1)) * 100)
        #self.shortcut_gain2 = nn.Parameter(torch.ones((1)) * 100)
        #self.shortcut_gain3 = nn.Parameter(torch.ones((1)) * 100)
        self.expansion_ratio = nn.Parameter(torch.ones((1)) * 8)
        
        #self.gamma_fc1 = nn.Parameter(torch.ones(1, self.dim_in))
        #self.gamma_fc2 = nn.Parameter(torch.ones(1, self.dim_hidden))
        
        self.std_1 = nn.Parameter(torch.ones((1)), requires_grad=False)
        self.std_1_accumulation = nn.Parameter(torch.zeros((1)), requires_grad=False)
        self.std_2 = nn.Parameter(torch.ones((1)), requires_grad=False)
        self.std_2_accumulation = nn.Parameter(torch.zeros((1)), requires_grad=False)
        ########################## ↑↑↑ Shortcut scale ↑↑↑ ##########################
        
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        if True:
            B, N, C = x.shape
            ###################### ↓↓↓ Standardization ↓↓↓ ######################
            """
            fc1_weight = stadardization(self.fc1_weight, 
                                        dim_in = self.dim_in, 
                                        num_head = self.num_head, 
                                        dim_head = self.dim_hidden//self.num_head) * self.gamma_fc1
                                        
            fc2_weight = stadardization(self.fc2_weight, 
                                        dim_in = self.dim_hidden, 
                                        num_head = self.num_head,
                                        dim_head = self.dim_out//self.num_head) * self.gamma_fc2
            """
            ###################### ↑↑↑ Standardization ↑↑↑ ######################
            
            ######################## ↓↓↓ 2-layer MLP ↓↓↓ ########################
            # FFN in
            #self.std_1_accumulation.data = self.std_1_accumulation + x.std(-1).mean()
            shortcut = x #.repeat(1, 1, self.dim_hidden//C) # B, N, 4C
            x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True)
            x = torch.nn.functional.linear(x, self.fc1_weight.T, self.fc1_bias) # B, N, 4C
            #x = self.drop_path(x) * nn.functional.sigmoid(self.shortcut_gain1) * 0.5 + shortcut # B, N, 4C
            
            # Activation
            #shortcut = x # B, N, 4C
            x = (self.act(x) - 0.5) * self.expansion_ratio # B, N, 4C
            #x = self.drop_path(x) * nn.functional.sigmoid(self.shortcut_gain2) * 0.5 + shortcut # B, N, 4C
                
            # FFN out
            #self.std_2_accumulation.data = self.std_2_accumulation + x.std(-1).mean()
            #shortcut = x.reshape(B, N, -1, C).mean(2)
            x = torch.nn.functional.linear(x, self.fc2_weight.T, self.fc2_bias)
            #x = self.drop_path(x) * nn.functional.sigmoid(self.shortcut_gain3) * 0.5 + shortcut
            x = self.drop_path(x) + shortcut
            ######################## ↑↑↑ 2-layer MLP ↑↑↑ ########################
        return x
        
    def adaptive_gamma(self, steps):
        self.std_1.data = (self.std_1_accumulation/steps) * self.std_1
        self.std_2.data = (self.std_2_accumulation/steps) * self.std_2
        self.std_1_accumulation.data = self.std_1_accumulation.data * 0
        self.std_2_accumulation.data = self.std_2_accumulation.data * 0
        
        
class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_head=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., drop_path=0., layer=0):
        super().__init__()
        
        #################### ↓↓↓ Self Attention ↓↓↓ ###################
        # Hyperparameters
        self.num_head = num_head
        self.dim_head = dim // num_head
        self.dim_in = dim
        self.scale = qk_scale or self.dim_head ** -0.5 # scale
        
        # Attention drop
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Take place
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv.weight.requires_grad_(False)
        self.qkv.bias.requires_grad_(False)
        
        # Trainable parameters
        self.q_weight = nn.Parameter(torch.rand((dim, dim)))
        self.k_weight = nn.Parameter(torch.rand((dim, dim)))
        self.v_weight = nn.Parameter(torch.rand((dim, dim)))
        
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros((dim))) 
            self.k_bias = nn.Parameter(torch.zeros((dim)))
            self.v_bias = nn.Parameter(torch.zeros((dim)))
        else:
            self.q_bias = torch.zeros((dim))
            self.k_bias = torch.zeros((dim))
            self.v_bias = torch.zeros((dim))
        #################### ↑↑↑ Self Attention ↑↑↑ ###################
        
        #################### ↓↓↓ Output Linear ↓↓↓ ####################
        # Take place
        self.proj = nn.Linear(dim, dim)
        self.proj.weight.requires_grad_(False)
        self.proj.bias.requires_grad_(False)
        
        # Trainable parameters
        self.proj_weight = nn.Parameter(torch.rand((dim, dim)))
        self.proj_bias = nn.Parameter(torch.rand((dim)))
        
        # proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        #################### ↑↑↑ Output Linear ↑↑↑ ####################
        
        #################### ↓↓↓ Shortcut scale ↓↓↓ ###################
        #self.shortcut_gain1 = nn.Parameter(torch.ones((1)) * 100)
        #self.shortcut_gain2 = nn.Parameter(torch.ones((1)) * 100)
        #self.shortcut_gain3 = nn.Parameter(torch.ones((1)) * 100)
        
        #self.gamma_q = nn.Parameter(torch.ones(1, dim))
        #self.gamma_k = nn.Parameter(torch.ones(1, dim))
        #self.gamma_v = nn.Parameter(torch.ones(1, dim))
        #self.gamma_f = nn.Parameter(torch.ones(1, dim))
        
        self.std_1 = nn.Parameter(torch.ones((1)), requires_grad=False)
        self.std_1_accumulation = nn.Parameter(torch.zeros((1)), requires_grad=False)
        self.std_2 = nn.Parameter(torch.ones((1)), requires_grad=False)
        self.std_2_accumulation = nn.Parameter(torch.zeros((1)), requires_grad=False)
        #################### ↑↑↑ Shortcut scale ↑↑↑ ####################
        
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        if True: #self.training:
            B, N, C = x.shape
            ####################### ↓↓↓ Standardization ↓↓↓ #######################
            """
            q_weight = stadardization(self.q_weight, 
                                      dim_in=self.dim_in, 
                                      num_head=self.num_head, 
                                      dim_head=self.dim_head) * self.gamma_q
                                      
            k_weight = stadardization(self.k_weight, 
                                      dim_in=self.dim_in, 
                                      num_head=self.num_head, 
                                      dim_head=self.dim_head) * self.gamma_k
                                      
            v_weight = stadardization(self.v_weight, 
                                      dim_in=self.dim_in, 
                                      num_head=self.num_head, 
                                      dim_head=self.dim_head) * self.gamma_v
                                      
            proj_weight = stadardization(self.proj_weight, 
                                         dim_in=self.dim_in,
                                         num_head=self.num_head,
                                         dim_head=self.dim_head) * self.gamma_f
            """
            ####################### ↑↑↑ Standardization ↑↑↑ #######################
            
            ####################### ↓↓↓ Self-attention ↓↓↓ ########################
            # Shortcut and update std_1
            shortcut = x
            #self.std_1_accumulation.data = self.std_1_accumulation + x.std(-1).mean()
                
            x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True)
            # Calculate Query (Q), Key (K) and Value (V)
            q = torch.nn.functional.linear(x, self.q_weight, self.q_bias) # B, N, C
            k = torch.nn.functional.linear(x, self.k_weight, self.k_bias) # B, N, C
            v = torch.nn.functional.linear(x, self.v_weight, self.v_bias) # B, N, C
                
            # Add shortcut and droppath to V
            # v = self.drop_path(v) * nn.functional.sigmoid(self.shortcut_gain1) * 0.5 + shortcut # B, N, C
            #v = self.drop_path(v) + shortcut # B, N, C
            
            # Reshape Query (Q), Key (K) and Value (V)
            q = rearrange(q, 'b n (nh hc) -> b nh n hc', nh=self.num_head) # B, nh, N, C//nh
            k = rearrange(k, 'b n (nh hc) -> b nh hc n', nh=self.num_head) # B, nh, C//nh, N
            v = rearrange(v, 'b n (nh hc) -> b nh n hc', nh=self.num_head) # B, nh, N, C//nh
                
            # Shortcut
            #shortcut = v # B, nh, N, C//nh 
            
            # Calculate attention map
            attn = q @ (k * self.scale) # B, nh, N, N
            attn = attn.softmax(dim=-1) # B, nh, N, N
            
            # Calculate attended x
            x = attn @ v # B, nh, N, C//nh
            
            # Add shortcut and droppath to attended x
            #x = self.drop_path(x) * nn.functional.sigmoid(self.shortcut_gain2) * 0.5 + shortcut # B, nh, N, C//nh
            #x = self.drop_path(x) + shortcut # B, nh, N, C//nh
            
            # Reshape x back to input shape
            x = rearrange(x, 'b nh n hc -> b n (nh hc)', nh=self.num_head) # B, nh, N, C//nh
            ####################### ↑↑↑ Self-attention ↑↑↑ ########################
            
            ###################### ↓↓↓ Linear projection ↓↓↓ ######################
            # Shortcut
            #shortcut = x # B, N, C
            #self.std_2_accumulation.data = self.std_2_accumulation.data + x.std(-1).mean()
            
            # Linear projection
            x = torch.nn.functional.linear(x, self.proj_weight, self.proj_bias) # B, N, C 
            
            # Add shortcut and droppath to output
            #x = self.drop_path(x) * nn.functional.sigmoid(self.shortcut_gain3) * 0.5 + shortcut # B, N, C
            x = self.drop_path(x) + shortcut # B, N, C
            ###################### ↑↑↑ Linear projection ↑↑↑ ######################
            
            #if x.get_device() == 0:
                #print("x:", x.std(-1).mean().item(), x.mean().item(), x.max().item(), x.min().item())
                #print("mhsa gammas:", self.gamma_input1.data, self.gamma_input2.data, self.gamma_input3.data)
                #print("q_weight:", q_weight.var(-1).mean(), q_weight.mean(), q_weight.max(), q_weight.min())
                #print("V:", v_bias)
                #print("V:", self.v_weight.grad.mean(), self.v_weight.grad.max(), self.v_weight.grad.min())
                #print("V weight:", self.v_weight.grad.mean(), self.v_weight.grad.max(), self.v_weight.grad.min())
                #print("K weight:", self.k_weight.grad.mean(), self.k_weight.grad.max(), self.k_weight.grad.min())
                #print("Q weight:", self.q_weight.grad.mean(), self.q_weight.grad.max(), self.q_weight.grad.min())
                #print("F weight:", self.proj_weight.grad.mean(), self.proj_weight.grad.max(), self.proj_weight.grad.min())
            return x
    
    def adaptive_gamma(self, steps):
        self.std_1.data = (self.std_1_accumulation/steps) * self.std_1
        self.std_2.data = (self.std_2_accumulation/steps) * self.std_2
        self.std_1_accumulation.data = self.std_1_accumulation.data * 0
        self.std_2_accumulation.data = self.std_2_accumulation.data * 0
        
        

class NFAttentionBlock(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_head, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.Sigmoid, layer=0): 
        super().__init__()
        self.attn = Attention(dim, num_head=num_head, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                              attn_drop=attn_drop, proj_drop=drop, drop_path=drop_path, layer=layer)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim_in=dim, dim_hidden=mlp_hidden_dim, num_head=num_head, bias=qkv_bias,
                       act_layer=act_layer, drop=drop, drop_path=drop_path, layer=layer)
    
    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x
        
    def adaptive_gamma(self, steps):
        self.attn.adaptive_gamma(steps)
        self.mlp.adaptive_gamma(steps)



class NFTransformer(VisionTransformer):
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
            act_layer=nn.Sigmoid,
            block_fn=NFAttentionBlock,
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
            
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule   
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_head=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                act_layer=act_layer,
                layer=i
            )
            for i in range(depth)])
        
        self.num_head = num_heads
        self.dim_head = embed_dim//self.num_head
        self.norm_pre = norm_layer(self.dim_head, elementwise_affine=False) if pre_norm else nn.Identity()
        self.norm = None
        
        self.std_head = nn.Parameter(torch.ones((1)), requires_grad=False)
        self.std_head_accumulation = nn.Parameter(torch.zeros((1)), requires_grad=False)
        #self.gamma_head = nn.Parameter(torch.ones(1, embed_dim))
        self.head_weight = nn.Parameter(torch.rand((embed_dim, num_classes)))
        self.head_bias = nn.Parameter(torch.zeros((num_classes)))
        
        self.head.weight.requires_grad_(False)
        self.head.bias.requires_grad_(False)
        
        self._init_standard_weights()
            
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        
        B, N, C = x.shape
        x = x.reshape(B, N, self.num_head, self.dim_head)
        x = self.norm_pre(x)
        x = x.reshape(B, N, C)
        
        for i, blk in enumerate(self.blocks):
            if self.training:
                #x.register_hook(make_print_grad("x of layer "+str(i)))
                #print(i)
                x = blk(x)#ckpt.checkpoint(blk, x)
            else:
                x = blk(x)
            
        return x
    
    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        
        if pre_logits:
            return x
        else:
            """
            head_weight = stadardization(self.head_weight, 
                                         dim_in=self.embed_dim, 
                                         num_head=1, 
                                         dim_head=1000) * self.gamma_head
            """
            #self.std_head_accumulation.data = self.std_head_accumulation + x.std(-1, correction=0).mean()
            x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True)
            x = torch.nn.functional.linear(x, self.head_weight.T, self.head_bias) # B, N, C
            return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
        
    def _init_standard_weights(self):
        for name, param in self.named_parameters():
            if "_weight" in name:
                nn.init.trunc_normal_(param, mean=0.0, std=.02, a=-2, b=2)
            elif "_bias" in name:
                nn.init.zeros_(param)
                
    def adaptive_gamma(self, steps):
        return
        for blk in self.blocks:
            prev = blk.adaptive_gamma(steps)
            
        self.std_head.data = (self.std_head_accumulation/steps) * self.std_head
        self.std_head_accumulation.data = self.std_head_accumulation.data * 0
            
            
        
@register_model
def normalization_free_deit_small_patch16_224_layer12(pretrained=False, pretrained_cfg=None, **kwargs):
    model = NFTransformer(patch_size=16, embed_dim=192, depth=12, pre_norm=True,
                          num_heads=3, mlp_ratio=4, qkv_bias=True, fc_norm=False,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    
    
    
@register_model
def normalization_free_deit_small_patch16_224_layer12(pretrained=False, pretrained_cfg=None, **kwargs):
    model = NFTransformer(patch_size=16, embed_dim=384, depth=12, pre_norm=True,
                          num_heads=6, mlp_ratio=4, qkv_bias=True, fc_norm=False,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model