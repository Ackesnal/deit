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
import torch.utils.checkpoint as checkpoint


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
    std = W.std(dim=-1, keepdim=True, correction=0)
    scale = std*((dim_head)**0.5) + 1e-9
    W = (W - mean) / scale
    
    #W.register_hook(make_print_grad("W after standardization"))
        
    if num_dim == 1:
        W = W.reshape(num_head * dim_head)
    elif num_dim == 2:
        W = W.reshape(dim_in, num_head * dim_head).T
    elif num_dim == 4:
        W = W.reshape(dim_in, 1, k, k)
        
    assert not W.isnan().any() # prevent nan value after standardization
        
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
            self.fc1_bias = nn.Parameter(torch.rand((self.dim_hidden)))
            self.fc2_bias = nn.Parameter(torch.rand((self.dim_out)))
        else:
            self.fc1_bias = torch.zeros((self.dim_hidden))
            self.fc2_bias = torch.zeros((self.dim_out))
        
        # Activation, Sigmoid by default
        self.act = act_layer()
        ############################ ↑↑↑ 2-layer MLP ↑↑↑ ###########################
        
        ########################## ↓↓↓ Shortcut scale ↓↓↓ ##########################
        self.gamma_input1 = nn.Parameter(torch.ones((1)) * 50, requires_grad=False)
        self.gamma_input2 = nn.Parameter(torch.ones((1)) * 50, requires_grad=False)
        self.gamma_input3 = nn.Parameter(torch.ones((1)) * 50, requires_grad=False)
        self.gamma_input1_accumulation = 0
        self.gamma_input2_accumulation = 0
        self.gamma_input3_accumulation = 0
        
        self.gamma1 = 1 # nn.Parameter(torch.zeros(1))
        self.gamma1_accumulation = 0
        self.gamma2 = 1 # nn.Parameter(torch.zeros(1))
        self.gamma2_accumulation = 0
        self.gamma3 = 1 # nn.Parameter(torch.zeros(1))
        self.gamma3_accumulation = 0
        ########################## ↑↑↑ Shortcut scale ↑↑↑ ##########################
        
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        if True:
            ###################### ↓↓↓ Standardization ↓↓↓ ######################
            fc1_weight = stadardization(self.fc1_weight, 
                                        dim_in = self.dim_in, 
                                        num_head = self.num_head, 
                                        dim_head = self.dim_hidden//self.num_head)
            fc2_weight = stadardization(self.fc2_weight, 
                                        dim_in = self.dim_hidden, 
                                        num_head = self.num_head,
                                        dim_head = self.dim_out//self.num_head)
            fc1_bias = stadardization(self.fc1_bias, 
                                      dim_in = self.dim_in, 
                                      num_head = self.num_head, 
                                      dim_head = self.dim_hidden//self.num_head)
            fc2_bias = stadardization(self.fc2_bias, 
                                      dim_in = self.dim_hidden, 
                                      num_head = self.num_head, 
                                      dim_head = self.dim_out//self.num_head)
            ###################### ↑↑↑ Standardization ↑↑↑ ######################
            
            B, N, C = x.shape
            ######################## ↓↓↓ 2-layer MLP ↓↓↓ ########################
            # FFN in
            
            shortcut = x.repeat(1,1,4) # B, N, 4C
            self.gamma_input1_accumulation += x.std(-1, correction=0).mean().item()
            x = torch.nn.functional.linear(x / self.gamma_input1, fc1_weight, fc1_bias)
            x = self.drop_path(x) * self.gamma1 + shortcut
            
            # Activation
            shortcut = x
            self.gamma_input2_accumulation += x.std(-1, correction=0).mean().item()
            x = self.act(x / self.gamma_input2) - 0.5
            x = self.drop_path(x) * self.gamma2 + shortcut
            
            # FFN out
            shortcut = x[:,:,:C]
            self.gamma_input3_accumulation += x.std(-1, correction=0).mean().item()
            x = torch.nn.functional.linear(x / self.gamma_input3, fc2_weight, fc2_bias)
            x = self.drop_path(x) * self.gamma3 + shortcut
            
            ######################## ↑↑↑ 2-layer MLP ↑↑↑ ########################
        else:
            x = self.fc1(x)
            x = self.act(x)
            x = self.fc2(x)
        return x
        
    def adaptive_gamma(self, steps):
        self.gamma_input1 = nn.Parameter(self.gamma_input1*0.8+(self.gamma_input1_accumulation/steps)*0.2, requires_grad=False)
        self.gamma_input2 = nn.Parameter(self.gamma_input2*0.8+(self.gamma_input2_accumulation/steps)*0.2, requires_grad=False)
        self.gamma_input3 = nn.Parameter(self.gamma_input3*0.8+(self.gamma_input3_accumulation/steps)*0.2, requires_grad=False)
        # print("mlp gammas:", self.gamma_input1.data, self.gamma_input2.data, self.gamma_input3.data)
        self.gamma_input1_accumulation = 0
        self.gamma_input2_accumulation = 0
        self.gamma_input3_accumulation = 0
        
        
class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_head=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., drop_path=0.):
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
            self.q_bias = nn.Parameter(torch.rand((dim))) 
            self.k_bias = nn.Parameter(torch.rand((dim)))
            self.v_bias = nn.Parameter(torch.rand((dim)))
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
        self.gamma_input1 = nn.Parameter(torch.ones((1)) * 50, requires_grad=False)
        self.gamma_input2 = nn.Parameter(torch.ones((1)) * 50, requires_grad=False)
        self.gamma_input3 = nn.Parameter(torch.ones((1)) * 50, requires_grad=False)
        self.gamma_input1_accumulation = 0
        self.gamma_input2_accumulation = 0
        self.gamma_input3_accumulation = 0
        
        self.gamma1 = 1 # nn.Parameter(torch.zeros(1))
        self.gamma1_accumulation = 0
        self.gamma2 = 1 # nn.Parameter(torch.zeros(1))
        self.gamma2_accumulation = 0
        self.gamma3 = 1 # nn.Parameter(torch.zeros(1))
        self.gamma3_accumulation = 0
        #################### ↑↑↑ Shortcut scale ↑↑↑ ####################
        
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x, display=False):
        if True: #self.training:
            B, N, C = x.shape
            
            ########### ↓↓↓ Standardization ↓↓↓ ###########
            q_weight = stadardization(self.q_weight, dim_in=self.dim_in, num_head=self.num_head, dim_head=self.dim_head)
            k_weight = stadardization(self.k_weight, dim_in=self.dim_in, num_head=self.num_head, dim_head=self.dim_head)
            v_weight = stadardization(self.v_weight, dim_in=self.dim_in, num_head=self.num_head, dim_head=self.dim_head)
            
            q_bias = stadardization(self.q_bias, dim_in=self.dim_in, num_head=self.num_head, dim_head=self.dim_head)
            k_bias = stadardization(self.k_bias, dim_in=self.dim_in, num_head=self.num_head, dim_head=self.dim_head)
            v_bias = stadardization(self.v_bias, dim_in=self.dim_in, num_head=self.num_head, dim_head=self.dim_head)
            
            proj_weight = stadardization(self.proj_weight, dim_in=self.dim_in, num_head=self.num_head, dim_head=self.dim_head)
            proj_bias = stadardization(self.proj_bias, dim_in=self.dim_in, num_head=self.num_head, dim_head=self.dim_head)
            ########### ↑↑↑ Standardization ↑↑↑ ###########
            
            ########### ↓↓↓ Self-attention ↓↓↓ ############
            # V's shortcut
            shortcut = x
            self.gamma_input1_accumulation += x.std(-1, correction=0).mean().item()
                
            # Calculate Query (Q), Key (K) and Value (V)
            q = torch.nn.functional.linear(x / self.gamma_input1, q_weight, q_bias) # B, N, C
            k = torch.nn.functional.linear(x, k_weight, k_bias) # B, N, C
            v = torch.nn.functional.linear(x / self.gamma_input1, v_weight, v_bias) # B, N, C
                
            # Add shortcut and droppath to V
            v = self.drop_path(v) * self.gamma1 + shortcut # B, N, C
            
            # Reshape Query (Q), Key (K) and Value (V)
            q = rearrange(q, 'b n (nh hc) -> b nh n hc', nh=self.num_head) # B, nh, N, C//nh
            k = rearrange(k, 'b n (nh hc) -> b nh hc n', nh=self.num_head) # B, nh, C//nh, N
            v = rearrange(v, 'b n (nh hc) -> b nh n hc', nh=self.num_head) # B, nh, N, C//nh
                
            # Attended x's shortcut
            shortcut = v # B, nh, N, C//nh
            self.gamma_input2_accumulation += v.std(-1, correction=0).mean().item()
            
            # Calculate attention map
            attn = q @ k # B, nh, N, N
            attn = attn.softmax(dim=-1) # B, nh, N, N
            attn = self.attn_drop(attn) # B, nh, N, N
            
            # Calculate attended x
            x = attn @ (v / self.gamma_input2) # B, nh, N, C//nh
            
            # Add shortcut and droppath to attended x
            x = self.drop_path(x) * self.gamma2 + shortcut # B, nh, N, C//nh
            
            # Reshape x back to input shape
            x = rearrange(x, 'b nh n hc -> b n (nh hc)', nh=self.num_head) # B, nh, N, C//nh
            ########### ↑↑↑ Self-attention ↑↑↑ ############
            
            ########## ↓↓↓ Linear projection ↓↓↓ ##########
            # Linear projection
            shortcut = x # B, N, C
            self.gamma_input3_accumulation += x.std(-1, correction=0).mean().item()
            
            x = torch.nn.functional.linear(x / self.gamma_input3, proj_weight, proj_bias) # B, N, C
            
            # Add shortcut and droppath to output
            x = self.drop_path(x) * self.gamma3 + shortcut # B, N, C
            ########## ↑↑↑ Linear projection ↑↑↑ ##########
            
            #if display and x.get_device() == 0:# and self.v_weight.grad is not None:
                #print("mhsa gammas:", self.gamma_input1.data, self.gamma_input2.data, self.gamma_input3.data)
                #print("x:", x.var(-1).mean(), x.mean(), x.max(), x.min())
                #print("q_weight:", q_weight.var(-1).mean(), q_weight.mean(), q_weight.max(), q_weight.min())
                #print("V:", v_bias)
                #print("V:", self.v_weight.grad.mean(), self.v_weight.grad.max(), self.v_weight.grad.min())
                #print("V weight:", self.v_weight.grad.mean(), self.v_weight.grad.max(), self.v_weight.grad.min())
                #print("K weight:", self.k_weight.grad.mean(), self.k_weight.grad.max(), self.k_weight.grad.min())
                #print("Q weight:", self.q_weight.grad.mean(), self.q_weight.grad.max(), self.q_weight.grad.min())
                #print("F weight:", self.proj_weight.grad.mean(), self.proj_weight.grad.max(), self.proj_weight.grad.min())
            return x
    
    def adaptive_gamma(self, steps):
        self.gamma_input1 = nn.Parameter(self.gamma_input1*0.8+(self.gamma_input1_accumulation/steps)*0.2, requires_grad=False)
        self.gamma_input2 = nn.Parameter(self.gamma_input2*0.8+(self.gamma_input2_accumulation/steps)*0.2, requires_grad=False)
        self.gamma_input3 = nn.Parameter(self.gamma_input3*0.8+(self.gamma_input3_accumulation/steps)*0.2, requires_grad=False)
        #print("mhsa gammas:", self.gamma_input1.data, self.gamma_input2.data, self.gamma_input3.data)
        self.gamma_input1_accumulation = 0
        self.gamma_input2_accumulation = 0
        self.gamma_input3_accumulation = 0
        
        

class NFAttentionBlock(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_head, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.Sigmoid): 
        super().__init__()
        self.attn = Attention(dim, num_head=num_head, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                              attn_drop=attn_drop, proj_drop=drop, drop_path=drop_path)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim_in=dim, dim_hidden=mlp_hidden_dim, num_head=num_head, bias=qkv_bias,
                       act_layer=act_layer, drop=drop, drop_path=drop_path)
    
    def forward(self, x, display=True):
        x = self.attn(x, display)
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
                act_layer=act_layer
            )
            for i in range(depth)])
        
        self.num_head = num_heads
        self.dim_head = embed_dim//self.num_head
        self.norm_pre = norm_layer(self.dim_head, elementwise_affine=False) if pre_norm else nn.Identity()
        self._init_standard_weights()
        self.head_weight = nn.Parameter(torch.rand((embed_dim, num_classes)))
        self.head_bias = nn.Parameter(torch.rand((num_classes)))
        self.norm = None
        self.head.weight.requires_grad_(False)
        self.head.bias.requires_grad_(False)
        self.gamma = nn.Parameter(torch.ones((1)) * 50, requires_grad=False)
        self.gamma_accumulation = 0
            
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        
        B, N, C = x.shape
        x = x.reshape(B, N, self.num_head, self.dim_head)
        x = self.norm_pre(x)
        x = x.reshape(B, N, C)
        
        for i, blk in enumerate(self.blocks):
            #x.register_hook(make_print_grad("x of layer "+str(i)))
            x = blk(x)
            
        # x = self.norm(x)
        return x
    
    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        
        if pre_logits:
            return x
        else:
            head_weight = stadardization(self.head_weight, dim_in=self.embed_dim, num_head=1, dim_head=1000)
            head_bias = stadardization(self.head_bias, dim_in=self.embed_dim, num_head=1, dim_head=1000)
            self.gamma_accumulation += x.std(-1, correction=0).mean().item()
            x = torch.nn.functional.linear(x / self.gamma, head_weight, head_bias) # B, N, C
            return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        #x.register_hook(make_print_grad("x"))
        return x
        
    def _init_standard_weights(self):
        for name, param in self.named_parameters():
            if "_weight" in name:
                nn.init.trunc_normal_(param, std=.1, a=-2, b=2)
            elif "_bias" in name:
                nn.init.trunc_normal_(param, std=.1, a=-2, b=2)
                #nn.init.zeros_(param)
                
    def adaptive_gamma(self, steps):
        self.gamma = nn.Parameter(self.gamma*0.8+(self.gamma_accumulation/steps)*0.2, requires_grad=False)
        self.gamma_accumulation = 0
        #print("output gamma:", self.gamma.data)
        for blk in self.blocks:
            blk.adaptive_gamma(steps)
        
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