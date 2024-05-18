# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from einops import rearrange, reduce

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models._registry import register_model
from timm.layers import trunc_normal_, PatchEmbed
from timm.layers.helpers import to_2tuple
import math
import torch.autograd.profiler as profiler
import torch.utils.checkpoint as ckpt



class CustomizedDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(CustomizedDropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x, origin=None):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep and origin is None:
            random_tensor.div_(keep_prob)
        
        if origin is None:
            return x * random_tensor
        else:
            assert origin.shape==x.shape
            return x * random_tensor + origin * (1 - random_tensor)



def make_print_grad(parameter):
    def print_grad(grad):
        print(f"Gradient of ({parameter}):", grad.max(), grad.min(), grad.norm())
    return print_grad

    

def standardization(W, dim_in, num_head, dim_head):
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
    scale = torch.maximum(std*(dim_head**0.5), torch.ones(std.shape, device=std.device)*1e-4) #
    W = (W - mean) / scale
    
    #W.register_hook(make_print_grad("W after standardization"))
        
    if num_dim == 1:
        W = W.reshape(num_head * dim_head)
    elif num_dim == 2:
        W = W.reshape(dim_in, num_head * dim_head).T
    elif num_dim == 4:
        W = W.reshape(dim_in, 1, k, k)
        
    assert not W.isnan().any() # prevent nan value after standardization
    #print("W:", W.var(), W.max(), W.min())
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
            act_layer=nn.GELU,
            bias=True,
            use_conv=False,
            drop=0.,
            drop_path=0.,
            shortcut_type='PerLayer',
            weight_standardization=False,
            feature_norm="LayerNorm",
            shortcut_gain=0.0,
            gamma=0.1,
            std=1.0
            ):
        super().__init__()
        
        # Hyperparameters
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden or dim_in
        self.dim_out = dim_out or dim_in
        self.num_head = num_head
        self.shortcut_type = shortcut_type
        self.weight_standardization = weight_standardization
        
        ############################ ↓↓↓ 2-layer MLP ↓↓↓ ###########################
        # Trainable parameters
        self.fc1_weight = nn.Parameter(torch.zeros((self.dim_in, self.dim_hidden)))
        self.fc2_weight = nn.Parameter(torch.zeros((self.dim_hidden, self.dim_out)))
        self.fc1_bias = nn.Parameter(torch.zeros((self.dim_hidden)), requires_grad=bias)
        self.fc2_bias = nn.Parameter(torch.zeros((self.dim_out)), requires_grad=bias)
        
        # Weight standardization parameters
        if self.weight_standardization:
            self.gamma_fc1 = nn.Parameter(torch.ones((1, dim_in))*gamma, requires_grad=False)
            self.gamma_fc2 = nn.Parameter(torch.ones((1, dim_hidden))*gamma, requires_grad=False)
            
        # Activation, GELU by default
        self.act = act_layer()
        ############################ ↑↑↑ 2-layer MLP ↑↑↑ ###########################
        
        ########################## ↓↓↓ Shortcut scale ↓↓↓ ##########################
        self.shortcut_type = shortcut_type
        if self.shortcut_type == "PerOperation":
            self.shortcut_gain = nn.Parameter(torch.ones((1))*shortcut_gain, requires_grad=False)
        ########################## ↑↑↑ Shortcut scale ↑↑↑ ##########################
        
        ########################### ↓↓↓ Normalization ↓↓↓ ##########################
        self.feature_norm = feature_norm
        if self.feature_norm == "LayerNorm":
            self.norm = nn.LayerNorm(dim_in, elementwise_affine=True)
        elif self.feature_norm == "BatchNorm":
            self.norm1 = nn.BatchNorm1d(dim_in, affine=True)
            self.norm2 = nn.BatchNorm1d(dim_hidden, affine=True)
        elif self.feature_norm == "EmpiricalSTD":
            self.feature_std = nn.Parameter(torch.ones((1))*std, requires_grad=False)
        ########################### ↑↑↑ Normalization ↑↑↑ ##########################
        
        ######################### ↓↓↓ DropPath & Dropout ↓↓↓ #######################
        # Drop path
        self.drop_path = CustomizedDropPath(drop_path) if drop_path > 0. else None
        ######################### ↑↑↑ DropPath & Dropout ↑↑↑ #######################
            
    def forward(self, x):
        B, N, C = x.shape
        ###################### ↓↓↓ Standardization ↓↓↓ ######################
        if self.weight_standardization:
            fc1_weight = standardization(self.fc1_weight, 
                                         dim_in = self.dim_in, 
                                         num_head = self.num_head, 
                                         dim_head = self.dim_hidden//self.num_head) * self.gamma_fc1
                                         
            fc2_weight = standardization(self.fc2_weight, 
                                         dim_in = self.dim_hidden, 
                                         num_head = self.num_head,
                                         dim_head = self.dim_out//self.num_head) * self.gamma_fc2
            
            fc1_bias = self.fc1_bias - self.fc1_bias.mean()
            fc2_bias = self.fc2_bias - self.fc2_bias.mean()
            
        else:
            fc1_weight = self.fc1_weight.T
            fc2_weight = self.fc2_weight.T
            fc1_bias = self.fc1_bias
            fc2_bias = self.fc2_bias
        ###################### ↑↑↑ Standardization ↑↑↑ ######################
            
        ######################## ↓↓↓ 2-layer MLP ↓↓↓ ########################
        if self.shortcut_type == "PerLayer":
            shortcut = x # B, N, C
            
            # Feature normalization
            if self.feature_norm == "LayerNorm":
                x = self.norm(x)
            elif self.feature_norm == "BatchNorm":
                x = self.norm(x.transpose(-1,-2)).transpose(-1, -2)
            elif self.feature_norm == "EmpiricalSTD":
                x = x / self.feature_std.unsqueeze(0).unsqueeze(-1)
            else:
                pass
            
            # FFN in
            x = nn.functional.linear(x, fc1_weight, fc1_bias) # B, N, 4C
            
            x = self.act(x)
                    
            # FFN out
            x = nn.functional.linear(x, fc2_weight, fc2_bias) # B, N, C
            
            x = self.drop_path(x, None) if self.drop_path is not None else x
            
            x = x + shortcut
            
        elif self.shortcut_type == "PerOperation":            
            shortcut = x
            
            # Feature normalization
            if self.feature_norm == "LayerNorm":
                x = self.norm(x)
            elif self.feature_norm == "BatchNorm":
                x = self.norm1(x.transpose(-1,-2)).transpose(-1, -2)
            elif self.feature_norm == "EmpiricalSTD":
                x = x / self.feature_std.unsqueeze(0).unsqueeze(-1)
            else:
                pass
                
            # FFN in
            x = nn.functional.linear(x, fc1_weight, fc1_bias) # B, N, 4C
            
            # Activation on part of the channels
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask[:, :, :C] = True
            x = torch.where(mask, self.act(x), x)
            
            if self.feature_norm == "BatchNorm":
                x = self.norm2(x.transpose(-1,-2)).transpose(-1, -2)

            # FFN out
            x = nn.functional.linear(x, fc2_weight, fc2_bias) # B, N, C
            
            # Add DropPath and shortcut gain
            x = self.drop_path(x * self.shortcut_gain) if self.drop_path is not None else x * self.shortcut_gain
            
            # Add shortcut to x
            x = x + shortcut # B, N, C
            
        ######################## ↑↑↑ 2-layer MLP ↑↑↑ ########################
        #if x.get_device() == 0:
            #print("x after ffn:", x.std(-1).mean().item(), x.mean().item(), x.max().item(), x.min().item())
            #print("Shortcut gain", self.shortcut_gain.data.item())
        return x
        
    def reparam(self):
        if self.weight_standardization:
            fc1_weight = standardization(self.fc1_weight, 
                                         dim_in = self.dim_in, 
                                         num_head = self.num_head, 
                                         dim_head = self.dim_hidden//self.num_head) 
                                            
            fc2_weight = standardization(self.fc2_weight, 
                                         dim_in = self.dim_hidden, 
                                         num_head = self.num_head,
                                         dim_head = self.dim_out//self.num_head) 
            fc1_bias = self.fc1_bias - self.fc1_bias.mean()
            fc2_bias = self.fc2_bias - self.fc2_bias.mean()
                                          
            ffn_weight = fc2_weight @ fc1_weight + torch.eye(self.dim_in)
            ffn_bias = nn.functional.linear(fc1_bias.unsqueeze(0), fc2_weight, fc2_bias).squeeze()
        else:
            assert False; "This model is not RepViT or lacks proper weight gammas"
        
        return ffn_weight, ffn_bias
        
        
        
class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, 
                 dim, 
                 num_head=8, 
                 qkv_bias=False, 
                 qk_scale=None, 
                 attn_drop=0., 
                 proj_drop=0., 
                 drop_path=0., 
                 shortcut_type='PerLayer',
                 weight_standardization=False,
                 feature_norm="LayerNorm",
                 shortcut_gain=0.0,
                 gamma=0.1,
                 std=1.0,
                 ):
        super().__init__()
        
        # Hyperparameters
        self.num_head = num_head
        self.dim_head = dim // num_head
        self.dim_in = dim
        self.scale = qk_scale or self.dim_head ** -0.5 # scale
        self.weight_standardization = weight_standardization
        
        #################### ↓↓↓ Self Attention ↓↓↓ ####################
        # Trainable parameters
        self.q_weight = nn.Parameter(torch.zeros((dim, dim)))
        self.k_weight = nn.Parameter(torch.zeros((dim, dim)))
        self.v_weight = nn.Parameter(torch.zeros((dim, dim)))
        self.q_bias = nn.Parameter(torch.zeros((dim)), requires_grad=qkv_bias) 
        self.k_bias = nn.Parameter(torch.zeros((dim)), requires_grad=qkv_bias) 
        self.v_bias = nn.Parameter(torch.zeros((dim)), requires_grad=qkv_bias) 
        
        # Weight standardization parameters
        if self.weight_standardization:
            self.gamma_q = nn.Parameter(torch.ones((1, dim))*gamma, requires_grad=False)
            self.gamma_k = nn.Parameter(torch.ones((1, dim))*gamma, requires_grad=False)
            self.gamma_v = nn.Parameter(torch.ones((1, dim))*gamma, requires_grad=False)
        #################### ↑↑↑ Self Attention ↑↑↑ ####################
        
        #################### ↓↓↓ Output Linear ↓↓↓ #####################
        # Trainable parameters
        self.proj_weight = nn.Parameter(torch.rand((dim, dim)))
        self.proj_bias = nn.Parameter(torch.rand((dim)), requires_grad=qkv_bias)
        
        # Weight standardization parameters
        if self.weight_standardization:
            self.gamma_proj = nn.Parameter(torch.ones((1, dim))*gamma, requires_grad=False)
        #################### ↑↑↑ Output Linear ↑↑↑ #####################
        
        #################### ↓↓↓ Shortcut scale ↓↓↓ ####################
        self.shortcut_type = shortcut_type
        if self.shortcut_type == "PerOperation":
            self.shortcut_gain1 = nn.Parameter(torch.ones((1))*shortcut_gain, requires_grad=False)
            self.shortcut_gain2 = nn.Parameter(torch.ones((1))*shortcut_gain, requires_grad=False)
            self.shortcut_gain3 = nn.Parameter(torch.ones((1))*shortcut_gain, requires_grad=False)
        #################### ↑↑↑ Shortcut scale ↑↑↑ ####################
        
        ################### ↓↓↓ DropPath & Dropout ↓↓↓ #################
        # Drop path
        self.drop_path = CustomizedDropPath(drop_path) if drop_path > 0. else None
        # Attention drop
        self.attn_drop = attn_drop
        ################### ↑↑↑ DropPath & Dropout ↑↑↑ #################
        
        ##################### ↓↓↓ Normalization ↓↓↓ ####################
        self.feature_norm = feature_norm
        if self.feature_norm == "LayerNorm":
            if self.shortcut_type == "PerOperation":
                self.norm1 = nn.LayerNorm(dim)
                self.norm2 = nn.LayerNorm(dim)
                self.norm3 = nn.LayerNorm(dim)
            else:
                self.norm = nn.LayerNorm(dim)
        elif self.feature_norm == "BatchNorm":
            if self.shortcut_type == "PerOperation":
                self.norm1 = nn.BatchNorm1d(dim)
                self.norm2 = nn.BatchNorm1d(dim)
                self.norm3 = nn.BatchNorm1d(dim)
            else:
                self.norm = nn.BatchNorm1d(dim)
        elif self.feature_norm == "EmpiricalSTD":
            if self.shortcut_type == "PerOperation":
                self.feature_std1 = nn.Parameter(torch.ones((1))*std)
                self.feature_std2 = nn.Parameter(torch.ones((1))*std)
                self.feature_std3 = nn.Parameter(torch.ones((1))*std)
            else:
                self.feature_std = nn.Parameter(torch.ones((1))*std)
        ##################### ↑↑↑ Normalization ↑↑↑ ####################
        
    def forward(self, x):
        B, N, C = x.shape
        ######################### ↓↓↓ Standardization ↓↓↓ #########################
        if self.weight_standardization:
            q_weight = standardization(self.q_weight, 
                                       dim_in=self.dim_in,
                                       num_head=self.num_head,
                                       dim_head=self.dim_head) * self.gamma_q 
                                          
            k_weight = standardization(self.k_weight,
                                       dim_in=self.dim_in,
                                       num_head=self.num_head,
                                       dim_head=self.dim_head) * self.gamma_k
                                          
            v_weight = standardization(self.v_weight, 
                                       dim_in=self.dim_in,
                                       num_head=self.num_head,
                                       dim_head=self.dim_head) * self.gamma_v
                                          
            proj_weight = standardization(self.proj_weight,
                                          dim_in=self.dim_in,
                                          num_head=self.num_head,
                                          dim_head=self.dim_head) * self.gamma_proj
            
            q_bias = self.q_bias - self.q_bias.mean()
            k_bias = self.k_bias - self.k_bias.mean()
            v_bias = self.v_bias - self.v_bias.mean()
            proj_bias = self.proj_bias - self.proj_bias.mean()
        
        else:
            q_weight = self.q_weight.T
            k_weight = self.k_weight.T
            v_weight = self.v_weight.T
            proj_weight = self.proj_weight.T
            q_bias = self.q_bias
            k_bias = self.k_bias
            v_bias = self.v_bias
            proj_bias = self.proj_bias
        ######################### ↑↑↑ Standardization ↑↑↑ #########################
            
        ######################### ↓↓↓ Self-attention ↓↓↓ ##########################
        if self.shortcut_type == "PerLayer":
            # Shortcut
            shortcut = x
            
            # Feature normalization
            if self.feature_norm == "LayerNorm":
                x = self.norm(x)
            elif self.feature_norm == "BatchNorm":
                x = self.norm(x.transpose(-1,-2)).transpose(-1,-2)
            elif self.feature_norm == "EmpiricalSTD":
                x = x / self.feature_std.unsqueeze(0).unsqueeze(-1)
            else:
                pass
                
            # Calculate Query (Q), Key (K) and Value (V)
            q = nn.functional.linear(x, q_weight, q_bias) # B, N, C
            k = nn.functional.linear(x, k_weight, k_bias) # B, N, C
            v = nn.functional.linear(x, v_weight, v_bias) # B, N, C
                
            # Reshape Query (Q), Key (K) and Value (V)
            q = rearrange(q, 'b n (nh hc) -> b nh n hc', nh=self.num_head) # B, nh, N, C//nh
            k = rearrange(k, 'b n (nh hc) -> b nh n hc', nh=self.num_head) # B, nh, N, C//nh
            v = rearrange(v, 'b n (nh hc) -> b nh n hc', nh=self.num_head) # B, nh, N, C//nh
                
            # Calculate self-attention
            x = nn.functional.scaled_dot_product_attention(q, k, v) # B, nh, N, C//nh
                
            # Reshape x back to input shape
            x = rearrange(x, 'b nh n hc -> b n (nh hc)', nh=self.num_head) # B, N, C
                
            # Output linear projection
            x = nn.functional.linear(x, proj_weight, proj_bias) # B, N, C 
                
            # Add DropPath
            x = self.drop_path(x, None) if self.drop_path is not None else x # B, N, C
            
            # Add shortcut
            x = x + shortcut # B, N, C
                
        elif self.shortcut_type == "PerOperation":
            # Shortcut
            shortcut = x
            
            # Feature normalization
            if self.feature_norm == "LayerNorm":
                x = self.norm1(x)
            elif self.feature_norm == "BatchNorm":
                x = self.norm1(x.transpose(-1,-2)).transpose(-1,-2)
            elif self.feature_norm == "EmpiricalSTD":
                x = x / self.feature_std1.unsqueeze(0).unsqueeze(-1)
            
            # Calculate Query (Q), Key (K) and Value (V)
            q = nn.functional.linear(x, q_weight, q_bias) # B, N, C
            k = nn.functional.linear(x, k_weight, k_bias) # B, N, C
            v = nn.functional.linear(x, v_weight, v_bias) # B, N, C
                
            # Add DropPath and shortcut gain
            v = self.drop_path(v * self.shortcut_gain1) if self.drop_path is not None else v * self.shortcut_gain1 # B, N, C
            
            # Add shortcut
            v = v + shortcut # B, N, C
            
            # Shortcut
            shortcut = v
            
            # Feature normalization
            if self.feature_norm == "LayerNorm":
                v = self.norm2(v)
            elif self.feature_norm == "BatchNorm":
                v = self.norm2(v.transpose(-1,-2)).transpose(-1,-2)
            elif self.feature_norm == "EmpiricalSTD":
                v = v / self.feature_std2.unsqueeze(0).unsqueeze(-1)
            
            # Reshape Query (Q), Key (K) and Value (V)
            q = rearrange(q, 'b n (nh hc) -> b nh n hc', nh=self.num_head) # B, nh, N, C//nh
            k = rearrange(k, 'b n (nh hc) -> b nh n hc', nh=self.num_head) # B, nh, N, C//nh
            v = rearrange(v, 'b n (nh hc) -> b nh n hc', nh=self.num_head) # B, nh, N, C//nh
                
            # Calculate self-attention
            x = nn.functional.scaled_dot_product_attention(q, k, v) # B, nh, N, C//nh
            
            # Reshape x back to input shape
            x = rearrange(x, 'b nh n hc -> b n (nh hc)', nh=self.num_head) # B, N, C
            
            # Add DropPath and shortcut gain
            x = self.drop_path(x * self.shortcut_gain2) if self.drop_path is not None else x * self.shortcut_gain2 # B, N, C
            
            # Add shortcut
            x = x + shortcut # B, N, C
            
            # Shortcut
            shortcut = x
            
            # Feature normalization
            if self.feature_norm == "LayerNorm":
                x = self.norm3(x)
            elif self.feature_norm == "BatchNorm":
                x = self.norm3(x.transpose(-1,-2)).transpose(-1,-2)
            elif self.feature_norm == "EmpiricalSTD":
                x = x / self.feature_std3.unsqueeze(0).unsqueeze(-1)
                
            # Linear projection
            x = nn.functional.linear(x, proj_weight, proj_bias) # B, N, C
            
            # Add DropPath and shortcut gain
            x = self.drop_path(x * self.shortcut_gain3) if self.drop_path is not None else x * self.shortcut_gain3 # B, N, C
            
            # Add shortcut
            x = x + shortcut # B, N, C
        ######################### ↑↑↑ Self-attention ↑↑↑ ##########################
        #if x.get_device() == 0:
            #print("x after mhsa:", x.std(-1).mean().item(), x.mean().item(), x.max().item(), x.min().item())
            #print("Shortcut gain", self.shortcut_gain1.data.item(), self.shortcut_gain2.data.item(), self.shortcut_gain3.data.item())
        return x
        
    def reparam(self):
        if self.weight_standardization:
            q_weight = standardization(self.q_weight, 
                                       dim_in=self.dim_in, 
                                       num_head=self.num_head, 
                                       dim_head=self.dim_head) * self.gamma_q
                                          
            k_weight = standardization(self.k_weight, 
                                       dim_in=self.dim_in, 
                                       num_head=self.num_head, 
                                       dim_head=self.dim_head) * self.gamma_k
                                          
            v_weight = standardization(self.v_weight, 
                                       dim_in=self.dim_in, 
                                       num_head=self.num_head, 
                                       dim_head=self.dim_head) * self.gamma_v
                                          
            proj_weight = standardization(self.proj_weight, 
                                          dim_in=self.dim_in,
                                          num_head=self.num_head,
                                          dim_head=self.dim_head) * self.gamma_proj
                                          
            v_weight = (proj_weight + torch.eye(self.dim_in)) @ (v_weight + torch.eye(self.dim_in))
            q_bias = self.q_bias - self.q_bias.mean()
            k_bias = self.k_bias - self.k_bias.mean()
            v_bias = self.proj_bias - self.proj_bias.mean()
        else:
            assert False; "This model is not RepViT or lacks proper weight gammas"
        
        return q_weight, k_weight, v_weight, q_bias, k_bias, v_bias



class RepAttention(nn.Module):
    def __init__(self, 
                 dim, 
                 num_head,
                 q_weight,
                 k_weight,
                 v_weight,
                 q_bias,
                 k_bias,
                 v_bias
                 ):
        super().__init__()
        
        # Hyperparameters
        self.num_head = num_head
        self.dim_head = dim // num_head
        self.dim = dim
        self.scale = self.dim_head ** -0.5 # scale
        
        self.q_weight = nn.Parameter(q_weight)
        self.k_weight = nn.Parameter(k_weight)
        self.v_weight = nn.Parameter(v_weight)
        self.q_bias = nn.Parameter(q_bias)
        self.k_bias = nn.Parameter(k_bias)
        self.v_bias = nn.Parameter(v_bias)
        
    def forward(self, x):
        B, N, C = x.shape
            
        # Calculate Query (Q), Key (K) and Value (V)
        q = nn.functional.linear(x, self.q_weight, self.q_bias) # B, N, C
        k = nn.functional.linear(x, self.k_weight, self.k_bias) # B, N, C
        v = nn.functional.linear(x, self.v_weight, self.v_bias) # B, N, C
                
        # Reshape Query (Q), Key (K) and Value (V)
        q = rearrange(q, 'b n (nh hc) -> b nh n hc', nh=self.num_head) # B, nh, N, C//nh
        k = rearrange(k, 'b n (nh hc) -> b nh n hc', nh=self.num_head) # B, nh, N, C//nh
        v = rearrange(v, 'b n (nh hc) -> b nh n hc', nh=self.num_head) # B, nh, N, C//nh
                
        # Calculate self-attention
        x = nn.functional.scaled_dot_product_attention(q, k, v) # B, nh, N, C//nh
        
        x = rearrange(x, 'b nh n hc -> b n (nh hc)', nh=self.num_head) # B, N, C
        
        x = nn.functional.gelu(x)
        
        return x
                    
        

class NFAttentionBlock(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_head, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, shortcut_type='PerLayer', weight_standardization=False,
                 affected_layers='None', feature_norm="LayerNorm", shortcut_gain=0.0, gamma=0.1, std=[1.0],
                 qk_gamma=1, proj_gamma=1): 
        super().__init__()
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.rep = False
        self.dim = dim
        self.num_head = num_head
        
        if feature_norm == "GroupedLayerNorm":
            feature_norm = "None"
            
        if affected_layers=="Both":
            self.attn = Attention(dim, num_head=num_head, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                  attn_drop=attn_drop, proj_drop=drop, drop_path=drop_path, 
                                  shortcut_type=shortcut_type, weight_standardization=weight_standardization,
                                  feature_norm=feature_norm, shortcut_gain=shortcut_gain, gamma=gamma, std=std[0])
            self.mlp = Mlp(dim_in=dim, dim_hidden=mlp_hidden_dim, num_head=num_head, bias=qkv_bias,
                           act_layer=act_layer, drop=drop, drop_path=drop_path, shortcut_type=shortcut_type,
                           weight_standardization=weight_standardization, feature_norm=feature_norm, 
                           shortcut_gain=shortcut_gain, gamma=gamma, std=std[0])
        elif affected_layers=="MHSA":
            self.attn = Attention(dim, num_head=num_head, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                  attn_drop=attn_drop, proj_drop=drop, drop_path=drop_path, 
                                  shortcut_type=shortcut_type, weight_standardization=weight_standardization,
                                  feature_norm=feature_norm, shortcut_gain=shortcut_gain, gamma=gamma, std=std[0])
            self.mlp = Mlp(dim_in=dim, dim_hidden=mlp_hidden_dim, num_head=num_head, bias=qkv_bias,
                           act_layer=act_layer, drop=drop, drop_path=drop_path)
        elif affected_layers=="FFN":
            self.attn = Attention(dim, num_head=num_head, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                  attn_drop=attn_drop, proj_drop=drop, drop_path=drop_path)
            self.mlp = Mlp(dim_in=dim, dim_hidden=mlp_hidden_dim, num_head=num_head, bias=qkv_bias,
                           act_layer=act_layer, drop=drop, drop_path=drop_path, shortcut_type=shortcut_type,
                           weight_standardization=weight_standardization, feature_norm=feature_norm,
                           shortcut_gain=shortcut_gain, gamma=gamma, std=std[0])
        elif affected_layers=="None":
            self.attn = Attention(dim, num_head=num_head, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                  attn_drop=attn_drop, proj_drop=drop, drop_path=drop_path)
            self.mlp = Mlp(dim_in=dim, dim_hidden=mlp_hidden_dim, num_head=num_head, bias=qkv_bias,
                           act_layer=act_layer, drop=drop, drop_path=drop_path)
    
    def forward(self, x):
        if not self.rep:
            x = self.attn(x)
            x = self.mlp(x)
            #print("x after mhsa:", x.std(-1).mean().item(), x.mean().item(), x.max().item(), x.min().item())
        else:
            x = self.attn(x)
        return x
        
    def adaptive_std(self, steps):
        self.attn.adaptive_std(steps)
        self.mlp.adaptive_std(steps)
        
    def clean_std(self):
        self.attn.clean_std()
        self.mlp.clean_std()
    
    def reparam(self):
        q_weight, k_weight, v_weight, q_bias, k_bias, v_bias = self.attn.reparam()
        ffn_weight, ffn_bias = self.mlp.reparam()
        v_weight = ffn_weight @ v_weight
        v_bias = nn.functional.linear(v_bias.unsqueeze(0), ffn_weight, ffn_bias).squeeze()
        self.rep = True
        del self.attn
        del self.mlp
        self.attn = RepAttention(self.dim, self.num_head, q_weight, k_weight, v_weight, q_bias, k_bias, v_bias)



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
            mlp_ratio=2.,
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
            block_fn=NFAttentionBlock,
            shortcut_type='PerLayer', # ['PerLayer', 'PerOperation']
            affected_layers='None', # ['None', 'Both', 'MHSA', 'FFN']
            feature_norm='LayerNorm', # ['GroupedLayerNorm', 'LayerNorm', 'BatchNorm', 'EmpiricalSTD', 'None']
            weight_standardization=False, # [True, False]
            shortcut_gain=0.0,
            gamma=0.1,
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
        std = [x.item() for x in torch.logspace(start=0, end=2, steps=depth, base=2)]
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
                shortcut_type=shortcut_type,
                affected_layers=affected_layers,
                weight_standardization=weight_standardization,
                feature_norm=feature_norm,
                shortcut_gain=shortcut_gain,
                gamma=gamma,
                std=std[i:i+1],
            )
            for i in range(depth)])
        
        self.num_head = num_heads
        self.dim_head = embed_dim//self.num_head
        self.pre_norm = pre_norm
        
        self.feature_norm = feature_norm
        self.weight_standardization=weight_standardization
        
        self.checkpointing = False
        ############################ ↓↓↓ Output Head ↓↓↓ ###########################
        # Trainable parameters
        self.head = None
        self.head_weight = nn.Parameter(torch.rand((embed_dim, num_classes)))
        self.head_bias = nn.Parameter(torch.zeros((num_classes)))
        ############################ ↑↑↑ Output Head ↑↑↑ ###########################
        
        if self.feature_norm in ["LayerNorm", "EmpiricalSTD"]:
            self.norm_pre = nn.LayerNorm(embed_dim) if pre_norm else nn.Identity()
            self.norm = nn.LayerNorm(embed_dim) if not fc_norm else nn.Identity()
        elif self.feature_norm == "BatchNorm":
            self.norm_pre = nn.BatchNorm1d(embed_dim) if pre_norm else nn.Identity()
            self.norm = nn.BatchNorm1d(embed_dim) if not fc_norm else nn.Identity()
        elif self.feature_norm == "GroupedLayerNorm":
            self.norm_pre = nn.LayerNorm(self.dim_head, elementwise_affine=False) if pre_norm else nn.Identity()
            self.norm = nn.LayerNorm(embed_dim) if not fc_norm else nn.Identity()
        elif self.feature_norm == "None":
            self.norm_pre = nn.LayerNorm(embed_dim) if pre_norm else nn.Identity()
            self.norm = nn.Identity()
        else:
            assert False, "Feature normalization type not supported"
        
        self._init_standard_weights()
            
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        B, N, C = x.shape
        
        if self.pre_norm:
            if self.feature_norm in ["LayerNorm", "EmpiricalSTD", "None"]:
                x = self.norm_pre(x)
            elif self.feature_norm == "BatchNorm":
                x = self.norm_pre(x.transpose(-1, -2)).transpose(-1, -2)
            elif self.feature_norm == "GroupedLayerNorm":
                x = self.norm_pre(x.reshape(B, N, self.num_head, self.dim_head)).reshape(B, N, C)
            else:
                pass
        
        
        for i, blk in enumerate(self.blocks):
            if self.training and self.checkpointing:
                x = ckpt.checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.feature_norm == "BatchNorm":
            x = self.norm(x.transpose(-1, -2)).transpose(-1, -2)
        elif self.feature_norm in ["LayerNorm", "GroupedLayerNorm", "EmpiricalSTD", "None"]:
            x = self.norm(x)
        else:
            pass
        return x
    
    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        
        return x if pre_logits else nn.functional.linear(x, self.head_weight.T, self.head_bias)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
        
    def _init_standard_weights(self):
        for name, param in self.named_parameters():
            if "blocks" in name:
                layer = int(name.split(".")[1])
                if "_weight" in name:
                    #nn.init.xavier_normal_(param)
                    nn.init.trunc_normal_(param, mean=0.0, std=.02, a=-2, b=2)
                    #param.data.mul_(math.pow(12, -0.5))
                    #param.data.div_(max(math.pow(12, 0.5), 1))
                    #param.data.mul_(0.67*math.pow(12, -0.25))
                    #param.data.div_(max(math.pow(8.0*(layer+1), 0.25), 1))
                    #if "v_weight" in name or "proj_weight" in name or "fc1_weight" in name or "fc2_weight" in name:
                    #    param.data.mul_(0.67*math.pow(12, -0.25))
                elif "_bias" in name:
                    nn.init.zeros_(param)
            else:
                if "_weight" in name:
                    #nn.init.xavier_normal_(param)
                    nn.init.trunc_normal_(param, mean=0.0, std=.02, a=-2, b=2)
                    #param.data.mul_(0.67*math.pow(12, -0.25))
                    #param.data.mul_(math.pow(24, -0.5))
                    #param.data.mul_(math.pow(8.0 * 12, 0.25))
                elif "_bias" in name:
                    nn.init.zeros_(param)
                
            
    def reparam(self):
        for blk in self.blocks:
            blk.reparam()
            
            
        
@register_model
def normalization_free_deit_tiny_patch16_224_layer12(pretrained=False, pretrained_cfg=None, **kwargs):
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
    
@register_model
def normalization_free_deit_base_patch16_224_layer12(pretrained=False, pretrained_cfg=None, **kwargs):
    model = NFTransformer(patch_size=16, embed_dim=768, depth=12, pre_norm=True,
                          num_heads=12, mlp_ratio=4, qkv_bias=True, fc_norm=False,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model