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
        print(f"Gradient of ({parameter}):", grad.max(), grad.min())
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
            shortcut_gain=1.0,
            gamma=0.1,
            std=0.1
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
        if bias:
            self.fc1_bias = nn.Parameter(torch.zeros((self.dim_hidden)))
            self.fc2_bias = nn.Parameter(torch.zeros((self.dim_out)))
        else:
            self.fc1_bias = nn.Parameter(torch.zeros((self.dim_hidden)), requires_grad=False)
            self.fc2_bias = nn.Parameter(torch.zeros((self.dim_out)), requires_grad=False)
        
        # Weight standardization parameters
        if self.weight_standardization:
            self.gamma_fc1 = nn.Parameter(torch.ones((1))*gamma, requires_grad=False)
            self.gamma_fc2 = nn.Parameter(torch.ones((1))*gamma, requires_grad=False)
            
        # Activation, GELU by default
        self.act = act_layer()
        ############################ ↑↑↑ 2-layer MLP ↑↑↑ ###########################
        
        ########################## ↓↓↓ Shortcut scale ↓↓↓ ##########################
        self.shortcut_type = shortcut_type
        if self.shortcut_type == "PerOperation":
            self.shortcut_gain = 1 #nn.Parameter(torch.ones((1))*shortcut_gain, requires_grad=False)
        elif self.shortcut_type == "PerLayer":
            self.shortcut_gain = 1
        ########################## ↑↑↑ Shortcut scale ↑↑↑ ##########################
        
        ########################### ↓↓↓ Normalization ↓↓↓ ##########################
        self.feature_norm = feature_norm
        if self.feature_norm == "LayerNorm":
            self.norm = nn.LayerNorm(dim_in, elementwise_affine=False)
        elif self.feature_norm == "BatchNorm":
            self.norm = nn.BatchNorm1d(dim_in, affine=False)
        elif self.feature_norm == "None":
            self.feature_std = nn.Parameter(torch.ones((1))*std, requires_grad=False)
            self.feature_std_accumulation = nn.Parameter(torch.zeros((1)), requires_grad=False)
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
                                         dim_head = self.dim_hidden//self.num_head) * nn.functional.sigmoid(self.gamma_fc1) * 0.62
                                         
            fc2_weight = standardization(self.fc2_weight, 
                                         dim_in = self.dim_hidden, 
                                         num_head = self.num_head,
                                         dim_head = self.dim_out//self.num_head) * nn.functional.sigmoid(self.gamma_fc2) * 0.62
            
            fc1_bias = self.fc1_bias - self.fc1_bias.mean()
            fc2_bias = self.fc2_bias - self.fc2_bias.mean()
        else:
            fc1_weight = self.fc1_weight.T
            fc2_weight = self.fc2_weight.T
            fc1_bias = self.fc1_bias
            fc2_bias = self.fc2_bias
        ###################### ↑↑↑ Standardization ↑↑↑ ######################
            
        ######################## ↓↓↓ 2-layer MLP ↓↓↓ ########################
        # Shortcut
        shortcut = x # B, N, C
        
        #print("x before std:", x.std(-1).mean().item(), x.mean().item(), x.max().item(), x.min().item())
        
        # Feature normalization
        if self.feature_norm == "LayerNorm":
            x = self.norm(x)
        elif self.feature_norm == "BatchNorm":
            x = x.transpose(-1, -2)
            x = self.norm(x)
            x = x.transpose(-1, -2)
        else:
            #self.feature_std_accumulation.data = self.feature_std_accumulation.data + x.std(-1).mean().item()
            x = x / self.feature_std
        
        # FFN in
        x = nn.functional.linear(x, fc1_weight, fc1_bias) # B, N, 4C
                
        # FFN out
        x = nn.functional.linear(x, fc2_weight, fc2_bias) * self.shortcut_gain # B, N, C
                
        # Add DropPath
        x = self.drop_path(x) if self.drop_path is not None else x
        
        # Add shortcut
        x = x + shortcut
        ######################## ↑↑↑ 2-layer MLP ↑↑↑ ########################
        
        
        ######################## ↓↓↓ Activation ↓↓↓ #########################
        # Activation
        x = self.act(x)
        
        # If feature norm is `None`, i.e., weight standardization, 
        # then re-centerize the feature per head
        if self.weight_standardization:
            if self.feature_norm == "None":
                x = x.reshape(B, N, self.num_head, C//self.num_head)
                x = x - x.mean(-1, keepdim=True) # B, N, C
                x = x.reshape(B, N, C)
            else:
                x = x - x.mean(-1, keepdim=True) # B, N, C
        ######################## ↑↑↑ Activation ↑↑↑ #########################
        
        
        #if x.get_device() == 0:
            #print("x std:", self.feature_std.item())
            #print("x after ffn:", x.std(-1).mean().item(), x.mean().item(), x.max().item(), x.min().item())
            #print("gamma:", self.gamma_fc1.data, self.gamma_fc2.data)
            #print("weight 1:", fc1_weight.max(), fc1_weight.min())
            #print("weight 2:", fc2_weight.max(), fc2_weight.min())
            #print("act_ratio:", self.act_ratio.item())
            #print("Shortcut gain:", self.shortcut_gain)
        return x
        
    def adaptive_std(self, steps):
        print(self.feature_std.data, self.feature_std_accumulation.data)
        self.feature_std.data = self.feature_std.data * 0.7 + (self.feature_std_accumulation/steps).data * 0.3
        print(self.feature_std.data)
        self.feature_std_accumulation.data = self.feature_std_accumulation.data * 0
        
    def clean_std(self):
        self.feature_std_accumulation.data = self.feature_std_accumulation.data * 0
        
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
                 shortcut_gain=1.0,
                 gamma=0.1,
                 std=1,
                 qk_gamma=1,
                 proj_gamma=1
                 ):
        super().__init__()
        
        # Hyperparameters
        self.num_head = num_head
        self.dim_head = dim // num_head
        self.dim_in = dim
        self.scale = qk_scale or self.dim_head ** -0.5 # scale
        self.weight_standardization = weight_standardization
        
        #################### ↓↓↓ Self Attention ↓↓↓ ####################
        # Take place
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv.weight.requires_grad_(False)
        self.qkv.bias.requires_grad_(False)
        
        # Trainable parameters
        self.q_weight = nn.Parameter(torch.zeros((dim, dim)))
        self.k_weight = nn.Parameter(torch.zeros((dim, dim)))
        self.v_weight = nn.Parameter(torch.zeros((dim, dim)))
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros((dim))) 
            self.k_bias = nn.Parameter(torch.zeros((dim)))
            self.v_bias = nn.Parameter(torch.zeros((dim)), requires_grad=False)
        else:
            self.q_bias = nn.Parameter(torch.zeros((dim)), requires_grad=False) 
            self.k_bias = nn.Parameter(torch.zeros((dim)), requires_grad=False) 
            self.v_bias = nn.Parameter(torch.zeros((dim)), requires_grad=False) 
        
        # Weight standardization parameters
        if self.weight_standardization:
            self.gamma_q = nn.Parameter(torch.ones((1))*gamma, requires_grad=False)
            self.gamma_k = nn.Parameter(torch.ones((1))*gamma, requires_grad=False)
            self.gamma_v = nn.Parameter(torch.ones((1))*gamma, requires_grad=False)
            self.qk_gamma = qk_gamma
        #################### ↑↑↑ Self Attention ↑↑↑ ####################
        
        #################### ↓↓↓ Output Linear ↓↓↓ #####################
        # Take place
        self.proj = nn.Linear(dim, dim)
        self.proj.weight.requires_grad_(False)
        self.proj.bias.requires_grad_(False)
        
        # Trainable parameters
        self.proj_weight = nn.Parameter(torch.rand((dim, dim)))
        self.proj_bias = nn.Parameter(torch.rand((dim)))
        
        # Weight standardization parameters
        if self.weight_standardization:
            self.gamma_proj = nn.Parameter(torch.ones(1)*gamma, requires_grad=False)
            self.proj_gamma = proj_gamma
        #################### ↑↑↑ Output Linear ↑↑↑ #####################
        
        #################### ↓↓↓ Shortcut scale ↓↓↓ ####################
        self.shortcut_type = shortcut_type
        if self.shortcut_type == "PerOperation":
            self.shortcut_gain1 = nn.Parameter(torch.ones((1))*shortcut_gain, requires_grad=False)
            #self.shortcut_gain2 = nn.Parameter(torch.ones((1))*shortcut_gain, requires_grad=False)
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
            self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        elif self.feature_norm == "BatchNorm":
            self.norm = nn.BatchNorm1d(dim, affine=False)
        elif self.feature_norm == "None":
            self.feature_std = nn.Parameter(torch.ones((1))*std, requires_grad=False)
            self.feature_std_accumulation = nn.Parameter(torch.zeros((1)), requires_grad=False)
        ##################### ↑↑↑ Normalization ↑↑↑ ####################
        
    def forward(self, x):
        B, N, C = x.shape
        ######################### ↓↓↓ Standardization ↓↓↓ #########################
        if self.weight_standardization:
            q_weight = standardization(self.q_weight, 
                                       dim_in=self.dim_in,
                                       num_head=self.num_head,
                                       dim_head=self.dim_head) * nn.functional.sigmoid(self.gamma_q) * 0.32
                                          
            k_weight = standardization(self.k_weight,
                                       dim_in=self.dim_in,
                                       num_head=self.num_head,
                                       dim_head=self.dim_head) * nn.functional.sigmoid(self.gamma_k) * 0.32
                                          
            v_weight = standardization(self.v_weight, 
                                       dim_in=self.dim_in,
                                       num_head=self.num_head,
                                       dim_head=self.dim_head) * nn.functional.sigmoid(self.gamma_v) * 0.32
                                          
            proj_weight = standardization(self.proj_weight,
                                          dim_in=self.dim_in,
                                          num_head=self.num_head,
                                          dim_head=self.dim_head) * nn.functional.sigmoid(self.gamma_proj) * 0.32
            
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
                x = x.transpose(-1, -2)
                x = self.norm(x)
                x = x.transpose(-1, -2)
            else:
                self.feature_std_accumulation.data = self.feature_std_accumulation.data + x.std(-1).mean().item()
                x = x / self.feature_std
                
            # Calculate Query (Q), Key (K) and Value (V)
            q = nn.functional.linear(x, q_weight, q_bias) # B, N, C
            k = nn.functional.linear(x, k_weight, k_bias) # B, N, C
            v = nn.functional.linear(x, v_weight, v_bias) # B, N, C
                
            # Reshape Query (Q), Key (K) and Value (V)
            q = rearrange(q, 'b n (nh hc) -> b nh n hc', nh=self.num_head) # B, nh, N, C//nh
            k = rearrange(k, 'b n (nh hc) -> b nh n hc', nh=self.num_head) # B, nh, N, C//nh
            v = rearrange(v, 'b n (nh hc) -> b nh n hc', nh=self.num_head) # B, nh, N, C//nh
                
            # Calculate self-attention
            x = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop) # B, nh, N, C//nh
                
            # Reshape x back to input shape
            x = rearrange(x, 'b nh n hc -> b n (nh hc)', nh=self.num_head) # B, N, C
                
            # Output linear projection
            x = nn.functional.linear(x, proj_weight, proj_bias) # B, N, C 
                
            # Add DropPath and shortcut
            x = self.drop_path(x, None) + shortcut if self.drop_path is not None else x + shortcut # B, N, C
                
        elif self.shortcut_type == "PerOperation":
            # Shortcut
            shortcut = x
                
            # Feature normalization
            if self.feature_norm == "LayerNorm":
                x = self.norm(x)
            elif self.feature_norm == "BatchNorm":
                x = x.transpose(-1, -2)
                x = self.norm(x)
                x = x.transpose(-1, -2)
            elif self.feature_norm == "None":
                #self.feature_std_accumulation.data = self.feature_std_accumulation.data + x.std(-1).mean().item()
                x = x / self.feature_std
            
            # Calculate Query (Q), Key (K) and Value (V)
            q = nn.functional.linear(x, q_weight * self.qk_gamma, q_bias) # B, N, C
            k = nn.functional.linear(x, k_weight, k_bias) # B, N, C
            v = nn.functional.linear(x, v_weight, v_bias) # B, N, C
            
            # Reshape Query (Q), Key (K)
            q = rearrange(q, 'b n (nh hc) -> b nh n hc', nh=self.num_head) # B, nh, N, C//nh
            k = rearrange(k, 'b n (nh hc) -> b nh n hc', nh=self.num_head) # B, nh, C//nh, N
            
            # 1. Calculate self-attention for input feature (AX)
            x_attn = rearrange(x, 'b n (nh hc) -> b nh n hc', nh=self.num_head) # B, nh, N, C//nh
            x_attn = nn.functional.scaled_dot_product_attention(q, k, x_attn) # B, nh, N, C//nh
            x_attn = rearrange(x_attn, 'b nh n hc -> b n (nh hc)', nh=self.num_head) # B, N, C
            
            # 2. Calculate self-attention for Value (AXV)
            v_attn = rearrange(v, 'b n (nh hc) -> b nh n hc', nh=self.num_head) # B, nh, N, C//nh
            v_attn = nn.functional.scaled_dot_product_attention(q, k, v_attn) # B, nh, N, C//nh
            v_attn = rearrange(v_attn, 'b nh n hc -> b n (nh hc)', nh=self.num_head) # B, N, C
            
            # 3. Calculate linear projection for attended Value (AXVO)
            v_out = nn.functional.linear(v_attn, proj_weight * self.proj_gamma, proj_bias) # B, N, C
            
            # 4. Calculate linear projection for Value (XVO)
            x_out = nn.functional.linear(v, proj_weight * self.proj_gamma, proj_bias) # B, N, C
            
            # 5. Calculate output (0.1A+I)X(VO+I)=0.1AXVO+2XVO+0.05AX+X
            x = v_out + \
                x_out * 1 / self.shortcut_gain1 + \
                x_attn * self.shortcut_gain1 + \
                shortcut
            
            # Add DropPath
            x = self.drop_path(x, shortcut) if self.drop_path is not None else x
            
            # Add shortcut
            #x = out #+ shortcut
        ######################### ↑↑↑ Self-attention ↑↑↑ ##########################
        
        
        #if x.get_device() == 0:
            #print("x std:", self.feature_std.item())
            #print("X after mhsa:", shortcut.std(-1).mean().item(), shortcut.mean().item(), shortcut.max().item(), shortcut.min().item())
            #print("AX after mhsa:", x_attn.std(-1).mean().item(), x_attn.mean().item(), x_attn.max().item(), x_attn.min().item())
            #print("XVO after mhsa:", x_out.std(-1).mean().item(), x_out.mean().item(), x_out.max().item(), x_out.min().item())
            #print("AXVO after mhsa:", v_out.std(-1).mean().item(), v_out.mean().item(), v_out.max().item(), v_out.min().item())
            #print("x after mhsa:", x.std(-1).mean().item(), x.mean().item(), x.max().item(), x.min().item())
            #print("Shortcut gain", self.shortcut_gain1.item(), self.shortcut_gain2.item(), self.shortcut_gain3.item())
            #print("mhsa gammas:", self.gamma_q.data, self.gamma_k.data, self.gamma_v.data, self.gamma_proj.data)
            #print("mhsa gammas:", nn.functional.sigmoid(self.gamma_q) * 0.2, nn.functional.sigmoid(self.gamma_k) * 0.2, nn.functional.sigmoid(self.gamma_v) * 0.2, nn.functional.sigmoid(self.gamma_proj) * 0.2)
            #print("v_weight:", v_weight.var(-1).mean(), v_weight.mean(), v_weight.max(), v_weight.min())
            #print("V:", v_weight.std(-1).mean(), v_weight.mean(), v_weight.max(), v_weight.min())
        return x
    
    def adaptive_std(self, steps):
        print(self.feature_std.data, self.feature_std_accumulation.data)
        self.feature_std.data = self.feature_std.data * 0.7 + (self.feature_std_accumulation/steps).data * 0.3
        print(self.feature_std.data)
        self.feature_std_accumulation.data = self.feature_std_accumulation.data * 0
        
    def clean_std(self):
        self.feature_std_accumulation.data = self.feature_std_accumulation.data * 0
        
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
                 affected_layers='None', feature_norm="LayerNorm", shortcut_gain=1.0, gamma=0.1, std=[1.0, 1.0],
                 qk_gamma=1): 
        super().__init__()
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.rep = False
        self.dim = dim
        self.num_head = num_head
                
        assert feature_norm in ["LayerNorm", "BatchNorm", "GroupedLayerNorm"]
        if feature_norm == "GroupedLayerNorm":
            feature_norm = "None"
            
        if affected_layers=="Both":
            self.attn = Attention(dim, num_head=num_head, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                  attn_drop=attn_drop, proj_drop=drop, drop_path=drop_path, 
                                  shortcut_type=shortcut_type, weight_standardization=weight_standardization,
                                  feature_norm=feature_norm, shortcut_gain=shortcut_gain, gamma=gamma, std=std[0], 
                                  qk_gamma=qk_gamma)
            self.mlp = Mlp(dim_in=dim, dim_hidden=mlp_hidden_dim, num_head=num_head, bias=qkv_bias,
                           act_layer=act_layer, drop=drop, drop_path=drop_path, shortcut_type=shortcut_type,
                           weight_standardization=weight_standardization, feature_norm=feature_norm, 
                           shortcut_gain=shortcut_gain, gamma=gamma, std=std[1])
        elif affected_layers=="MHSA":
            self.attn = Attention(dim, num_head=num_head, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                  attn_drop=attn_drop, proj_drop=drop, drop_path=drop_path, 
                                  shortcut_type=shortcut_type, weight_standardization=weight_standardization,
                                  feature_norm=feature_norm, shortcut_gain=shortcut_gain, gamma=gamma, std=std[0], 
                                  qk_gamma=qk_gamma)
            self.mlp = Mlp(dim_in=dim, dim_hidden=mlp_hidden_dim, num_head=num_head, bias=qkv_bias,
                           act_layer=act_layer, drop=drop, drop_path=drop_path)
        elif affected_layers=="FFN":
            self.attn = Attention(dim, num_head=num_head, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                  attn_drop=attn_drop, proj_drop=drop, drop_path=drop_path, shortcut_gain=shortcut_gain,)
            self.mlp = Mlp(dim_in=dim, dim_hidden=mlp_hidden_dim, num_head=num_head, bias=qkv_bias,
                           act_layer=act_layer, drop=drop, drop_path=drop_path, shortcut_tyspe=shortcut_type,
                           weight_standardization=weight_standardization, feature_norm=feature_norm,
                           shortcut_gain=shortcut_gain, gamma=gamma, std=std[1])
        elif affected_layers=="None":
            self.attn = Attention(dim, num_head=num_head, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                  attn_drop=attn_drop, proj_drop=drop, drop_path=drop_path, shortcut_gain=shortcut_gain,)
            self.mlp = Mlp(dim_in=dim, dim_hidden=mlp_hidden_dim, num_head=num_head, bias=qkv_bias,
                           act_layer=act_layer, drop=drop, drop_path=drop_path)
    
    def forward(self, x):
        if not self.rep:
            x = self.attn(x)
            x = self.mlp(x)
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
            block_fn=NFAttentionBlock,
            shortcut_type='PerLayer', # ['PerLayer', 'PerOperation']
            affected_layers='None', # ['None', 'Both', 'MHSA', 'FFN']
            feature_norm='LayerNorm', # ['GroupedLayerNorm', 'LayerNorm', 'BatchNorm']
            weight_standardization=False, # [True, False]
            shortcut_gain=1.0,
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
        std = [x.item() for x in torch.logspace(start=0, end=1, steps=24, base=2)]
        qk_gamma = [1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.11, 0.11, 0.06, 0.06, 0.03, 0.03]
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
                std=std[2*i:2*(i+1)],
                qk_gamma=qk_gamma[i]
            )
            for i in range(depth)])
        
        self.num_head = num_heads
        self.dim_head = embed_dim//self.num_head
        
        self.feature_norm = feature_norm
        self.weight_standardization=weight_standardization
        
        if self.feature_norm == "LayerNorm":
            self.norm_pre = nn.LayerNorm(embed_dim, elementwise_affine=False) if pre_norm else nn.Identity()
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False) if not fc_norm else nn.Identity()
        elif self.feature_norm == "BatchNorm":
            self.norm_pre = nn.BatchNorm1d(embed_dim, affine=False) if pre_norm else nn.Identity()
            self.norm = nn.BatchNorm1d(embed_dim, affine=False) if not fc_norm else nn.Identity()
        elif self.feature_norm == "GroupedLayerNorm":
            self.norm_pre = nn.LayerNorm(self.dim_head, elementwise_affine=False) if pre_norm else nn.Identity()
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False) if not fc_norm else nn.Identity()
        else:
            assert False, "Feature normalization type not supported"
        
        ############################ ↓↓↓ Output Head ↓↓↓ ###########################
        # Take place
        self.head.weight.requires_grad_(False)
        self.head.bias.requires_grad_(False)
        
        # Trainable parameters
        self.head_weight = nn.Parameter(torch.rand((embed_dim, num_classes)))
        self.head_bias = nn.Parameter(torch.zeros((num_classes)))
        ############################ ↑↑↑ Output Head ↑↑↑ ###########################
        
        self._init_standard_weights()
            
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        B, N, C = x.shape
        
        if self.feature_norm == "BatchNorm":
            x = x.transpose(-1, -2)
            x = self.norm_pre(x)
            x = x.transpose(-1, -2)
        elif self.feature_norm == "LayerNorm":
            x = self.norm_pre(x)
        elif self.feature_norm == "GroupedLayerNorm":
            x = x.reshape(B, N, self.num_head, self.dim_head)
            x = self.norm_pre(x)
            x = x.reshape(B, N, C)
        
        for i, blk in enumerate(self.blocks):
            if self.training:
                #x.register_hook(make_print_grad("x of layer "+str(i)))
                x = blk(x) #ckpt.checkpoint(blk, x)
            else:
                x = blk(x)
                
        if self.feature_norm == "BatchNorm":
            x = x.transpose(-1, -2)
            x = self.norm(x)
            x = x.transpose(-1, -2)
        elif self.feature_norm == "LayerNorm":
            x = self.norm(x)
        elif self.feature_norm == "GroupedLayerNorm":
            x = self.norm(x)
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
            if "_weight" in name:
                nn.init.trunc_normal_(param, mean=0.0, std=.02, a=-2, b=2)
            elif "_bias" in name:
                nn.init.zeros_(param)
                
    def adaptive_std(self, steps):
        for blk in self.blocks:
            blk.adaptive_std(steps)
            
    def clean_std(self):
        for blk in self.blocks:
            blk.clean_std()
            
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
def normalization_free_deit_medium_patch16_224_layer12(pretrained=False, pretrained_cfg=None, **kwargs):
    model = NFTransformer(patch_size=16, embed_dim=768, depth=12, pre_norm=True,
                          num_heads=12, mlp_ratio=4, qkv_bias=True, fc_norm=False,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model