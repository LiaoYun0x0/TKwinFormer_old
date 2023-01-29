import copy
import torch
import torch.nn as nn
from .linear_attention import LinearAttention, FullAttention, RPEAttention
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        num_dim = len(x.shape) 
        if num_dim == 4:
            c,h,w = x.shape[1:]
            x = rearrange(x,'b c h w -> b (h w) c')
            source = rearrange(source,'b c h w -> b (h w) c')
            
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)
        
        x = x + message
        if num_dim == 4:
            x = rearrange(x,'b (h w) c -> b c h w',h=h,w=w)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.dwconv(x, H, W)
        
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x
    
class MaxViTEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(MaxViTEncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        # self.mlp = nn.Sequential(
        #     nn.Linear(d_model*2, d_model*2, bias=False),
        #     nn.ReLU(True),
        #     nn.Linear(d_model*2, d_model, bias=False),
        # )
        self.mlp = Mlp(in_features=d_model*2,hidden_features=d_model*2,out_features=d_model)

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, h,w):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        num_dim = len(x.shape) 
        if num_dim == 4:
            x = rearrange(x,'b c h w -> b (h w) c')
            source = rearrange(source,'b c h w -> b (h w) c')
            
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=None, kv_mask=None)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2),h,w)
        message = self.norm2(message)
        
        x = x + message
        if num_dim == 4:
            x = rearrange(x,'b (h w) c -> b c h w',h=h,w=w)
        return x
        
        
    
class TopKWindowAttentionLayer(nn.Module):
    def __init__(self,d_model,nhead,attention,w=7,k=8):
        super(TopKWindowAttentionLayer, self).__init__()
        self.w = w
        self.k = k
        self.dim = d_model // nhead
        self.nhead = nhead
        # multi-head attention
        self.q_proj = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1,bias=False)
        self.k_proj = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1,bias=False)
        self.v_proj = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1,bias=False)
        
        if attention == "linear":
            self.attention = LinearAttention()
        elif attention == "full":
            self.attention = FullAttention()
        elif attention == "rpe":
            self.attention = RPEAttention(d_model)
        else:
            raise NotImplementedError()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # self.mlp = Mlp(in_features=d_model*2,hidden_features=d_model*2,out_features=d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source):
        bs,d,h,w = x.shape
            
        query, key, value = x, source, source
        # TopK-Window-multihead-Attention
        queries = self.q_proj(query)
        keys = self.k_proj(key)
        values = self.v_proj(value)
        queries = rearrange(queries,'b d (m w1) (n w2) -> b m n w1 w2 d',w1=self.w,w2=self.w)
        _,m,n,w1,w2,_ = queries.shape
        queries = rearrange(queries,'b m n w1 w2 d -> b (m n) (w1 w2) d')
        keys = rearrange(keys, 'b d (m w1) (n w2) -> b (m n) (w1 w2) d',w1=self.w,w2=self.w)
        values = rearrange(values, 'b d (m w1) (n w2) -> b (m n) (w1 w2) d',w1=self.w,w2=self.w)
        query_mean = torch.mean(queries,dim=2)
        key_mean = torch.mean(keys,dim=2)
        value_mean = torch.mean(values,dim=2)
          
        window_similarity = torch.einsum('bmd,bnd->bmn',query_mean,key_mean)
        topk_values,topk_indices = torch.topk(window_similarity,dim=-1,k=self.k)
        
        fine_keys = []
        fine_values = []
        for i in range(bs):
            fine_keys.append(keys[i][topk_indices[i]])
            fine_values.append(values[i][topk_indices[i]])
            
        fine_keys = torch.stack(fine_keys).reshape(bs,m*n,-1,d) # [B, m*n, k*w1*w2, D]
        fine_values = torch.stack(fine_values).reshape(bs,m*n,-1,d)
        
        keys = torch.cat([fine_keys,torch.tile(key_mean.unsqueeze(1),(1,m*n,1,1))],2)
        values = torch.cat([fine_values,torch.tile(value_mean.unsqueeze(1),(1,m*n,1,1))],2)
        
        queries = rearrange(queries,'b nw ws (h d) -> (b nw) ws h d',h=self.nhead)
        keys = rearrange(keys,'b nw ws (h d) -> (b nw) ws h d',h=self.nhead)
        values = rearrange(values,'b nw ws (h d) -> (b nw) ws h d',h=self.nhead)
        
        message = self.attention(queries, keys, values, q_mask=None, kv_mask=None)  # [N, L, (H, D)]
        message = rearrange(message,'(b m n) (w1 w2) h d -> b (m w1 n w2) (h d)',b=bs,m=m,n=n,w1=self.w,w2=self.w)
        message = self.norm1(self.merge(message))
        
        
        # feed-forward network
        x = rearrange(x,'b d h w -> b (h w) d')
        message = self.mlp(torch.cat([x, message], dim=2),h,w)
        message = self.norm2(message)
        
        x = x + message
        x = rearrange(x,'b (h w) d -> b d h w',h=h,w=w)
        return x
   
class PositionEmbedding(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.dwconv = nn.Conv2d(d_model, d_model, 3, 1, 1, groups=d_model)
    def forward(self,x):
        x = x + self.dwconv(x)
        return x
        

class TopKWindowAttentionLayerV2(nn.Module):
    def __init__(self,d_model,nhead,attention,w=7,k=8):
        super(TopKWindowAttentionLayerV2, self).__init__()
        self.w = w
        self.k = k
        self.dim = d_model // nhead
        self.nhead = nhead
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        
        if attention == "linear":
            self.attention = LinearAttention()
        elif attention == "full":
            self.attention = FullAttention()
        elif attention == "rpe":
            self.attention = RPEAttention(d_model)
        else:
            raise NotImplementedError()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        self.mlp = Mlp(in_features=2*d_model,hidden_features=2*d_model,out_features=d_model)
        """
        self.mlp = nn.Sequential(
             nn.Linear(2*d_model, 2*d_model, bias=False),
             nn.ReLU(True),
             nn.Linear(2*d_model, d_model, bias=False),
        )
        """

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source):
        bs,d,h,w = x.shape
        m = h // self.w
        n = w // self.w
        x = rearrange(x,'b d h w -> b (h w) d')
        source = rearrange(source,'b d h w -> b (h w) d')
        
        queries, keys, values = x, source, source
        # TopK-Window-multihead-Attention
        queries = self.q_proj(queries)
        keys = self.k_proj(keys)
        values = self.v_proj(values)
        queries = rearrange(queries,'b (m w1 n w2) d -> b (m n) (w1 w2) d',m=m,w1=self.w,n=n,w2=self.w)
        keys = rearrange(keys,'b (m w1 n w2) d -> b (m n) (w1 w2) d',m=m,w1=self.w,n=n,w2=self.w)
        values = rearrange(values,'b (m w1 n w2) d -> b (m n) (w1 w2) d',m=m,w1=self.w,n=n,w2=self.w)
        query_mean = torch.mean(queries,dim=2)
        key_mean = torch.mean(keys,dim=2)
        value_mean = torch.mean(values,dim=2)
          
        window_similarity = torch.einsum('bmd,bnd->bmn',query_mean,key_mean)
        topk_values,topk_indices = torch.topk(window_similarity,dim=-1,k=self.k)
        
        fine_keys = []
        fine_values = []
        for i in range(bs):
            fine_keys.append(keys[i][topk_indices[i]])
            fine_values.append(values[i][topk_indices[i]])
            
        fine_keys = torch.stack(fine_keys).reshape(bs,m*n,-1,d) # [B, m*n, k*w1*w2, D]
        fine_values = torch.stack(fine_values).reshape(bs,m*n,-1,d)
        
        keys = torch.cat([fine_keys,torch.tile(key_mean.unsqueeze(1),(1,m*n,1,1))],2)
        values = torch.cat([fine_values,torch.tile(value_mean.unsqueeze(1),(1,m*n,1,1))],2)
        
        queries = rearrange(queries,'b nw ws (h d) -> (b nw) ws h d',h=self.nhead)
        keys = rearrange(keys,'b nw ws (h d) -> (b nw) ws h d',h=self.nhead)
        values = rearrange(values,'b nw ws (h d) -> (b nw) ws h d',h=self.nhead)
        message = self.attention(queries, keys, values, q_mask=None, kv_mask=None)  # [N, L, (H, D)]
        message = torch.cat([x,self.norm1(self.merge(message.reshape(bs,-1,d)))],dim=2)

        # feed-forward
        x = self.norm2(self.mlp(message,h,w)) + x
        #x = self.norm2(self.mlp(message)) + x
        x = rearrange(x,'b (h w) d -> b d h w',h=h,w=w)
        return x
        



class WindowGridAttentionLayer(nn.Module):
    def __init__(self,d_model,nhead,attention,w=7):
        super().__init__()
        # self.atten = LoFTREncoderLayer(d_model, nhead,attention=attention)
        self.atten = MaxViTEncoderLayer(d_model, nhead,attention=attention)
        self.w = w
    
    # def forward(self, x, source, x_mask=None, source_mask=None):
    #     # window_attention
    #     h,w = x.shape[2:]
    #     x = rearrange(x,'b d (m w1) (n w2) -> b m n w1 w2 d',w1=self.w, w2=self.w)
    #     b,m,n,w1,w2,d = x.shape
    #     x = rearrange(x,'b m n w1 w2 d -> (b m n) (w1 w2) d')
    #     x_mask = rearrange(x_mask,'b (m w1 n w2) -> (b m n) (w1 w2)',m=m,n=n,w1=self.w,w2=self.w)
    #     x = self.atten(x,x,x_mask,x_mask)
    #     x = rearrange(x,'(b m n) (w1 w2) d -> b d (m w1) (n w2)',m=m,n=n,w1=self.w,w2=self.w)
    #     x_mask = rearrange(x_mask,'(b m n) (w1 w2) -> b (m w1 n w2)',m=m,n=n,w1=self.w,w2=self.w)
        
    #     # grid_attention
    #     x = rearrange(x,'b d (w1 m) (w2 n) -> (b m n) (w1 w2) d',w1=self.w, w2=self.w)
    #     x_mask = rearrange(x_mask,'b (w1 m w2 n) -> (b m n) (w1 w2)',m=m,n=n,w1=self.w,w2=self.w)
    #     x = self.atten(x,x,x_mask,x_mask)
    #     x = rearrange(x,'(b m n) (w1 w2) d -> b d (w1 m) (w2 n)',m=m,n=n,w1=self.w,w2=self.w)
    #     return x
    
    def forward(self, x):
        # window_attention
        h,w = x.shape[2:]
        x = rearrange(x,'b d (m w1) (n w2) -> b m n w1 w2 d',w1=self.w, w2=self.w)
        b,m,n,w1,w2,d = x.shape
        x = rearrange(x,'b m n w1 w2 d -> (b m n) (w1 w2) d')
        x = self.atten(x,x,self.w,self.w)
        x = rearrange(x,'(b m n) (w1 w2) d -> b d (m w1) (n w2)',m=m,n=n,w1=self.w,w2=self.w)
        
        # grid_attention
        x = rearrange(x,'b d (w1 m) (w2 n) -> (b m n) (w1 w2) d',w1=self.w, w2=self.w)
        x = self.atten(x,x,self.w,self.w)
        x = rearrange(x,'(b m n) (w1 w2) d -> b d (w1 m) (w2 n)',m=m,n=n,w1=self.w,w2=self.w)
        return x
        
        
        

class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        # encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        # self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self.layers = nn.ModuleList()
        for layer_name in self.layer_names:
            if layer_name in ['self', 'cross']:
                self.layers.append(LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention']))
            elif layer_name == "wgself":
                self.layers.append(WindowGridAttentionLayer(config['d_model'], config['nhead'], config['attention']))
            elif layer_name in ["tkwself2","tkwcross2"]:
                self.layers.append(TopKWindowAttentionLayerV2(config['d_model'], config['nhead'], config['attention']))
            elif layer_name in ["tkwself","tkwcross"]:
                self.layers.append(TopKWindowAttentionLayer(config['d_model'], config['nhead'], config['attention']))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        # assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"
        if len(feat0.shape) == 3:
            assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"
        elif len(feat0.shape) == 4:
            assert self.d_model == feat0.size(1), "the feature number of src and transformer must be equal"
        else:
            raise NotImplementedError()

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'wgself':
                feat0 = layer(feat0)
                feat1 = layer(feat1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            elif name == 'tkwself':
                feat0 = layer(feat0,feat0)
                feat1 = layer(feat1,feat1)
            elif name == 'tkwcross':
                feat0,feat1 = layer(feat0,feat1),layer(feat1,feat0)
            elif name == 'tkwself2':
                feat0 = layer(feat0,feat0)
                feat1 = layer(feat1,feat1)
            elif name == 'tkwcross2':
                feat0,feat1 = layer(feat0,feat1),layer(feat1,feat0)
            else:
                raise KeyError

        return feat0, feat1


if __name__ == "__main__":
    config = {
        "d_model":128,
        "nhead":8,
        "layer_names":["tkwself","tkwcross"]*4,
        "attention":"linear"
    }
    model = LocalFeatureTransformer(config)
    x = torch.rand(2,128,49,49)
    y0,y1 = model(x,x)
    print(y0.shape,y1.shape)
