from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# helper classes

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# MBConv

class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

class Dropsample(nn.Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

# def MBConv(
#     dim_in,
#     dim_out,
#     *,
#     downsample,
#     expansion_rate = 4,
#     shrinkage_rate = 0.25,
#     dropout = 0.
# ):
#     hidden_dim = int(expansion_rate * dim_out)
#     stride = 2 if downsample else 1

#     net = nn.Sequential(
#         nn.Conv2d(dim_in, dim_out, 1),
#         nn.BatchNorm2d(dim_out),
#         nn.SiLU(),
#         nn.Conv2d(dim_out, dim_out, 3, stride = stride, padding = 1, groups = dim_out),
#         SqueezeExcitation(dim_out, shrinkage_rate = shrinkage_rate),
#         nn.Conv2d(dim_out, dim_out, 1),
#         nn.BatchNorm2d(dim_out)
#     )

#     if dim_in == dim_out and not downsample:
#         net = MBConvResidual(net, dropout = dropout)

#     return net
class MBConv(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        *,
        downsample,
        expansion_rate = 4,
        shrinkage_rate = 0.25,
        dropout = 0.
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.downsample = downsample
        hidden_dim = int(expansion_rate * dim_out)
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(dim_in, dim_out, 1)
        self.norm1 = nn.LayerNorm(dim_out)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, stride = stride, padding = 1, groups = dim_out)
        self.se = SqueezeExcitation(dim_out, shrinkage_rate = shrinkage_rate)
        self.conv3 = nn.Conv2d(dim_out, dim_out, 1)
        self.norm2 = nn.LayerNorm(dim_out)

    def forward(self,x):
        _x = self.conv1(x)
        _x = self.norm1(_x.permute(0,2,3,1)).permute(0,3,1,2)
        _x = self.act1(_x)
        _x = self.conv2(_x)
        _x = self.se(_x)
        _x = self.conv3(_x)
        _x = self.norm2(_x.permute(0,2,3,1)).permute(0,3,1,2)
        if self.dim_in == self.dim_out:
            _x = x+_x
        return _x

# attention related classes

class Positional(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))
    
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768, with_pos=True):
        super().__init__()
        patch_size = (patch_size,patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = Positional(embed_dim)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        if self.with_pos:
            x = self.pos(x)
        
        x = x.permute(0,2,3,1)
        x = self.norm(x)
        x = x.permute(0,3,1,2)
        return x
    
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        # relative positional bias

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = torch.arange(window_size)
        # grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = torch.stack(torch.meshgrid(pos, pos))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        # flatten

        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h = h), (q, k, v))

        # scale

        q = q * self.scale

        # sim

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias

        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)

        # combine heads out

        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)

class MaxViT(nn.Module):
    def __init__(
        self,
        *,
        dims=[128,192,256,320],
        depths,
        dim_head = 32,
        dim_conv_stem = None,
        window_size = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        in_chans = 1
    ):
        super().__init__()
        assert isinstance(depths, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'

        # convolutional stem

        dim_conv_stem = default(dim_conv_stem, 64)

        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_chans, dim_conv_stem, 7, stride = 2, padding = 3),
            nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding = 1)
        )

        # variables

        num_stages = len(depths)

        # dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])
        self.out_norms = nn.ModuleList()
        # shorthand for window size for efficient block - grid like attention

        w = window_size

        # iterate through stages

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depths)):
            self.out_norms.append(nn.LayerNorm(layer_dim))
            self.layers.append(nn.ModuleList())
            layers = self.layers[ind]
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                block = nn.Sequential(
                    MBConv(
                        stage_dim_in,
                        layer_dim,
                        downsample = (is_first and ind!=0),
                        expansion_rate = mbconv_expansion_rate,
                        shrinkage_rate = mbconv_shrinkage_rate
                    ),
                    # PatchEmbed(
                    #     patch_size=3,
                    #     stride=2 if (is_first and ind!=0) else 1,
                    #     in_chans=stage_dim_in,
                    #     embed_dim=layer_dim,
                    #     with_pos=True
                    # ),
                    Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w),  # block-like attention
                    PreNormResidual(layer_dim, Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
                    PreNormResidual(layer_dim, FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),

                    Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w),  # grid-like attention
                    PreNormResidual(layer_dim, Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
                    PreNormResidual(layer_dim, FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
                )
                layers.append(block)

    def forward(self, x):
        x = self.conv_stem(x)

        fs = []
        for i,stage in enumerate(self.layers):
            for _stage in stage:
                x = _stage(x)
            _x = x.permute(0,2,3,1)
            _x = self.out_norms[i](_x)
            _x = _x.permute(0,3,1,2)
            fs.append(_x)
        return fs

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class MaxViTFPN(nn.Module):
    def __init__(self,config):
        super(MaxViTFPN,self).__init__()
        self.backbone = MaxViT(**config)
        embed_dims = config["dims"]
        self.layer4_outconv = conv1x1(embed_dims[3], embed_dims[3])
        self.layer3_outconv = conv1x1(embed_dims[2], embed_dims[3])
        self.layer3_outconv2 = nn.Sequential(
            conv3x3(embed_dims[3], embed_dims[3]),
            nn.BatchNorm2d(embed_dims[3]),
            nn.LeakyReLU(),
            conv3x3(embed_dims[3], embed_dims[2]),
        )

        self.layer2_outconv = conv1x1(embed_dims[1], embed_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(embed_dims[2], embed_dims[2]),
            nn.BatchNorm2d(embed_dims[2]),
            nn.LeakyReLU(),
            conv3x3(embed_dims[2], embed_dims[1]),
        )
        self.layer1_outconv = conv1x1(embed_dims[0], embed_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(embed_dims[1], embed_dims[1]),
            nn.BatchNorm2d(embed_dims[1]),
            nn.LeakyReLU(),
            conv3x3(embed_dims[1], embed_dims[0]),
        )
    
    def forward(self,x):
        out1,out2,out3,out4 = self.backbone(x)

        c4_out = self.layer4_outconv(out4)
        _,_,H,W = out3.shape
        c4_out_2x = F.interpolate(c4_out, size =(H,W), mode='bilinear', align_corners=True)
        c3_out = self.layer3_outconv(out3)
        _,_,H,W = out2.shape
        c3_out = self.layer3_outconv2(c3_out +c4_out_2x)
        c3_out_2x = F.interpolate(c3_out, size =(H,W), mode='bilinear', align_corners=True)
        c2_out = self.layer2_outconv(out2)
        _,_,H,W = out1.shape
        c2_out = self.layer2_outconv2(c2_out +c3_out_2x)
        c2_out_2x = F.interpolate(c2_out, size =(H,W), mode='bilinear', align_corners=True)
        c1_out = self.layer1_outconv(out1)
        c1_out = self.layer1_outconv2(c1_out+c2_out_2x)
        return c3_out, c1_out



if __name__ == "__main__":
    config = {
        "dims":[128,192,256,320],
        "depths":(1,1,2,1)
    }
    model = MaxViTFPN(config)
    torch.save(model.state_dict(),"maxvitfpn_v2.pth")
    x = torch.rand(1,1,672,672)
    ys = model(x)
    for y in ys:
        print(y.shape)