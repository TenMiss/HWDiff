import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
from einops import rearrange


def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)
    # return x_LL+ x_HL + x_LH + x_HH

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class CrossAttention(nn.Module):
    def __init__(self, dim, q_channel, num_heads =8, bias = False):
        super(CrossAttention, self).__init__()
        self.dwt = DWT()
        self.idwt = IWT()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.q = nn.Conv2d(q_channel, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
               
    def forward(self, x, query):
        b, c, h, w = x.shape
        kv = self.kv(x)
        k, v = kv.chunk(2, dim=1)
        q = self.q(query)  

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)


        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)


        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)


        out = self.project_out(out)

        return out


##########################################################################
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, 2 * (i // 2) / np.float32(d_model))
    return pos * angle_rates



def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[:, np.newaxis, :]
    return torch.tensor(pos_encoding, dtype=torch.float32)

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def nonlinearity(x):
    return x*torch.sigmoid(x)

class DTB(nn.Module):

    def __init__(self, dim, num_heads, ffn_factor, bias, LayerNorm_type):
        super(DTB, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_factor, bias)

    def forward(self, x, t):
        if len(t.size()) == 1:
            t = nonlinearity(get_timestep_embedding(t, x.size(1)))[:, :, None, None]

        x = x + self.attn(self.norm1(x) + t)
        x = x + self.ffn(self.norm2(x) + t)

        return x, t


##########################################################################
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


# handle multiple input 处理多个input
class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
    

# HF_guided_CA
class HF_guided_CA(nn.Module):
    def __init__(self, in_channel, q_channel, norm_groups=32):
        super().__init__()

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        # self.norm = LayerNorm(in_channel, LayerNorm_type= 'WithBias')
        self.q = nn.Conv2d(q_channel, q_channel, 1, bias=False)
        self.kv = nn.Conv2d(in_channel, q_channel * 2, 1, bias=False)
        self.out = nn.Conv2d(q_channel, in_channel, 1)

    def forward(self, input, quary):
        batch, channel, height, width = quary.shape
        head_dim = channel

        norm = self.norm(input)

        kv = self.kv(norm).view(batch, 1, head_dim * 2, height, width)
        key, value = kv.chunk(2, dim=2)  # bhdyx
        quary = self.q(quary).unsqueeze(1)

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", quary, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, 1, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, 1, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        # print(out.shape)
        out = self.out(out.view(batch, channel, height, width))

        return out + input



#####################################Diffusion Transformer DFT################################
class DFT(nn.Module):
    def __init__(self,
                 inp_channels=6,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 heads=[1, 2, 4, 8],
                 ffn_factor = 4.0,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False
                 ):

        super(DFT, self).__init__()

        self.dwt = DWT()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = MySequential(*[
            DTB(dim=dim, num_heads=heads[0], ffn_factor=ffn_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.dwon1_2_atten = CrossAttention(96, 12)
        self.encoder_level2 = MySequential(*[
            DTB(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.dwon2_3_atten = CrossAttention(192, 48)
        self.encoder_level3 = MySequential(*[
            DTB(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.dwon3_4_atten = CrossAttention(384, 192)
        self.latent = MySequential(*[
            DTB(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.up4_3_atten = CrossAttention(384, 192)
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = MySequential(*[
            DTB(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.up3_2_atten = CrossAttention(192, 48)
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = MySequential(*[
            DTB(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.up2_1_atten = CrossAttention(96, 12)
        self.decoder_level1 = MySequential(*[
            DTB(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, high_freq, t):
        # print(inp_img.shape, high_freq.shape)
        # 1,6,128,128
        inp_enc_level1 = self.patch_embed(inp_img) # 1,48,128,128
        out_enc_level1,_ = self.encoder_level1(inp_enc_level1, t) # 1,48,128,128

        inp_enc_level2 = self.down1_2(out_enc_level1) # 1,96,64,64
        
        # atten down1_2
        multi_scale_hf = []
        hf = self.dwt(high_freq)
        multi_scale_hf.append(hf)

        inp_enc_level2 = self.dwon1_2_atten(inp_enc_level2, hf)

        out_enc_level2,_ = self.encoder_level2(inp_enc_level2, t) # 1,96,64,64

        inp_enc_level3 = self.down2_3(out_enc_level2) # 1,192,32,32

        # atten down2_3
        hf = self.dwt(hf)
        multi_scale_hf.append(hf)
        inp_enc_level3 = self.dwon2_3_atten(inp_enc_level3, hf)

        out_enc_level3,_ = self.encoder_level3(inp_enc_level3, t) # 1,192,32,32

        inp_enc_level4 = self.down3_4(out_enc_level3) # 1,384,16,16

        # atten down3_4
        hf = self.dwt(hf)
        multi_scale_hf.append(hf)
        inp_enc_level4 = self.dwon3_4_atten(inp_enc_level4, hf)

        latent,_ = self.latent(inp_enc_level4, t) # 1,384,16,16


        # atten up4_3
        latent = self.up4_3_atten(latent, multi_scale_hf.pop())
        inp_dec_level3 = self.up4_3(latent) # 1,192,32,32

        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)# 1,192,32,32
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3) # 1,192,32,32
        out_dec_level3,_ = self.decoder_level3(inp_dec_level3, t) # 1,192,32,32

        # atten up3_2
        out_dec_level3 = self.up3_2_atten(out_dec_level3, multi_scale_hf.pop())
        inp_dec_level2 = self.up3_2(out_dec_level3) # 1,96,64,64

        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)# 1,96,64,64
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2) # 1,96,64,64
        out_dec_level2,_ = self.decoder_level2(inp_dec_level2, t) # 1,96,64,64

        # atten up2_1
        out_dec_level2 = self.up2_1_atten(out_dec_level2, multi_scale_hf.pop())
        inp_dec_level1 = self.up2_1(out_dec_level2) # 1,48,128,128

        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1) # 1,96,128,128
        out_dec_level1,_ = self.decoder_level1(inp_dec_level1, t) # 1,96,128,128

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img[:,-3:,:,:]

        return out_dec_level1 # 1,3,128,128


# def main():
#
#     x = torch.rand(1, 6, 32, 32)
#     t = torch.tensor([1])
#     print(x.shape)
#     model = DFT()
#     # print(model)
#     y = model(x, t)
#     print(y.shape)
#
#
#
if __name__ == '__main__':
    x = torch.rand(1, 6, 128, 128)
    high_freq = torch.rand(1,3,128,128)
    t = torch.tensor([1])

    print(x.shape)
    model = DFT()
    # print(model)
    y = model(x, high_freq, t)
    print(y.shape)
