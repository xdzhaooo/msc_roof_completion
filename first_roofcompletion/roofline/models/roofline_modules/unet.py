from abc import abstractmethod
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import (
    checkpoint,
    zero_module,
    normalization,
    count_flops_attn,
    gamma_embedding
)

from einops import rearrange, repeat
from .attention import BasicTransformerBlock

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class EmbedBlock(nn.Module):
    """
    Any module where forward() takes embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` embeddings.
        """

class EmbedSequential(nn.Sequential, EmbedBlock):
    """
    A sequential module that passes embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb,cond2=None):
        for layer in self:
            if isinstance(layer, EmbedBlock):  # 如果是EmbedBlock，将emb传入
                x = layer(x, emb)
            elif layer.__class__.__name__ == "AttentionBlock_trans":  # 如果是AttentionBlock_self_crosss，将cond2传入
                x = layer(x, cond2)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module): #先放大两倍，再进行3x3卷积
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.

    """

    def __init__(self, channels, use_conv, out_channel=None):
        super().__init__()
        self.channels = channels
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channel, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest") #
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):   #如果use_conv, 用stride=2的3x3卷积缩小两倍，否则用stride=2的2x2平均池化缩小两倍
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv, out_channel=None):
        super().__init__()
        self.channels = channels
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        stride = 2
        if use_conv:
            self.op = nn.Conv2d(
                self.channels, self.out_channel, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channel
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)
            # self.op = nn.MaxPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(EmbedBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of embedding channels.
    :param dropout: the rate of dropout.
    :param out_channel: if specified, the number of out channels.
    :param use_conv: if True and out_channel is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels, #嵌入通道数，用于条件嵌入
        dropout,
        dilation=1,
        out_channel=None, #输出通道数,默认为输入
        use_conv=False,  #是否使用卷积，如果为true且指定了out_channel，使用空间卷积而不是较小的1x1卷积来改变跳过连接中的通道。因为out_channel改变了通道数，所以需要使用卷积来改变通道数
        use_scale_shift_norm=False,
        use_checkpoint=False, #是否使用梯度检查点在这个模块上
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        padding = 2 ** (dilation - 1) #膨胀率，不膨胀，padding=1

        self.in_layers = nn.Sequential(#输入层
            normalization(channels),   #这里是groupnorm32
            SiLU(),
            nn.Conv2d(channels, self.out_channel, 3, padding=padding, dilation=dilation),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False)   
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)  #用平均池化缩小两倍
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity() #恒等映射，不做任何操作

        self.emb_layers = nn.Sequential(   #嵌入层
            SiLU(),
            nn.Linear(                     #全连接层，输入为emb_channels，输出为2*out_channel如果用scale_shift_norm，否则输出为out_channel
                emb_channels,
                2 * self.out_channel if use_scale_shift_norm else self.out_channel,
            ),
        )
        self.out_layers = nn.Sequential(           #输出层
            normalization(self.out_channel),       #groupnorm32
            SiLU(),
            nn.Dropout(p=dropout),                 
            zero_module(                           #初始时将模块的参数置零，输出也为0.decoder中某一层接受F(x)+y,y来自上一层，F(x)为resblock的输出，所以初始时F(x)为0k可以在训练初始阶段更接近与y，减少残差影响，避免残差随机初始化造成影响。
                nn.Conv2d(self.out_channel, self.out_channel, 3, dilation=dilation, padding=padding)
            ),
        )

        if self.out_channel == channels:          
            self.skip_connection = nn.Identity() 
        elif use_conv:
            self.skip_connection = nn.Conv2d(     #如果out_channel改变了通道数，使用3x3卷积来改变通道数，并且不改变hw
                channels, self.out_channel, 3, padding=padding, dilation=dilation
            )
        else:                                     #如果out_channel改变了通道数，但未指明使用卷积，使用1x1卷积来改变通道数，这样输出的hw和输入的hw一样
            self.skip_connection = nn.Conv2d(channels, self.out_channel, 1, dilation=dilation)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:                                                    #如果有上采样或下采样
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]     
            h = in_rest(x)                                                 #x经过groupnorm32和SiLU（输入层的前两层）
            h = self.h_upd(h)                                              #上面结果，再经过上采样或下采样   
            x = self.x_upd(x)                                              #x也经过上采样或下采样       
            h = in_conv(h)                                                 #h再经过卷积层，至此h从输入层输出，且经过了上采样或下采样，x也经过了上采样或下采样     
        else:
            h = self.in_layers(x)                                          #如果没有上采样或下采样，直接经过输入层
        emb_out = self.emb_layers(emb).type(h.dtype)                       #将emb经过嵌入层，输出的类型和h的类型一样
        while len(emb_out.shape) < len(h.shape):                           #如果emb_out的维度小于h的维度，不断扩展emb_out的维度直到和h的维度一样 emb_out[..., None] 后，其形状会变为 [N, emb_channels, 1]
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:                                      #如果使用scale_shift_norm，即FiLM-like的条件机制公式为h = norm(h) * (1 + scale) + shift
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]   #将输出层分为norm和rest两部分
            scale, shift = torch.chunk(emb_out, 2, dim=1)                  #沿着第二个维度，也就是2*out_channel的维度，将emb_out分为两部分，scale和shift
            h = out_norm(h) * (1 + scale) + shift                          #调制
            h = out_rest(h)                                                #再经过rest                                   
        else:                                                              #如果不使用scale_shift_norm，直接将emb_out加到h上，相加嵌入
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1, #每个注意力层的注意力头数
        num_head_channels=-1, #每个注意力头的通道数
        use_checkpoint=False,
        use_new_attention_order=False,   #是否使用新的注意力顺序
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:   #如果没有指定num_head_channels，直接使用num_heads
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}" 
            self.num_heads = channels // num_head_channels 
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels) #groupnorm32
        self.qkv = nn.Conv1d(channels, channels * 3, 1) #1x1卷积，输入通道数为channels，输出通道数为3*channels。相当于xwq,xwk,xwv
        if use_new_attention_order:      
            # split qkv before split heads，如果使用新的注意力顺序，先再分割qkv
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv，否则先分割qkv再分割头。这两种方式区别不大
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1)) #多头注意力中，简单concate后再通过一个1x1卷积，可以融合多头注意力的结果

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape   #b是batch_size，c是通道数，spatial是空间维度
        x = x.reshape(b, c, -1)    #将空间维度合并,b，c,h*w
        qkv = self.qkv(self.norm(x))  #先经过groupnorm32，再经过1x1卷积，得到qkv。b，3*c,h*w
        h = self.attention(qkv)       
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial) #残差连接,最后再reshape回原来的形状




class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads #注意力头数

    def forward(self, qkv):
        """
        传统的QKV注意力，先reshape输入，再分割成qkv
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs. #N是batch_size，H是注意力头数，C是每个头的通道数，T是序列长度
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads) #每个头的通道数
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1) #先reshape成N*H, 3*c, T, 然后在第二个维度上分割，分割成q,k,v.每个的形状为N*H, c, T
        scale = 1 / math.sqrt(math.sqrt(ch)) #缩放因子,作用是为了使得点积结果更稳定
        weight = torch.einsum(               #einsum是张量的乘法，这里是点积。bct 乘 bcs，得到bts。这里是计算q和k的点积，形状为N*H, T, T
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards   这种方式比先除再计算更稳定在f16下
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype) #在最后一个维度上进行softmax，得到注意力权重 。这里是计算注意力权重，形状为N*H, T, T
        a = torch.einsum("bts,bcs->bct", weight, v)    #计算注意力权重和v的点积，得到输出。N*H，C，T点成N*H，T，T，得到N*，C，T
        return a.reshape(bs, -1, length)               #N，H*C，T

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads  #注意力头数

    def forward(self, qkv):
        """
        先分割成三块qkv，再reshape计算注意力权重
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)    #在第channel维度上分割，分割成q,k,v。每个的形状为N, H*C, T
        scale = 1 / math.sqrt(math.sqrt(ch)) #缩放因子
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length), #先将q，k的形状，然后reshape成N*H, c, T
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)    #计算注意力权重
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))    #权重形状 N*H, T, T，v形状N*H, C, T
        return a.reshape(bs, -1, length) #N, H*C, T

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)
    

class CrossAttention(nn.Module):
    """
    实现交叉注意力：
    query: [N, H*C, T_query]
    kv: [N, 2*H*C, T_key]，其中前半部分为键，后半部分为值
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, q, kv):
        bs, width_q, T_q = q.shape
        bs, width_kv, T_k = kv.shape

        ch = width_q // self.n_heads
        # 检查 kv 的通道数是否符合预期：2 * n_heads * ch
        assert width_kv == 2 * self.n_heads * ch, (
            f"kv 的通道数 {width_kv} 不等于 2 * n_heads * ch = {2 * self.n_heads * ch}"
        )
        # 将 kv 分割成键和值，形状均为 [N, H*C, T_k]
        k, v = kv.chunk(2, dim=1)

        scale = 1 / math.sqrt(math.sqrt(ch))
        # 先缩放，再 reshape 成 [N * n_heads, ch, T]
        q = (q * scale).view(bs * self.n_heads, ch, T_q)
        k = (k * scale).view(bs * self.n_heads, ch, T_k)

        # 计算注意力权重: 对每个 head 计算 (ch, T_q) 与 (ch, T_k) 的点乘
        weight = torch.einsum("bct,bcs->bts", q, k)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        # 处理值 v，reshape 成 [N * n_heads, ch, T_k]
        v = v.reshape(bs * self.n_heads, ch, T_k)
        # 根据注意力权重计算输出
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, T_q)

    @staticmethod
    def count_flops(model, q, kv):
        # 这里假设有辅助函数来统计 FLOPs
        return count_flops_attn(model, q, kv)


class AttentionBlock_self_crosss(nn.Module):
    """
    一个注意力模块：
    1. 首先对输入 x（包含噪声图片和图片条件1 concat 后的结果）进行自注意力处理；
    2. 然后利用额外的图片条件2（cond2）进行交叉注意力，
       将自注意力的结果作为 query，cond2作为键和值。
    
    cond2 的空间尺寸可以与 x 不同；如果 cond2 的通道数不同，也会先通过 1x1 卷积投影到 self.channels。
    """
    def __init__(
        self,
        channels,
        num_heads=1,         # 每个注意力层的注意力头数
        num_head_channels=-1,  # 每个注意力头的通道数
        cond2_channels=1,   # 如果 cond2 的通道数和 x 不同，则传入 cond2 的通道数，否则默认为 channels
        use_checkpoint=False,
        use_new_attention_order=False,  # 是否使用新的注意力顺序
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, (
                f"q,k,v channels {channels} 不能被 num_head_channels {num_head_channels} 整除"
            )
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint

        # 自注意力部分
        self.norm = normalization(channels)  # 例如 GroupNorm
        self.qkv = nn.Conv1d(channels, channels * 3, 1)  # 1x1 卷积，输出 3 * channels
        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

        # 如果 cond2 的通道数与 x 不同，则先投影到 channels
        if cond2_channels is not None and cond2_channels != channels:
            self.cond2_proj = nn.Conv1d(cond2_channels, channels, 1)
        else:
            self.cond2_proj = nn.Identity()

        # 交叉注意力部分，用于结合图片条件2
        self.norm_cross = normalization(channels)
        # 提取 cond2（投影后）的键和值，输出通道数为 2 * channels
        self.cross_kv = nn.Conv1d(channels, channels * 2, 1)
        self.cross_attention = CrossAttention(self.num_heads)
        self.cross_proj_out = zero_module(nn.Conv1d(channels, channels, 1))

        self.ffn

    def forward(self, x, cond2):
        """
        :param x: 输入张量，形状为 [B, C, ...]，... 表示空间维度，后续会 flatten 成 [B, C, T]
        :param cond2: 额外的图片条件2，形状可能为 [B, cond2_channels, ...]，空间尺寸可与 x 不同
        """
        if self.use_checkpoint:
            return checkpoint(self._forward, (x, cond2), self.parameters(), True)
        else:
            return self._forward(x, cond2)

    def _forward(self, x, cond2):
        b, c, *spatial = x.shape
        T = int(torch.tensor(spatial).prod().item())
        # 将 x reshape 为 [B, C, T]
        x_flat = x.reshape(b, c, T)

        # 自注意力分支
        qkv = self.qkv(self.norm(x_flat))  # 得到 [B, 3*C, T]
        h_self = self.attention(qkv)         # 自注意力输出，形状 [B, C, T]
        h_self = self.proj_out(h_self)
        h_self = x_flat + h_self             # 残差连接

        # 交叉注意力分支
        # 如果 cond2 的通道数不匹配，则先投影\
            # 处理 cond2：如果 cond2 维度超过 3（例如 [B, 1, H, W]），先 flatten 空间维度
        if cond2.dim() > 3:
            #插值并缩小到和x一样的尺寸
            cond2 = F.interpolate(cond2, spatial, mode="nearest")   
            b2, c2, *spatial2 = cond2.shape
            T2 = int(torch.tensor(spatial2).prod().item())# cond2
            cond2 = cond2.reshape(b2, c2, T2)
        cond2 = self.cond2_proj(cond2)

        b2, c2, *spatial2 = cond2.shape
        T2 = int(torch.tensor(spatial2).prod().item()) # cond2 的序列长度
        cond2_flat = cond2.reshape(b2, c2, T2)         # 将 cond2 flatten 成 [B, C, T2]
        # 提取 cond2 的键和值
        kv_cross = self.cross_kv(self.norm_cross(cond2_flat))  # 得到 [B, 2*C, T2]
        # 以自注意力输出作为 query，与 cond2 的键值对做交叉注意力
        h_cross = self.cross_attention(h_self, kv_cross)         # 输出 [B, C, T]
        h_cross = self.cross_proj_out(h_cross)

        # 最后融合两路信息（残差连接）
        out = h_self + h_cross
        return out.reshape(b, c, *spatial)

# class AttentionBlock_trans(nn.Module):
#     def __init__(
#     self,
#     channels,
#     num_heads=1,         # 每个注意力层的注意力头数
#     num_head_channels=-1,  # 每个注意力头的通道数
#     cond2_channels=1,   # 如果 cond2 的通道数和 x 不同，则传入 cond2 的通道数，否则默认为 channels
#     use_checkpoint=False,
#     use_new_attention_order=False,  # 是否使用新的注意力顺序
#     dropout=0,
#     depth=1,
#     ):
#         super().__init__()
#         self.in_channels = channels
#         inner_dim = num_heads * num_head_channels
#         self.norm = normalization(channels) 

#         self.proj_in = nn.Conv2d(channels,
#                                 inner_dim,
#                                 kernel_size=1,
#                                 stride=1,
#                                 padding=0)

#         self.transformer_blocks = nn.ModuleList(
#             [BasicTransformerBlock(inner_dim, num_heads, num_head_channels, dropout=dropout, context_dim=cond2_channels)
#                 for d in range(depth)]
#         )

#         self.proj_out = zero_module(nn.Conv2d(inner_dim,
#                                             channels,
#                                             kernel_size=1,
#                                             stride=1,
#                                             padding=0))

#     def forward(self, x, context):
#         # note: if no context is given, cross-attention defaults to self-attention
#         b, c, h, w = x.shape
#         x_in = x
#         x = self.norm(x)
#         x = self.proj_in(x)
#         x = rearrange(x, 'b c h w -> b (h w) c')

#         if context.dim() > 3:
#             #插值并缩小到和x一样的尺寸
#             context = F.interpolate(context, (h, w), mode="nearest")
#             b2, c2, h2, w2 = context.shape
#             context = rearrange(context, 'b c h w -> b (h w) c')

#         for block in self.transformer_blocks:
#             x = block(x, context=context)
#         x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
#         x = self.proj_out(x)
#         return x + x_in
class AttentionBlock_trans(nn.Module):
    def __init__(
    self,
    channels,
    num_heads=1,         # 每个注意力层的注意力头数
    num_head_channels=-1,  # 每个注意力头的通道数
    cond2_channels=1,   # 如果 cond2 的通道数和 x 不同，则传入 cond2 的通道数，否则默认为 channels
    use_checkpoint=False,
    use_new_attention_order=False,  # 是否使用新的注意力顺序
    dropout=0,
    depth=1,
    ):
        super().__init__()
        self.in_channels = channels
        inner_dim = num_heads * num_head_channels
        self.norm = normalization(channels)

        self.proj_in = nn.Conv2d(channels,
                                inner_dim,
                                kernel_size=1,
                                stride=1,
                                padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, num_heads, num_head_channels, dropout=dropout, context_dim=inner_dim//2)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                            channels,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0))
        

        self.context_compressor = nn.Sequential(
            nn.Conv2d(cond2_channels, inner_dim // 4, kernel_size=3, stride=2, padding=1),  # (b, inner_dim//4, 64, 64)
            nn.ReLU(),
            nn.Conv2d(inner_dim // 4, inner_dim // 2, kernel_size=3, stride=2, padding=1),  # (b, inner_dim//2, 32, 32)
            nn.ReLU(),
            nn.Conv2d(inner_dim // 2, inner_dim//2, kernel_size=3, stride=1, padding=1),       # (b, inner_dim, 32, 32)
        )

    def forward(self, x, context):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        if context.dim() > 3:
            #插值并缩小到和x一样的尺寸
            #print("context shape",context.shape)
            context = self.context_compressor(context)  # (b, inner_dim, 32, 32)
            #print("context shape after compressor",context.shape)
            b2, c2, h2, w2 = context.shape
            context = rearrange(context, 'b c h w -> b (h w) c')

        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in



class UNet(nn.Module):
    """
    The full UNet model with attention and embedding.
    :param in_channel: channels in the input Tensor, for image colorization : Y_channels + X_channels .
    :param inner_channel: base channel count for the model.
    :param out_channel: channels in the output Tensor.
    :param res_blocks: number of residual blocks per downsample.
    :param attn_res: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mults: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channel,
        inner_channel, #基准通道数，用来乘以channel_mults
        out_channel,
        res_blocks, #每个上下采样的残差块数量
        attn_res, #在哪些下采样率使用注意力机制，比如4，那么在4倍下采样时使用注意力机制
        dropout=0, #dropout概率
        channel_mults=(1, 2, 4, 8), #每个层级的通道数乘数
        dilations=None, #卷积的膨胀率，膨胀
        conv_resample=True,  #是否使用学习的卷积进行上采样和下采样，如果为False，使用插值或池化方式固定权重，表达能力更弱
        use_checkpoint=False, #是否使用梯度检查点，启用后在前向传播时不会保存中间结果，节省显存，反向传播时会重新计算，因此会增加计算时间。通常用于显存不足的情况
        use_fp16=False, #半精度计算，减少显存占用
        num_heads=1, #每个注意力层的注意力头数
        num_head_channels=-1, #如果指定，忽略num_heads，而是使用固定的通道宽度
        num_heads_upsample=-1, #与num_heads一起使用，设置不同的头数进行上采样。已弃用。
        use_scale_shift_norm=True, #使用FiLM-like的条件机制，在残差块中使用。对归一化后的特征，引入由得到的条件shift和scale
        resblock_updown=True, #是否在残差块内部上采样，false在残差块外部上采样
        use_new_attention_order=False,
    ):

        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channel = in_channel
        self.inner_channel = inner_channel
        self.out_channel = out_channel
        self.res_blocks = res_blocks
        self.attn_res = attn_res
        self.dropout = dropout
        self.channel_mults = channel_mults
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        if not (dilations is not None and len(channel_mults) == len(dilations)): #如果没有指定膨胀率，或者通道数和膨胀率不匹配
            dilations = [1] * len(channel_mults)   #默认各个层级的膨胀率都是1

        self.self_condlayer = SelfConditionEncoder(in_channels=1)

        cond_embed_dim = inner_channel * 4 #条件嵌入维度
        self.cond_embed = nn.Sequential(   #条件嵌入网络,形状为inner_channel, 4*inner_channel,将低维的条件嵌入映射到高维。输出的维度是4倍的inner_channel
            nn.Linear(inner_channel, cond_embed_dim),
            SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
        )

        ch = input_ch = int(channel_mults[0] * inner_channel) #输入通道数
        self.input_blocks = nn.ModuleList(   #输入层
            [EmbedSequential(nn.Conv2d(in_channel, ch, 3, padding=2 ** (dilations[0] - 1), dilation=dilations[0]))]
        )   #输入层的第一个卷积层，输入通道数为in_channel，输出通道数为ch，卷积核大小为3，填充大小为2**(dilations[0]-1)，膨胀率为dilations[0]
        self._feature_size = ch
        input_block_chans = [ch] #输入层的通道数，第一个卷积层的输出通道数
        ds = 1       #层数，也是下采样率
        for level, (mult, dilation) in enumerate(zip(channel_mults, dilations)):  #默认为【1,2,4,8】，【1,1,1,1】
            for _ in range(res_blocks):  #每个层级的残差块数量，默认为2
                layers = [
                    ResBlock(     #建立一个resblock，输入通道数为ch，输出通道数为mult*inner_channel（最开始是64）
                        ch,
                        cond_embed_dim,
                        dropout,
                        dilation=dilation,
                        out_channel=int(mult * inner_channel),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * inner_channel)    #更新ch
                if ds in attn_res:        #如果当前下采样率在attn_res中，说明需要注意力机制
                    layers.append(        # 建立一个注意力块，输入通道数为ch，输出通道数为ch
                        AttentionBlock_trans(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(EmbedSequential(*layers))   #将残差块和注意力块加入到输入层中
                self._feature_size += ch                             #更新特征大小
                input_block_chans.append(ch)                         #更新输入层的通道数                        
            if level != len(channel_mults) - 1:                   #如果不是最后一层，即不是最后一个下采样率
                out_ch = ch                                       #输出通道数为ch
                self.input_blocks.append(                         #将下采样块加入到输入层中
                    EmbedSequential(                              #如果resblock_updown为True，建立一个可以下采样的残差块，否则建立一个单纯下采样块
                        ResBlock(
                            ch,
                            cond_embed_dim,
                            dropout,
                            out_channel=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, out_channel=out_ch
                        )
                    )
                )
                ch = out_ch                                      #更新ch
                input_block_chans.append(ch)                    #更新输入层的通道数
                ds *= 2                                             #下采样率翻倍
                self._feature_size += ch                            #更新特征大小
                                                            #至此，下采样部分建立完成
        self.middle_block = EmbedSequential(                #中间部分，建立一个残差块和一个注意力块和一个残差快
            ResBlock(
                ch,
                cond_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock_trans(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                cond_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])             #输出部分，上菜样部分
        for level, mult in list(enumerate(channel_mults))[::-1]: #level【3,2,1,0】，mult【8,4,2,1】
            for i in range(res_blocks + 1):    
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        cond_embed_dim,
                        dropout,
                        out_channel=int(inner_channel * mult),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(inner_channel * mult)
                if ds in attn_res:
                    layers.append(
                        AttentionBlock_trans(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            cond_embed_dim,
                            dropout,
                            out_channel=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, out_channel=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(nn.Conv2d(input_ch, out_channel, 3, padding=2 ** (dilations[-1] - 1), dilation=dilations[-1])),
        )

    def forward(self, x, gammas,cond2,self_cond=None):
        """
        Apply the model to an input batch.
        :param x: an [N x 2 x ...] Tensor of inputs (B&W)
        :param gammas: a 1-D batch of gammas.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        gammas = gammas.view(-1, ) #b,1 -> b
        emb = self.cond_embed(gamma_embedding(gammas, self.inner_channel))   #将嵌入的条件b映射到高维，N,inner_channel.N代表、

        if self_cond is not None:
            self_cond = self.self_condlayer(self_cond) #自条件编码器，输入为噪声图片，输出为4个下采样的特征图

        h = x.type(torch.float32)
        cond2 = cond2.type(torch.float32) 
        for module in self.input_blocks:
            if self_cond is not None:
                if h.shape== self_cond[0].shape:
                    h=h+self_cond[0]
                    self_cond.pop(0)
            h = module(h, emb,cond2)
            hs.append(h)#将每个模块的输出保存起来,跳跃连接
            
        if self_cond is not None:
            if h.shape== self_cond[0].shape:
                h=h+self_cond[0]
                self_cond.pop(0)
        h = self.middle_block(h, emb,cond2)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb,cond2)
        h = h.type(x.dtype)

        return self.out(h)





class SelfConditionEncoder(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        
        # 下采样路径
        self.enc1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.enc5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        # x: [B, 2, 128, 128]
        sc0 = self.enc1(x)              # [B, 64, 128, 128]
        x = self.pool(sc0)

        sc1 = self.enc2(x)             # [B, 128, 64, 64]
        x = self.pool(sc1)

        sc2 = self.enc3(x)             # [B, 256, 32, 32]
        x = self.pool(sc2)

        sc3 = self.enc4(x)             # [B, 512, 16, 16]

        sc4 = self.enc5(sc3)             # [B, 512, 16, 16]

        return [sc0, sc1, sc2, sc3, sc4]
    

if __name__ == '__main__':
    b, c, h, w = 3, 6, 64, 64
    timsteps = 100
    model = UNet(
        image_size=h,
        in_channel=c,
        inner_channel=64,
        out_channel=3,
        res_blocks=2,
        attn_res=[8]
    )
    x = torch.randn((b, c, h, w))
    emb = torch.ones((b, ))
    out = model(x, emb)