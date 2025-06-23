from abc import abstractmethod
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from numbers import Number

from .nn import (
    checkpoint,
    zero_module,
    normalization,
    count_flops_attn,
    gamma_embedding
)

from semencoder.semantic_encoder import SemanticEncoder

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

    def forward(self, x, emb, sem_cond=None):
        for layer in self:
            if isinstance(layer, EmbedBlock):  # 如果是EmbedBlock，将emb传入
                x = layer(x, emb, sem_cond)
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
        use_sem_cond=False,
        cond_channels=None,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_sem_cond = use_sem_cond
        self.cond_channels = cond_channels

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

        if self.use_sem_cond:
            self.sem_cond_layers = nn.Sequential(
                SiLU(),
                nn.Linear(cond_channels, self.out_channel),
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

    def forward(self, x, emb, sem_cond=None):
        """
        Apply the block to a Tensor, conditioned on a embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb, sem_cond), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb, sem_cond=None):
        if self.updown:                                                    #如果有上采样或下采样
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]     
            h = in_rest(x)                                                 #x经过groupnorm32和SiLU（输入层的前两层）
            h = self.h_upd(h)                                              #上面结果，再经过上采样或下采样   
            x = self.x_upd(x)                                              #x也经过上采样或下采样       
            h = in_conv(h)                                                 #h再经过卷积层，至此h从输入层输出，且经过了上采样或下采样，x也经过了上采样或下采样     
        else:
            h = self.in_layers(x)                                          #如果没有上采样或下采样，直接经过输入层
        emb_out = self.emb_layers(emb).type(h.dtype)                       #将emb经过嵌入层，输出的类型和h的类型一样
        if self.use_sem_cond:
            sem_cond_out = self.sem_cond_layers(sem_cond).type(h.dtype)
            
        while len(emb_out.shape) < len(h.shape):                           #如果emb_out的维度小于h的维度，不断扩展emb_out的维度直到和h的维度一样 emb_out[..., None] 后，其形状会变为 [N, emb_channels, 1]
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm and not self.use_sem_cond:                                      #如果使用scale_shift_norm，即FiLM-like的条件机制公式为h = norm(h) * (1 + scale) + shift.用于强化条件嵌入的影响.
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]   #将输出层分为norm和rest两部分
            scale, shift = torch.chunk(emb_out, 2, dim=1)                  #沿着第二个维度，也就是2*out_channel的维度，将emb_out分为两部分，scale和shift
            h = out_norm(h) * (1 + scale) + shift                          #调制
            h = out_rest(h)                                                #再经过rest                                   
        elif self.use_sem_cond and self.use_scale_shift_norm:
            h = apply_conditions(h,
                                 emb_out, 
                                 sem_cond_out, 
                                 self.out_layers, 
                                 scale_bias=1, 
                                 in_channels=self.out_channel)
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
        use_scale_shift_norm=True, #使用FiLM-like的条件机制，在残差块中使用。对归一化后的特征，引入由得到的条件shift和scale, 使得条件可以调制特征图的均值和方差，从而调制特征图的分布，增强条件嵌入的影响
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

        cond_embed_dim = inner_channel * 4 #条件嵌入维度
        self.cond_embed = nn.Sequential(   #条件嵌入网络,形状为inner_channel, 4*inner_channel,将低维的条件嵌入映射到高维。输出的维度是4倍的inner_channel
            nn.Linear(inner_channel, cond_embed_dim),
            SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
        )

        cond_embed_dim_half = inner_channel * 2 #条件嵌入维度的一半
        self.cond_embed_half = nn.Sequential(  #条件嵌入网络,形状为inner_channel, 2*inner_channel,将低维的条件嵌入映射到高维。输出的维度是2倍的inner_channel
            nn.Linear(inner_channel, cond_embed_dim_half),
            SiLU(),
            nn.Linear(cond_embed_dim_half, cond_embed_dim_half),
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
                        use_sem_cond=True,
                        cond_channels=768,
                    )
                ]
                ch = int(mult * inner_channel)    #更新ch
                if ds in attn_res:        #如果当前下采样率在attn_res中，说明需要注意力机制
                    layers.append(        # 建立一个注意力块，输入通道数为ch，输出通道数为ch
                        AttentionBlock(
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
                            use_sem_cond=True,
                            cond_channels=768,
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
                use_sem_cond=True,
                cond_channels=768,
            ),
            AttentionBlock(
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
                use_sem_cond=True,
                cond_channels=768,
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
                        AttentionBlock(
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

        self.sem_encoder = SemanticEncoder(
            pretrained_path=r'E:\thesiscode\secondpatchwithsem\semencoder\encoder_weights_checkpoint_epoch_15.pth',
            device='cuda',
            latent_dim=768,
            freeze_weights=True,
        )

        self.sem_cond_layers = nn.Sequential(
            nn.Linear(768, 768),
            nn.SiLU(),
            nn.Linear(768, 768),
        )

    def forward(self, x, gammas, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x 2 x ...] Tensor of inputs (B&W)
        :param gammas: a 1-D batch of gammas. #这里用gammas来表示时间步
        :return: an [N x C x ...] Tensor of outputs.
        """
        #gammas是padding后且patch后的时间步
        loss_mask = kwargs.get("loss_mask", None)
        patch_size = kwargs.get("patch_size", None)
        do_train = kwargs.get("do_train", True)
        ori_img_paded = kwargs.get("ori_img_paded", None)
        sample_gammas_ori_img_num = kwargs.get("sample_gammas_ori_img_num", None) #没有padding的原图的patch数量的gammas
        phase = kwargs.get("phase", None)
        padded_shape = kwargs.get("padded_shape", None)
        patch_pos = kwargs.get("patch_pos", None)
        embed_patch_pos = kwargs.get("embed_patch_pos", None) #没有padding的原图的patch的位置
        first_output = kwargs.get("first_output", None) #第一次diffusion的输出结果，如果没有就用默认值

        sem_cond = self.sem_encoder.encode(first_output) #b,768
        sem_cond = sem_cond.repeat_interleave(x.shape[0]//sem_cond.shape[0], dim=0) #b,768
        sem_cond = sem_cond.to(x.device)
        sem_cond = self.sem_cond_layers(sem_cond) #b,512


        hs = []
        hs_train = []
        gammas = gammas.view(-1, ) #b,1 -> b #训练时为p1 p2 b,1  p1p2为2。p1 p2为原图的patch大小
        sample_gammas_ori_img_num = sample_gammas_ori_img_num.view(-1, ) #b,1 -> b

        if patch_pos is None: #如果没有指定patch_pos,当前embed就只有gamma一个条件
            emb = self.cond_embed(gamma_embedding(gammas, self.inner_channel))   #将嵌入的条件b映射到高维，N,inner_channel.N代表
        else:
            patch_pos_x, patch_pos_y = patch_pos[:, 0], patch_pos[:, 1] #patch_pos的x和y坐标
            patch_pos_x = patch_pos_x.view(-1, ) #b,1 -> b
            patch_pos_y = patch_pos_y.view(-1, ) #b,1 -> b
            # pos_emb_x = self.cond_embed_half(gamma_embedding(patch_pos_x, self.inner_channel))
            # pos_emb_y = self.cond_embed_half(gamma_embedding(patch_pos_y, self.inner_channel))
            # pos_emb = torch.cat([pos_emb_x, pos_emb_y], dim=1) #将x和y的条件拼接在一起
            pos_emb_x = gamma_embedding(patch_pos_x, self.inner_channel//2).to(x.device)
            pos_emb_y = gamma_embedding(patch_pos_y, self.inner_channel//2).to(x.device)
            pos_emb = self.cond_embed_half(torch.cat([pos_emb_x, pos_emb_y], dim=1)) #将x和y的条件拼接在一起
            assert gammas.size(0) == patch_pos.size(0), "gamma size and patch_pos size should be the same {} != {}".format(gammas.size(0), patch_pos.size(0))
            gamma_embed = self.cond_embed_half(gamma_embedding(gammas, self.inner_channel)) #将gamma映射到高维
            emb = torch.cat([pos_emb, gamma_embed], dim=1) #将两个条件拼接在一起,最后维度是4倍的inner_channel

        if embed_patch_pos is None: #如果没有embed_patch_pos
            if sample_gammas_ori_img_num is not None: #未经过padding的原图的patch数量
                sample_gammas_ori_img_num = sample_gammas_ori_img_num.view(-1, ) #b,1 -> b
                emb_ori_img = self.cond_embed(gamma_embedding(sample_gammas_ori_img_num, self.inner_channel))
        else:
            embed_patch_pos_x, embed_patch_pos_y = embed_patch_pos[:, 0], embed_patch_pos[:, 1] #patch_pos的x和y坐标
            embed_patch_pos_x = embed_patch_pos_x.view(-1, ) #b,1 -> b
            embed_patch_pos_y = embed_patch_pos_y.view(-1, ) #b,1 -> b
            # pos_emb_x = self.cond_embed_half(gamma_embedding(embed_patch_pos_x, self.inner_channel))
            # pos_emb_y = self.cond_embed_half(gamma_embedding(embed_patch_pos_y, self.inner_channel))
            # pos_emb = torch.cat([pos_emb_x, pos_emb_y], dim=1) #将x和y的条件拼接在一起
            pos_emb_x = gamma_embedding(embed_patch_pos_x, self.inner_channel//2).to(x.device)
            pos_emb_y = gamma_embedding(embed_patch_pos_y, self.inner_channel//2).to(x.device)
            pos_emb = self.cond_embed_half(torch.cat([pos_emb_x, pos_emb_y], dim=1)) #将x和y的条件拼接在一起
            assert sample_gammas_ori_img_num.size(0) == embed_patch_pos.size(0), "gamma size and patch_pos size 0 should be the same {} != {}".format(gammas.shape, embed_patch_pos.shape)
            gamma_embed = self.cond_embed_half(gamma_embedding(sample_gammas_ori_img_num, self.inner_channel)) #将gamma映射到高维
            emb_ori_img = torch.cat([pos_emb, gamma_embed], dim=1) #将两个条件拼接在一起,最后维度是4倍的inner_channel

        #input blocks
        h = x.type(torch.float32)
        for module in self.input_blocks:
            h = module(h, emb, sem_cond)
            hs.append(h)#将每个模块的输出保存起来,跳跃连接
            hs_train.append(h)
        h = self.middle_block(h, emb, sem_cond)
        h_train = h.clone()#将中间块的输出保存起来，用于patch后的outputblocks
        

        #判断是否训练，如果训练，patch的数量为2，否则为原图的patch数量
        do_train = True if phase != "not train" else False
        if do_train:
            p1,p2 = 2,2
        else:
            p1 = padded_shape[2]//patch_size[0]
            p2 = padded_shape[3]//patch_size[1] 
        
        if do_train:
            #output blocks。patch不经过任何处理
            for module in self.output_blocks:
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, emb, sem_cond)
            h = h.type(x.dtype)
            pred_all = self.out(h)


        #patch后的outputblocks，这里经历了组合裁剪，输出的是patch化的
        for i in range(len(self.output_blocks)):
            b,c,height,width = h_train.shape
            if i == 0:
                halfp = int(height/2)
                #print('h_train',h_train.shape)
                h_ori = rearrange(h_train, '(b p1 p2) c h w -> b c (p1 h) (p2 w)', p1=p1, p2=p2)
                h_ori_crop = h_ori[:, :, halfp:-halfp, halfp:-halfp]
                h_shift = rearrange(h_ori_crop,'b c (p1 h) (p2 w) -> (b p1 p2) c h w', h=height, w=width) #crop后再进行patch
                h = h_shift           
            try:
                lateral = hs_train.pop()
                lateral_batch_size = lateral.shape[0]
                if lateral_batch_size != h.shape[0]:
                    b,c,height,width = h.shape
                    half_p = int(height//2)
                    lateral_ori = rearrange(lateral, '(b p1 p2) c h w -> b c (p1 h) (p2 w)', p1 = p1, p2 = p2)
                    lateral_ori_crop = lateral_ori[:, :, half_p:-half_p, half_p:-half_p]
                    lateral_shift = rearrange(lateral_ori_crop, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', h = height, w = width) #取出latera中的中间部分，然后变为原来的维度
                    lateral  = lateral_shift
            except IndexError:
                    #如果超出索引
                lateral = None

            if emb.size(0) != h.size(0):
                #print("emb.size", emb.size(), "h.size", h.size(), "lateral.size", lateral.size(), "emb_ori_img.size", emb_ori_img.size(), "sample_gammas_ori_img_num.size", sample_gammas_ori_img_num.size())
                assert sample_gammas_ori_img_num is not None and emb_ori_img.size(0) == h.size(0) == lateral.size(0)
                h = torch.cat([h, lateral], dim=1)
                h = self.output_blocks[i](h, emb_ori_img, sem_cond)
            else:
                assert emb.size(0) == h.size(0) == lateral.size(0)
                h = torch.cat([h, lateral], dim=1)
                h = self.output_blocks[i](h, emb, sem_cond)
        pred_cropped = self.out(h)
        if not do_train:
            pred_all=pred_cropped
        return {"pred_all": pred_all, "pred_cropped": pred_cropped}



        #return self.out(h)
def apply_conditions(
    h,
    emb=None,
    cond=None,
    layers=None,
    scale_bias=1,
    in_channels=512,
):
    """
    Apply conditions on the feature maps (adapted from unet_autoenc.py)
    
    Args:
        h: feature map
        emb: time conditional (ready to scale + shift)
        cond: semantic conditional (ready to scale)
        layers: sequential layers to apply
        scale_bias: bias for scaling
        in_channels: number of input channels
    """
    two_cond = emb is not None and cond is not None

    if emb is not None:
        # adjusting shapes
        while len(emb.shape) < len(h.shape):
            emb = emb[..., None]

    if two_cond:
        # adjusting shapes
        while len(cond.shape) < len(h.shape):
            cond = cond[..., None]
        # time first
        scale_shifts = [emb, cond]
    else:
        # "cond" is not used with single cond mode
        scale_shifts = [emb]

    # support scale, shift or shift only
    for i, each in enumerate(scale_shifts):
        if each is None:
            # special case: the condition is not provided
            a = None
            b = None
        else:
            if each.shape[1] == in_channels * 2:
                a, b = torch.chunk(each, 2, dim=1)
            else:
                a = each
                b = None
        scale_shifts[i] = (a, b)

    # condition scale bias could be a list
    if isinstance(scale_bias, Number):
        biases = [scale_bias] * len(scale_shifts)
    else:
        # a list
        biases = scale_bias

    # default, the scale & shift are applied after the group norm but BEFORE SiLU
    pre_layers, post_layers = layers[0], layers[1:]

    # split the post layer to be able to scale up or down before conv
    # post layers will contain only the conv
    mid_layers, post_layers = post_layers[:-2], post_layers[-2:]

    h = pre_layers(h)
    # scale and shift for each condition
    for i, (scale, shift) in enumerate(scale_shifts):
        # if scale is None, it indicates that the condition is not provided
        if scale is not None:
            h = h * (biases[i] + scale)
            if shift is not None:
                h = h + shift
    h = mid_layers(h)
    h = post_layers(h)
    return h


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