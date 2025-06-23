"""
Various utilities for neural networks.
"""

import math
import numpy as np
import torch 
import torch.nn as nn


class GroupNorm32(nn.GroupNorm): #作用是将输入的通道数分成若干组，每组进行归一化
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    #将模块的参数置零，并且脱离计算图，目的是残差网络中的残差块，不需要计算梯度
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)



def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    # 完全禁用梯度检查点以避免冻结参数导致的问题
    return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        # 只对需要梯度的参数进行梯度计算
        trainable_params = [p for p in ctx.input_params if p.requires_grad]
        
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        
        # 只对输入张量和可训练参数计算梯度
        all_inputs = ctx.input_tensors + trainable_params
        input_grads = torch.autograd.grad(
            output_tensors,
            all_inputs,
            output_grads,
            allow_unused=True,
        )
        
        # 重新构建完整的梯度列表，为冻结的参数填充None
        num_input_tensors = len(ctx.input_tensors)
        full_grads = list(input_grads[:num_input_tensors])  # 输入张量的梯度
        trainable_idx = 0
        
        for param in ctx.input_params:
            if param.requires_grad:
                full_grads.append(input_grads[num_input_tensors + trainable_idx])
                trainable_idx += 1
            else:
                full_grads.append(None)
        
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + tuple(full_grads)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


def gamma_embedding(gammas, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param gammas: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2  #整除2，向下取整，后续会计算余弦和正弦两组值，各占一半。
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=gammas.device)
    args = gammas[:, None].float() * freqs[None]  #广播，然后将gammas的每个元素分别乘以freqs的每个元素
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1) #将余弦和正弦的值拼接起来
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1) #如果dim是奇数，就在最后加上一个0
    return embedding