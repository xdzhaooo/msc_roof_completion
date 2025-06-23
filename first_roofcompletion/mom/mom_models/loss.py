import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchmetrics.image import StructuralSimilarityIndexMeasure

def masked_mse_loss(output, target, mask):
    loss = (output - target) ** 2 * mask
    return loss.sum() / mask.sum()

def masked_l1_loss(output, target, mask):
    loss = abs(output - target) * mask
    n = mask.sum()
    if n == 0:
        return 0
    return loss.sum() / n


@torch._dynamo.disable
def combined_loss_with_masked_ssim(output, target, mask, alpha=0.5):
    # 计算 masked L1 损失
    l1_loss = masked_l1_loss(output, target, mask)

    # 对 output 和 target 应用掩码
    masked_output = output * mask
    masked_target = target * mask

    # 计算 SSIM，data_range=2 因为范围是 [-1, 1]
    ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(output.device)
    ssim_value = ssim(masked_output, masked_target)
    ssim_loss = 1.0 - ssim_value

    # 结合损失
    total_loss = alpha * l1_loss + (1 - alpha) * ssim_loss
    return total_loss


def mse_loss(output, target):
    return F.mse_loss(output, target)

def l1_loss(output, target):
    return F.l1_loss(output, target)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

