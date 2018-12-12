import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable



class MSELoss2d(nn.Module):
    def __init__(self, size_average=True, reduce = True):
        super(MSELoss2d, self).__init__()
        self.reduce = reduce
        self.size_average = size_average

    def forward(self, y_preds, y_true):
        logp = F.log_softmax(y_preds, dim=1)
        return F.mse_loss(logp, y_true, size_average=self.size_average, reduce=self.reduce)

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-100):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, y_preds, y_true):
        '''
        :param y_preds: (N, C, H, W), Variable of FloatTensor
        :param y_true:  (N, H, W), Variable of LongTensor
        # :param weights: sample weights, (N, H, W), Variable of FloatTensor
        :return:
        '''
        logp = F.log_softmax(y_preds,dim=1)    # (N, C, H, W)
        ylogp = torch.gather(logp, 1, y_true.view(y_true.size(0), 1, y_true.size(1), y_true.size(2))) # (N, 1, H, W)
        return -(ylogp.squeeze(1)).mean()


class BCELogitsLossWithMask(nn.Module):

    def __init__(self, size_average=True):
        super(BCELogitsLossWithMask, self).__init__()
        self.size_average = size_average

    def forward(self, input, target, mask):
        '''
        :param input: Variable of shape (N, C, H, W)  logits
        :param target:  Variable of shape (N, C, H, W)  0~1 float
        :param mask: Variable of shape (N, C)  0. or 1.  float
        :return:
        '''
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        loss = loss * mask.unsqueeze(2).unsqueeze(3).expand_as(input)

        if self.size_average:
            return loss.sum() / (mask.sum()+1)
        else:
            return loss.sum()



class CrossEntropyLoss2d_sigmod_withmask(nn.Module):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w
    # label shape batch_size x channels x h x w
    def __init__(self):
        super(CrossEntropyLoss2d_sigmod_withmask, self).__init__()
        self.Sigmoid = nn.Sigmoid()
    def forward(self, inputs, targets,masks):
        if not (targets.size() == inputs.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), input.size()))

        inputs = self.Sigmoid(inputs)
        loss = -targets * torch.log(inputs + 1e-7) - (1 - targets) * torch.log(1 - inputs + 1e-7)
        loss = loss * masks.unsqueeze(2).unsqueeze(3).expand_as(inputs)
        return torch.sum(loss)/inputs.size(0)


class MESLossWithMask(nn.Module):
    def __init__(self, size_average):
        super(MESLossWithMask, self).__init__()
        self.size_average = size_average
    def forward(self, inputs, targets, masks):
        if not (targets.size() == inputs.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), input.size()))
        loss = (targets - inputs) ** 2 * masks.unsqueeze(2).unsqueeze(3).expand_as(inputs)

        if self.size_average:
            return loss.sum() / (masks.sum()+1)
        else:
            return loss.sum()
