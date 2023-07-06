import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):

        smooth = 1e-5
        # print(input.shape)
        # input = input.cpu().detach().numpy()
        # input_=np.zeros((input.shape[0],input.shape[1],input.shape[2]))
        # for i in range(input.shape[0]):
        #     for j in range(input.shape[1]):
        #         for k in range(input.shape[2]):
        #             if input[i,j,k]>0.5:
        #                 input_[i,j,k]=1 #>0.5
        # input_=round(input)
        # print(input_.shape)
        target_ = target #/255#> 127
        # print(input[10,200,200])
        #
        # print(target_[10,200,200])
        # input = torch.sigmoid(input)
        num1 = input.size(0)
        num2 = target_.size(0)
        input = input.view(num1, -1)
        target_ = target_.view(num2, -1)
        # print(input.shape)
        # print(target_.shape)
        intersection = (input * target_)

        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target_.sum(1) + smooth)
        dice = 1 - dice.sum() / num2
        bce = F.binary_cross_entropy_with_logits(input, target_)
        # print(dice,bce)
        return 0.5 * bce + dice
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, im1, im2):

        smooth = 1e-5
        # print(input.shape)
        # input = input.cpu().detach().numpy()
        # input_=np.zeros((input.shape[0],input.shape[1],input.shape[2]))
        # for i in range(input.shape[0]):
        #     for j in range(input.shape[1]):
        #         for k in range(input.shape[2]):
        #             if input[i,j,k]>0.5:
        #                 input_[i,j,k]=1 #>0.5
        # input_=round(input)
        # print(input_.shape)
        im1=im1.data.cpu().numpy()
        im2= im2.data.cpu().numpy()
        im1 = np.asarray(im1).astype(np.bool)
        im2 = np.asarray(im2).astype(np.bool)

        if im1.shape != im2.shape:
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

        im_sum = im1.sum() + im2.sum()
        if im_sum == 0:
            return empty_score

        # Compute Dice coefficient
        intersection = np.logical_and(im1, im2)
        # print(1- 2. * intersection.sum() / im_sum)
        return 1- 2. * intersection.sum() / im_sum

class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
