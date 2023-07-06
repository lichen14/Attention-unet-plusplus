"""

Dice loss
用来处理分割过程中的前景背景像素非平衡的问题
"""

import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # print(pred.shape)
        # print(target.shape)
        # #
        # print(pred[0,150,150])
        #
        # print(target[0,150,150])
        # dice系数的定义
        dice = 2 * (pred * target).sum(dim=1).sum(dim=1) / (pred.pow(2).sum(dim=1).sum(dim=1)+          #pow(2)..sum(dim=1).sum(dim=1)
                                            target.pow(2).sum(dim=1).sum(dim=1) + 1e-5)                     #pow(2)..sum(dim=1)

        # 返回的是dice距离
        # print((1 - dice).mean())
        return (1 - dice).mean()
