"""

Dice loss
用来处理分割过程中的前景背景像素非平衡的问题
"""

import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt # plt 用于显示图片
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred = pred.cpu()#.numpy()
        # target = target.cpu()#.numpy()
        # print(pred.shape)
        # print(target.shape)
        # t=pred[0,10,:,:].data.cpu().numpy()
        # # print(t.shape)
        # # tt=np.zeros((256,256,3))
        # # tt[:,:,0]=t[0,:,:]
        # # # tt[:,:,1]=t[0,:,:]
        # # # tt[:,:,2]=t[1,:,:]
        # # print(tt[0,100,100])
        # # # m=target[1,:,:,:]
        # # mm=np.zeros((512,512))
        # mm=target[0,10,:,:].data.cpu().numpy()
        # # # mm[:,:,1]=target[1,:,:]
        # # # mm[:,:,2]=target[2,:,:]
        # # # t = t.transpose((1,2,0))
        # # # print(tt.shape)
        # plt.subplot(121)
        # plt.imshow(t) # 显示图片
        # plt.subplot(122)
        # plt.imshow(mm)#, cmap='Greys_r') # 显示图片
        # # plt.axis('off') # 不显示坐标轴
        # plt.show()
        #
        # print(pred[0,10,150,150])
        # # # # #
        # print(target[0,10,150,150])
        # dice系数的定义
        dice = 2 * (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / (pred.pow(2).sum(dim=1).sum(dim=1).sum(dim=1)+          #pow(2)..sum(dim=1).sum(dim=1)
                                            target.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + 1e-5)                     #pow(2)..sum(dim=1)

        # 返回的是dice距离
        # print((1 - dice).mean())
        return (1 - dice).mean()
