# -*- coding: utf-8 -*-
import numpy as np
import os
from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
import matplotlib
import matplotlib.pyplot as plt # plt 用于显示图片
from glob import glob
from scipy import misc
from skimage.io import imread, imsave

class Conv_Block(nn.Module):
    def __init__(self, in_channels,  out_channels, act_func=nn.ReLU(inplace=True)):
        super(Conv_Block, self).__init__()
        self.act_func1 = act_func
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act_func2 = act_func
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dp = nn.Dropout(0.6)

    def forward(self, x):
        out = self.conv1(x)
        out2 = self.bn1(out)
        # out = self.dp(out)
        out1 = self.act_func1(out2)

        out3 = self.conv2(out1)
        out4 = self.bn2(out3)

        out5 = self.act_func2(out4)

        return out5

class Conv_Block2(nn.Module):
    def __init__(self, in_channels,  out_channels, act_func=nn.ReLU(inplace=True)):
        super(Conv_Block2, self).__init__()
        self.act_func1 = act_func
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act_func2 = act_func
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dp = nn.Dropout(0.6)
        self.in_channels = in_channels
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dp(out)
        out = self.act_func1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.act_func2(out)
        # if self.in_channels==3 and flag==2:
        #     C = x.shape[0]
        #     path_file_number=glob.glob(pathname='/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images/0-*.png')
        #     for k in range(C):
        #         A = x[k,:,:,:]
        #         # print(A.shape)
        #         A = A.cpu().detach().numpy()
        #         A = A.transpose((1, 2, 0))
        #         # matplotlib.image.imsave('/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images1/1-{}-{}.png'.format(ep, k+len(path_file_number)), A)
        #         matplotlib.image.imsave('/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images/0-{}.png'.format(k+len(path_file_number)), A)
        return out

class Conv_Block_3d(nn.Module):
    def __init__(self, in_channels,  out_channels, act_func=nn.ReLU(inplace=True)):
        super(Conv_Block_3d, self).__init__()
        self.act_func1 = act_func
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.act_func2 = act_func
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dp = nn.Dropout(0.6)

    def forward(self, x):
        out = self.conv1(x)
        out2 = self.bn1(out)
        # out = self.dp(out)
        out1 = self.act_func1(out2)

        out3 = self.conv2(out1)
        out4 = self.bn2(out3)

        out5 = self.act_func2(out4)

        return out5

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2),    #尺寸扩大2倍
            nn.ConvTranspose2d(in_ch, out_ch, 2, 2),    #通道数改变
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        x = self.up(x)
        return x

class UNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = Conv_Block(args.input_channels, filters[0])
        self.Conv2 = Conv_Block(filters[0], filters[1])
        self.Conv3 = Conv_Block(filters[1], filters[2])
        self.Conv4 = Conv_Block(filters[2], filters[3])
        self.Conv5 = Conv_Block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = Conv_Block(filters[3]+filters[3], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = Conv_Block(filters[2]+filters[2], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = Conv_Block(filters[1]+filters[1], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = Conv_Block(filters[0]+filters[0], filters[0])

        self.Conv = nn.Conv2d(filters[0], 1, kernel_size=1, stride=1, padding=1)
        self.final_map = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            #nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.v0 = nn.Conv2d(filters[0], 1, 1, 1)
        self.v4 = nn.Conv2d(filters[0], 1, 1, 1)
        # self.final_map0 = nn.Sequential(
        #     nn.Conv2d(filters[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     # nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final_map1 = nn.Sequential(
        #     nn.Conv2d(filters[1], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     # nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final_map2 = nn.Sequential(
        #     nn.Conv2d(filters[2], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     # nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final_map3 = nn.Sequential(
        #     nn.Conv2d(filters[3], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     # nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
       # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)
        #print(e1.shape)
        e2 = self.Maxpool1(e1)
        #print(e2.shape)
        # c1 = self.final_map0(e2).cpu().detach().numpy()
        # print(c1[0,0,:,:])
        # imsave('output/c1.png', (c1[0,0,:,:]))

        #print(e2.shape)
        e2 = self.Conv2(e2)

        #print(e2.shape)
        e3 = self.Maxpool2(e2)
        # c2 = self.final_map1(e3).cpu().detach().numpy()
        # imsave('output/c2.png', (c2[0,0,:,:]))
        #print(e3.shape)
        e3 = self.Conv3(e3)
        #print(e3.shape)
        e4 = self.Maxpool3(e3)
        # c3 = self.final_map2(e4).cpu().detach().numpy()
        # imsave('output/c3.png', (c3[0,0,:,:]))
        #print(e4.shape)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        # c4 = self.final_map3(e5).cpu().detach().numpy()
        # imsave('output/c4.png', (c4[0,0,:,:]))
        # for i in range(16):
        #     print(c4[0,0,i,:])

        #print(e5.shape)
        e5 = self.Conv5(e5)
        #print(e5.shape)
        d5 = self.Up5(e5)
        #print(d5.shape)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.final_map(d2)

        # fv0= self.v0(e1)
        # fv4= self.v4(d2)
        # imsave('feature visualization/x00.png', (fv0[0,0,:,:].cpu()))
        # imsave('feature visualization/x01.png', (fv1[0,0,:,:].cpu()))
        # imsave('feature visualization/x02.png', (fv2[0,0,:,:].cpu()))
        # imsave('feature visualization/x03.png', (fv3[0,0,:,:].cpu()))
        # imsave('feature visualization/x04.png', (fv4[0,0,:,:].cpu()))
        # print(out.shape)
        # imsave('output/c55.png', (out[0,0,:,:].cpu().detach().numpy()))
        #d1 = self.active(out)

        return out

class UNetL3(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = Conv_Block(args.input_channels, filters[0])
        self.Conv2 = Conv_Block(filters[0], filters[1])
        self.Conv3 = Conv_Block(filters[1], filters[2])
        self.Conv4 = Conv_Block(filters[2], filters[3])
        self.Conv5 = Conv_Block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = Conv_Block(filters[3]+filters[3], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = Conv_Block(filters[2]+filters[2], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = Conv_Block(filters[1]+filters[1], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = Conv_Block(filters[0]+filters[0], filters[0])

        self.Conv = nn.Conv2d(filters[0], 1, kernel_size=1, stride=1, padding=1)
        self.final_map = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            #nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        # self.v0 = nn.Conv2d(filters[0], 1, 1, 1)
        # self.v4 = nn.Conv2d(filters[0], 1, 1, 1)
        # self.final_map0 = nn.Sequential(
        #     nn.Conv2d(filters[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     # nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final_map1 = nn.Sequential(
        #     nn.Conv2d(filters[1], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     # nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final_map2 = nn.Sequential(
        #     nn.Conv2d(filters[2], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     # nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final_map3 = nn.Sequential(
        #     nn.Conv2d(filters[3], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     # nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
       # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)
        #print(e1.shape)
        e2 = self.Maxpool1(e1)
        #print(e2.shape)
        # c1 = self.final_map0(e2).cpu().detach().numpy()
        # print(c1[0,0,:,:])
        # imsave('output/c1.png', (c1[0,0,:,:]))

        #print(e2.shape)
        e2 = self.Conv2(e2)

        #print(e2.shape)
        e3 = self.Maxpool2(e2)
        # c2 = self.final_map1(e3).cpu().detach().numpy()
        # imsave('output/c2.png', (c2[0,0,:,:]))
        #print(e3.shape)
        e3 = self.Conv3(e3)
        #print(e3.shape)
        e4 = self.Maxpool3(e3)
        # c3 = self.final_map2(e4).cpu().detach().numpy()
        # imsave('output/c3.png', (c3[0,0,:,:]))
        #print(e4.shape)
        e4 = self.Conv4(e4)

        # e5 = self.Maxpool4(e4)
        # c4 = self.final_map3(e5).cpu().detach().numpy()
        # imsave('output/c4.png', (c4[0,0,:,:]))
        # for i in range(16):
        #     print(c4[0,0,i,:])

        #print(e5.shape)
        # e5 = self.Conv5(e5)
        #print(e5.shape)
        # d5 = self.Up5(e5)
        #print(d5.shape)
        # d5 = torch.cat((e4, d5), dim=1)

        # d5 = self.Up_conv5(d5)

        d4 = self.Up4(e4)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.final_map(d2)

        # fv0= self.v0(e1)
        # fv4= self.v4(d2)
        # imsave('feature visualization/x00.png', (fv0[0,0,:,:].cpu()))
        # imsave('feature visualization/x01.png', (fv1[0,0,:,:].cpu()))
        # imsave('feature visualization/x02.png', (fv2[0,0,:,:].cpu()))
        # imsave('feature visualization/x03.png', (fv3[0,0,:,:].cpu()))
        # imsave('feature visualization/x04.png', (fv4[0,0,:,:].cpu()))
        # print(out.shape)
        # imsave('output/c55.png', (out[0,0,:,:].cpu().detach().numpy()))
        #d1 = self.active(out)

        return out

class UNetL2(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = Conv_Block(args.input_channels, filters[0])
        self.Conv2 = Conv_Block(filters[0], filters[1])
        self.Conv3 = Conv_Block(filters[1], filters[2])
        self.Conv4 = Conv_Block(filters[2], filters[3])
        self.Conv5 = Conv_Block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = Conv_Block(filters[3]+filters[3], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = Conv_Block(filters[2]+filters[2], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = Conv_Block(filters[1]+filters[1], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = Conv_Block(filters[0]+filters[0], filters[0])

        self.Conv = nn.Conv2d(filters[0], 1, kernel_size=1, stride=1, padding=1)
        self.final_map = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            #nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        # self.v0 = nn.Conv2d(filters[0], 1, 1, 1)
        # self.v4 = nn.Conv2d(filters[0], 1, 1, 1)
        # self.final_map0 = nn.Sequential(
        #     nn.Conv2d(filters[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     # nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final_map1 = nn.Sequential(
        #     nn.Conv2d(filters[1], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     # nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final_map2 = nn.Sequential(
        #     nn.Conv2d(filters[2], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     # nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final_map3 = nn.Sequential(
        #     nn.Conv2d(filters[3], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     # nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
       # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)
        #print(e1.shape)
        e2 = self.Maxpool1(e1)
        #print(e2.shape)
        # c1 = self.final_map0(e2).cpu().detach().numpy()
        # print(c1[0,0,:,:])
        # imsave('output/c1.png', (c1[0,0,:,:]))

        #print(e2.shape)
        e2 = self.Conv2(e2)

        #print(e2.shape)
        e3 = self.Maxpool2(e2)
        # c2 = self.final_map1(e3).cpu().detach().numpy()
        # imsave('output/c2.png', (c2[0,0,:,:]))
        #print(e3.shape)
        e3 = self.Conv3(e3)
        #print(e3.shape)
        # e4 = self.Maxpool3(e3)
        # c3 = self.final_map2(e4).cpu().detach().numpy()
        # imsave('output/c3.png', (c3[0,0,:,:]))
        #print(e4.shape)
        # e4 = self.Conv4(e4)

        # e5 = self.Maxpool4(e4)
        # c4 = self.final_map3(e5).cpu().detach().numpy()
        # imsave('output/c4.png', (c4[0,0,:,:]))
        # for i in range(16):
        #     print(c4[0,0,i,:])

        #print(e5.shape)
        # e5 = self.Conv5(e5)
        #print(e5.shape)
        # d5 = self.Up5(e5)
        #print(d5.shape)
        # d5 = torch.cat((e4, d5), dim=1)

        # d5 = self.Up_conv5(d5)

        # d4 = self.Up4(e4)
        # d4 = torch.cat((e3, d4), dim=1)
        # d4 = self.Up_conv4(d4)

        d3 = self.Up3(e3)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.final_map(d2)

        # fv0= self.v0(e1)
        # fv4= self.v4(d2)
        # imsave('feature visualization/x00.png', (fv0[0,0,:,:].cpu()))
        # imsave('feature visualization/x01.png', (fv1[0,0,:,:].cpu()))
        # imsave('feature visualization/x02.png', (fv2[0,0,:,:].cpu()))
        # imsave('feature visualization/x03.png', (fv3[0,0,:,:].cpu()))
        # imsave('feature visualization/x04.png', (fv4[0,0,:,:].cpu()))
        # print(out.shape)
        # imsave('output/c55.png', (out[0,0,:,:].cpu().detach().numpy()))
        #d1 = self.active(out)

        return out

class UNetL1(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = Conv_Block(args.input_channels, filters[0])
        self.Conv2 = Conv_Block(filters[0], filters[1])
        self.Conv3 = Conv_Block(filters[1], filters[2])
        self.Conv4 = Conv_Block(filters[2], filters[3])
        self.Conv5 = Conv_Block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = Conv_Block(filters[3]+filters[3], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = Conv_Block(filters[2]+filters[2], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = Conv_Block(filters[1]+filters[1], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = Conv_Block(filters[0]+filters[0], filters[0])

        self.Conv = nn.Conv2d(filters[0], 1, kernel_size=1, stride=1, padding=1)
        self.final_map = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            #nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        # self.v0 = nn.Conv2d(filters[0], 1, 1, 1)
        # self.v4 = nn.Conv2d(filters[0], 1, 1, 1)
        # self.final_map0 = nn.Sequential(
        #     nn.Conv2d(filters[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     # nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final_map1 = nn.Sequential(
        #     nn.Conv2d(filters[1], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     # nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final_map2 = nn.Sequential(
        #     nn.Conv2d(filters[2], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     # nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final_map3 = nn.Sequential(
        #     nn.Conv2d(filters[3], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     # nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
       # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)
        #print(e1.shape)
        e2 = self.Maxpool1(e1)
        #print(e2.shape)
        # c1 = self.final_map0(e2).cpu().detach().numpy()
        # print(c1[0,0,:,:])
        # imsave('output/c1.png', (c1[0,0,:,:]))

        #print(e2.shape)
        e2 = self.Conv2(e2)

        #print(e2.shape)
        # e3 = self.Maxpool2(e2)
        # # c2 = self.final_map1(e3).cpu().detach().numpy()
        # # imsave('output/c2.png', (c2[0,0,:,:]))
        # #print(e3.shape)
        # e3 = self.Conv3(e3)
        #print(e3.shape)
        # e4 = self.Maxpool3(e3)
        # c3 = self.final_map2(e4).cpu().detach().numpy()
        # imsave('output/c3.png', (c3[0,0,:,:]))
        #print(e4.shape)
        # e4 = self.Conv4(e4)

        # e5 = self.Maxpool4(e4)
        # c4 = self.final_map3(e5).cpu().detach().numpy()
        # imsave('output/c4.png', (c4[0,0,:,:]))
        # for i in range(16):
        #     print(c4[0,0,i,:])

        #print(e5.shape)
        # e5 = self.Conv5(e5)
        #print(e5.shape)
        # d5 = self.Up5(e5)
        #print(d5.shape)
        # d5 = torch.cat((e4, d5), dim=1)

        # d5 = self.Up_conv5(d5)

        # d4 = self.Up4(e4)
        # d4 = torch.cat((e3, d4), dim=1)
        # d4 = self.Up_conv4(d4)

        # d3 = self.Up3(e3)
        # d3 = torch.cat((e2, d3), dim=1)
        # d3 = self.Up_conv3(d3)

        d2 = self.Up2(e2)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.final_map(d2)

        # fv0= self.v0(e1)
        # fv4= self.v4(d2)
        # imsave('feature visualization/x00.png', (fv0[0,0,:,:].cpu()))
        # imsave('feature visualization/x01.png', (fv1[0,0,:,:].cpu()))
        # imsave('feature visualization/x02.png', (fv2[0,0,:,:].cpu()))
        # imsave('feature visualization/x03.png', (fv3[0,0,:,:].cpu()))
        # imsave('feature visualization/x04.png', (fv4[0,0,:,:].cpu()))
        # print(out.shape)
        # imsave('output/c55.png', (out[0,0,:,:].cpu().detach().numpy()))
        #d1 = self.active(out)

        return out


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        # self.ep=epoch
    def forward(self,g,x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # print(g1.shape)
        # C = g1.shape[0]
        # path_file_number=glob.glob(pathname='/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images1/4-*.png')
        # if flag==2:
        #     for k in range(C):
        #         A = g1[k,0,:,:]
        #         A = A.cpu().detach().numpy()
        #         # matplotlib.image.imsave('/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images1/4-{}-{}.png'.format(ep, k+len(path_file_number)), A)
        #         matplotlib.image.imsave('/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images1/4-{}-{}.png'.format(ep, k), A)
        # print(g1.shape)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # print(x1.shape)
        # print(x1.shape)
        # concat + relu
        psi = self.relu(g1+x1)
        # print(psi.shape)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)

        # writer = pd.ExcelWriter('/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images1/1.xlsx')
        # xx_df =pd.DataFrame(xx)
        # xx_df.to_excel(writer,'page_1',float_format='%.5f')
        # writer.save()
        # # print(psi.shape)
        # if f==1:
        #     xx=psi.cpu().detach().numpy()   #[1,0,:,:]
        #     plotNNFilterOverlay(xx, figure_id=epoch, interp='bilinear', colormap=plt.cm.jet,  alpha=0.8)
        # plt.imshow(xx,cmap=plt.cm.jet)
        # plt.savefig('/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images1/1111.png')
        # print(psi.shape)
        result = x*psi
        # print(result.shape)
		# 返回加权的 x
        return result

class Attention_block2(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block2,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        # self.ep=epoch
    def forward(self,g,x,epoch,f):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # print(g1.shape)
        # C = g1.shape[0]
        # path_file_number=glob.glob(pathname='/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images1/4-*.png')
        # if flag==2:
        #     for k in range(C):
        #         A = g1[k,0,:,:]
        #         A = A.cpu().detach().numpy()
        #         # matplotlib.image.imsave('/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images1/4-{}-{}.png'.format(ep, k+len(path_file_number)), A)
        #         matplotlib.image.imsave('/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images1/4-{}-{}.png'.format(ep, k), A)
        # print(g1.shape)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # print(x1.shape)
        # print(x1.shape)
        # concat + relu
        psi = self.relu(g1+x1)
        # print(psi.shape)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)

        # writer = pd.ExcelWriter('/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images1/1.xlsx')
        # xx_df =pd.DataFrame(xx)
        # xx_df.to_excel(writer,'page_1',float_format='%.5f')
        # writer.save()
        # # print(psi.shape)
        if f==1:
            xx=psi.cpu().detach().numpy()   #[1,0,:,:]
            plotNNFilterOverlay(xx, figure_id=epoch, interp='bilinear', colormap=plt.cm.jet,  alpha=0.8)
        # plt.imshow(xx,cmap=plt.cm.jet)
        # plt.savefig('/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images1/1111.png')
        # print(psi.shape)
        result = x*psi
        # print(result.shape)
		# 返回加权的 x
        return result

class Attention_block3(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block3,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        # self.ep=epoch
    def forward(self,g,x,epoch,f):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # print(g1.shape)
        # C = g1.shape[0]
        # path_file_number=glob.glob(pathname='/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images1/4-*.png')
        # if flag==2:
        #     for k in range(C):
        #         A = g1[k,0,:,:]
        #         A = A.cpu().detach().numpy()
        #         # matplotlib.image.imsave('/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images1/4-{}-{}.png'.format(ep, k+len(path_file_number)), A)
        #         matplotlib.image.imsave('/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images1/4-{}-{}.png'.format(ep, k), A)
        # print(g1.shape)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # print(x1.shape)
        # print(x1.shape)
        # concat + relu
        psi = self.relu(g1+x1)
        # print(psi.shape)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)

        # writer = pd.ExcelWriter('/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images1/1.xlsx')
        # xx_df =pd.DataFrame(xx)
        # xx_df.to_excel(writer,'page_1',float_format='%.5f')
        # writer.save()
        # # print(psi.shape)

        # if f==1:
        #     xx=psi.cpu().detach().numpy()   #[1,0,:,:]
        #     plotNNFilterOverlay(xx, figure_id=epoch, interp='bilinear', colormap=plt.cm.jet,  alpha=0.8)

        # plt.imshow(xx,cmap=plt.cm.jet)
        # plt.savefig('/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images1/1111.png')
        # print(psi.shape)
        result = x*psi
        # print(result.shape)
		# 返回加权的 x
        return result

class Attention_block4(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block4,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        # self.ep=epoch
    def forward(self,g,x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # print(g1.shape)
        # C = g1.shape[0]
        # path_file_number=glob.glob(pathname='/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images1/4-*.png')
        # if flag==2:
        #     for k in range(C):
        #         A = g1[k,0,:,:]
        #         A = A.cpu().detach().numpy()
        #         # matplotlib.image.imsave('/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images1/4-{}-{}.png'.format(ep, k+len(path_file_number)), A)
        #         matplotlib.image.imsave('/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images1/4-{}-{}.png'.format(ep, k), A)
        # print(g1.shape)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # print(x1.shape)
        # print(x1.shape)
        # concat + relu
        psi = self.relu(g1+x1)
        # print(psi.shape)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)

        # writer = pd.ExcelWriter('/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images1/1.xlsx')
        # xx_df =pd.DataFrame(xx)
        # xx_df.to_excel(writer,'page_1',float_format='%.5f')
        # writer.save()
        # # print(psi.shape)

        # if f==1:
        #     xx=psi.cpu().detach().numpy()   #[1,0,:,:]
        #     plotNNFilterOverlay(xx, figure_id=epoch, interp='bilinear', colormap=plt.cm.jet,  alpha=0.8)

        # plt.imshow(xx,cmap=plt.cm.jet)
        # plt.savefig('/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images1/1111.png')
        # print(psi.shape)
        result = x*psi
        # print(result.shape)
		# 返回加权的 x
        return result

class Attention_block_3d(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block_3d,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        # self.ep=epoch
    def forward(self,g,x):#,epoch,f
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # print(g1.shape) #[1, 128, 6, 32, 32]
        # C = g1.shape[1]
        # path_file_number=glob(pathname='/home/lc/学习/DataBase/LITS(1)/attention map/*.png')
        # # if flag==2:
        # # for k in range(C):
        # A = g1[0,0,0,:,:]
        # A = A.cpu().detach().numpy()
        # # matplotlib.image.imsave('/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images1/4-{}-{}.png'.format(ep, k+len(path_file_number)), A)
        # matplotlib.image.imsave('/home/lc/学习/DataBase/LITS(1)/attention map/{}.png'.format(len(path_file_number)), A)
        # print(g1.shape)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # print(x1.shape)
        # print(x1.shape)
        # concat + relu
        psi = self.relu(g1+x1)
        # print(psi.shape)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)

        # writer = pd.ExcelWriter('/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images1/1.xlsx')
        # xx_df =pd.DataFrame(xx)
        # xx_df.to_excel(writer,'page_1',float_format='%.5f')
        # writer.save()
        # print(psi.shape)        #[1, 1, 6, 32, 32]

        # if f==1:
        #     xx=psi.cpu().detach().numpy()   #[1,0,:,:]
        #     plotNNFilterOverlay(xx, figure_id=epoch, interp='bilinear', colormap=plt.cm.jet,  alpha=0.8)

        # plt.imshow(xx[0,0,0,:,:],cmap=plt.cm.jet)
        # plt.savefig('/home/lc/学习/DataBase/LITS(1)/attention map/1111.png')
        # print(psi.shape)
        result = x*psi
        # print(result.shape)
		# 返回加权的 x
        return result


class AttentionUNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        filters = [ 64, 128, 256, 512, 1024]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(2, 2)
        self.Maxpool3 = nn.MaxPool2d(2, 2)
        self.Maxpool4 = nn.MaxPool2d(2, 2)

        self.Up5 = up_conv(filters[4], filters[4])
        self.Up4 = up_conv(filters[3], filters[3])
        self.Up3 = up_conv(filters[2], filters[2])
        self.Up2 = up_conv(filters[1], filters[1])

        self.conv0_0 = Conv_Block2(args.input_channels, filters[0])
        self.conv1_0 = Conv_Block(filters[0], filters[1])
        self.conv2_0 = Conv_Block(filters[1], filters[2])
        self.conv3_0 = Conv_Block(filters[2], filters[3])
        self.conv4_0 = Conv_Block(filters[3], filters[4])


        self.final = nn.Conv2d(filters[0], 1, kernel_size=1, stride=1, padding=1)

        self.Att5 = Attention_block(F_g=filters[4], F_l=filters[3], F_int=filters[2])
        self.Att4 = Attention_block2(F_g=filters[3], F_l=filters[2], F_int=filters[1])
        self.Att3 = Attention_block3(F_g=filters[2], F_l=filters[1], F_int=filters[0])
        self.Att2 = Attention_block4(F_g=filters[1], F_l=filters[0], F_int=32)

        self.Up_conv5 = Conv_Block(filters[4], filters[3])
        self.Up_conv4 = Conv_Block(filters[3], filters[2])
        self.Up_conv3 = Conv_Block(filters[2], filters[1])
        self.Up_conv2 = Conv_Block(filters[1], filters[0])

    def forward(self, input,epoch,flag):
        # encoding path
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.Maxpool1(x0_0))
        x2_0 = self.conv2_0(self.Maxpool2(x1_0))
        x3_0 = self.conv3_0(self.Maxpool3(x2_0))
        x4_0 = self.conv4_0(self.Maxpool4(x3_0))

        # decoding + concat path
        tmp4=self.Up5(x4_0)
        att4=self.Att5(g=tmp4,x=x3_0,epoch=epoch,f=flag)
        # print('att4:',att4.shape)
        # print('tmp4:',tmp4.shape)
        x3_1 = torch.cat([att4,tmp4], 1)
        x3_1 = self.Up_conv5(x3_1)

        tmp3=self.Up4(x3_1)
        att3=self.Att4(g=tmp3,x=x2_0,epoch=epoch,f=flag)
        x2_2 = torch.cat([att3, tmp3], 1)
        x2_2 = self.Up_conv4(x2_2)

        tmp2=self.Up3(x2_2)
        att2=self.Att3(g=tmp2,x=x1_0,epoch=epoch,f=flag)
        x1_3 = torch.cat([att2, tmp2], 1)
        x1_3 = self.Up_conv3(x1_3)

        tmp1=self.Up2(x1_3)
        att1=self.Att2(g=tmp1,x=x0_0,epoch=epoch,f=flag)
        x0_4 = torch.cat([att1, tmp1], 1)
        x0_4 = self.Up_conv2(x0_4)

        output = self.final(x0_4)
        # self.epoch+=1
        # print(epoch)
        return output

class AttentionUNet2(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.epoch = 1
        filters = [ 32, 64, 128, 256, 512, 1024]

        self.Maxpool1 = nn.MaxPool2d(2)
        self.Maxpool2 = nn.MaxPool2d(2)
        self.Maxpool3 = nn.MaxPool2d(2)
        self.Maxpool4 = nn.MaxPool2d(2)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up4 = up_conv(filters[3], filters[2])
        self.Up3 = up_conv(filters[2], filters[1])
        self.Up2 = up_conv(filters[1], filters[0])

        self.conv0_0 = Conv_Block(args.input_channels, filters[0])
        self.conv1_0 = Conv_Block(filters[0], filters[1])
        self.conv2_0 = Conv_Block(filters[1], filters[2])
        self.conv3_0 = Conv_Block(filters[2], filters[3])
        self.conv4_0 = Conv_Block(filters[3], filters[4])


        # self.final = nn.Conv2d(filters[0], 1, kernel_size=1, stride=1, padding=0)
        self.final = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            #nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.Att5 = Attention_block4(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Att4 = Attention_block4(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Att3 = Attention_block4(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Att2 = Attention_block4(F_g=filters[0], F_l=filters[0], F_int=32)

        self.Up_conv5 = Conv_Block(filters[3]+filters[3], filters[3])
        self.Up_conv4 = Conv_Block(filters[2]+filters[2], filters[2])
        self.Up_conv3 = Conv_Block(filters[1]+filters[1], filters[1])
        self.Up_conv2 = Conv_Block(filters[0]+filters[0], filters[0])

    def forward(self, input):
        # encoding path
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.Maxpool1(x0_0))
        x2_0 = self.conv2_0(self.Maxpool2(x1_0))
        x3_0 = self.conv3_0(self.Maxpool3(x2_0))
        x4_0 = self.conv4_0(self.Maxpool4(x3_0))

        # decoding + concat path
        tmp4=self.Up5(x4_0)
        att4=self.Att5(g=tmp4,x=x3_0)
        # print('att4:',att4.shape)
        # print('tmp4:',tmp4.shape)
        x3_1 = torch.cat([att4,tmp4], 1)
        x3_1 = self.Up_conv5(x3_1)

        tmp3=self.Up4(x3_1)
        att3=self.Att4(g=tmp3,x=x2_0)
        x2_2 = torch.cat([att3, tmp3], 1)
        x2_2 = self.Up_conv4(x2_2)

        tmp2=self.Up3(x2_2)
        att2=self.Att3(g=tmp2,x=x1_0)
        x1_3 = torch.cat([att2, tmp2], 1)
        x1_3 = self.Up_conv3(x1_3)

        tmp1=self.Up2(x1_3)
        att1=self.Att2(g=tmp1,x=x0_0)
        x0_4 = torch.cat([att1, tmp1], 1)
        x0_4 = self.Up_conv2(x0_4)

        output = self.final(x0_4)
        # self.epoch+=1
        # print(epoch)
        return output

class UNetPlusPlus(nn.Module):
    """
    UNet++ - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        n1 = 32
        nb_filter = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#线性插值
        self.Up5 = up_conv(nb_filter[4], nb_filter[3])
        self.Up4 = up_conv(nb_filter[3], nb_filter[2])
        self.Up3 = up_conv(nb_filter[2], nb_filter[1])
        self.Up2 = up_conv(nb_filter[1], nb_filter[0])

        self.conv0_0 = Conv_Block(args.input_channels, nb_filter[0])
        self.conv1_0 = Conv_Block(nb_filter[0], nb_filter[1])
        self.conv2_0 = Conv_Block(nb_filter[1], nb_filter[2])
        self.conv3_0 = Conv_Block(nb_filter[2], nb_filter[3])
        self.conv4_0 = Conv_Block(nb_filter[3], nb_filter[4])

        self.conv0_1 = Conv_Block(nb_filter[0]+nb_filter[0], nb_filter[0])
        self.conv1_1 = Conv_Block(nb_filter[1]+nb_filter[1], nb_filter[1])
        self.conv2_1 = Conv_Block(nb_filter[2]+nb_filter[2], nb_filter[2])
        self.conv3_1 = Conv_Block(nb_filter[3]+nb_filter[3], nb_filter[3])

        self.conv0_2 = Conv_Block(nb_filter[0]*3, nb_filter[0])
        self.conv1_2 = Conv_Block(nb_filter[1]*3, nb_filter[1])
        self.conv2_2 = Conv_Block(nb_filter[2]*3, nb_filter[2])

        self.conv0_3 = Conv_Block(nb_filter[0]*4, nb_filter[0])
        self.conv1_3 = Conv_Block(nb_filter[1]*4, nb_filter[1])

        self.conv0_4 = Conv_Block(nb_filter[0]*5, nb_filter[0])

        # if self.args.deepsupervision:
        # self.final1 = nn.Sequential(
        #     nn.Conv2d(nb_filter[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     #nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final2 = nn.Sequential(
        #     nn.Conv2d(nb_filter[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     #nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final3 = nn.Sequential(
        #     nn.Conv2d(nb_filter[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     #nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final4 = nn.Sequential(
        #     nn.Conv2d(nb_filter[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     #nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # else:
        # self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        # self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        # self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        # self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

        self.final1 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            #nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final2 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            #nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final3 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            #nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final4 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            #nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
       # self.active = torch.nn.Sigmoid()

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.Maxpool1(x0_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up2(x1_0)], 1))

        x2_0 = self.conv2_0(self.Maxpool2(x1_0))
        # print('before up3 ',x2_0.shape)#[B,128,64,64]
        # print('after up3 ',self.Up3(x2_0).shape)    #[B,64,128,128]
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up3(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up2(x1_1)], 1))

        x3_0 = self.conv3_0(self.Maxpool3(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up4(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up3(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up2(x1_2)], 1))

        x4_0 = self.conv4_0(self.Maxpool4(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up5(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up4(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up3(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up2(x1_3)], 1))

        # if self.args.deepsupervision:
        # output1 = self.final1(x0_1)
        # # print(output1[0,0,:,:])
        # output2 = self.final2(x0_2)
        # output3 = self.final3(x0_3)
        output4 = self.final4(x0_4)
        # imsave('output/coutput1.png', (output1[0,0,:,:].cpu().detach().numpy()))
        # imsave('output/coutput2.png', (output2[0,0,:,:].cpu().detach().numpy()))
        # imsave('output/coutput3.png', (output3[0,0,:,:].cpu().detach().numpy()))
        # imsave('output/coutput4.png', (output4[0,0,:,:].cpu().detach().numpy()))
        # print(output4.shape)  #[B,1,256,256]
        return output4#output1, output2, output3, output4

    # else:
        # output = self.final4(x0_4)
        # imsave('output/coutput_final.png', (output[0,0,:,:].cpu().detach().numpy()))
        # return output

class PointUNetPlusPlus(nn.Module):
    """
    UNet++ - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        n1 = 32
        nb_filter = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#线性插值
        self.Up5 = up_conv(nb_filter[4], nb_filter[3])
        self.Up4 = up_conv(nb_filter[3], nb_filter[2])
        self.Up3 = up_conv(nb_filter[2], nb_filter[1])
        self.Up2 = up_conv(nb_filter[1], nb_filter[0])

        self.conv0_0 = Conv_Block(args.input_channels, nb_filter[0])
        self.conv1_0 = Conv_Block(nb_filter[0], nb_filter[1])
        self.conv2_0 = Conv_Block(nb_filter[1], nb_filter[2])
        self.conv3_0 = Conv_Block(nb_filter[2], nb_filter[3])
        self.conv4_0 = Conv_Block(nb_filter[3], nb_filter[4])

        self.conv0_1 = Conv_Block(nb_filter[0]+nb_filter[0], nb_filter[0])
        self.conv1_1 = Conv_Block(nb_filter[1]+nb_filter[1], nb_filter[1])
        self.conv2_1 = Conv_Block(nb_filter[2]+nb_filter[2], nb_filter[2])
        self.conv3_1 = Conv_Block(nb_filter[3]+nb_filter[3], nb_filter[3])

        self.conv0_2 = Conv_Block(nb_filter[0]*3, nb_filter[0])
        self.conv1_2 = Conv_Block(nb_filter[1]*3, nb_filter[1])
        self.conv2_2 = Conv_Block(nb_filter[2]*3, nb_filter[2])

        self.conv0_3 = Conv_Block(nb_filter[0]*4, nb_filter[0])
        self.conv1_3 = Conv_Block(nb_filter[1]*4, nb_filter[1])

        self.conv0_4 = Conv_Block(nb_filter[0]*5, nb_filter[0])

        # if self.args.deepsupervision:
        # self.final1 = nn.Sequential(
        #     nn.Conv2d(nb_filter[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     #nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final2 = nn.Sequential(
        #     nn.Conv2d(nb_filter[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     #nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final3 = nn.Sequential(
        #     nn.Conv2d(nb_filter[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     #nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final4 = nn.Sequential(
        #     nn.Conv2d(nb_filter[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     #nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # else:
        # self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        # self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        # self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        # self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

        self.final1 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final2 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final3 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final4 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )

        self.head20_11 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        self.head11_02 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)

        self.head30_21 = PointHead(input_channels=nb_filter[3], out_channels=nb_filter[2],stride=4)
        self.head21_12 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        self.head12_03 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)

        self.head40_31 = PointHead(input_channels=nb_filter[4], out_channels=nb_filter[3],stride=8)
        self.head31_22 = PointHead(input_channels=nb_filter[3], out_channels=nb_filter[2],stride=4)
        self.head22_13 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        self.head13_04 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)

        self.v0 = nn.Conv2d(nb_filter[0], 1, 1, 1)
        self.v1 = nn.Conv2d(nb_filter[0], 1, 1, 1)
        self.v2 = nn.Conv2d(nb_filter[0], 1, 1, 1)
        self.v3 = nn.Conv2d(nb_filter[0], 1, 1, 1)
        self.v4 = nn.Conv2d(nb_filter[0], 1, 1, 1)
       # self.active = torch.nn.Sigmoid()

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.Maxpool1(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up2(x1_0)], 1))

        x2_0 = self.conv2_0(self.Maxpool2(x1_0))

        output1 = self.final1(x0_1)
        # imsave('output/coutput1.png', (output1[0,0,:,:].cpu().detach().numpy()))

        # result = {'res2': , 'coarse': }
        x2_0_1 = self.head20_11(x2_0, output1)
        # print('x2_0_1 shape',x2_0_1.shape)  #[B,128,128,128]
        # x2_0_1 = result.update(self.head(x2_0, output1))
        # x2_0_2 = self.upp(x2_0_1)	#通过MLP分类预测
        # print(x1_0.shape,x2_0_1.shape)
        x1_1 = self.conv1_1(torch.cat([x1_0, x2_0_1], 1))
        # x1_1 = self.conv1_1(torch.cat([x1_0, self.Up3(x2_0)], 1))
        # x1_1 = self.conv1_1(torch.cat([x1_0, self.Up3(x2_0),x2_0_1], 1))

        x1_1_1 = self.head11_02(x1_1, output1)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, x1_1_1], 1))
        # x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up2(x1_1)], 1))

        output2 = self.final2(x0_2)
        # imsave('output/coutput2.png', (output2[0,0,:,:].cpu().detach().numpy()))

        x3_0 = self.conv3_0(self.Maxpool3(x2_0))
        x3_0_1 = self.head30_21(x3_0,output2)
        x2_1 = self.conv2_1(torch.cat([x2_0, x3_0_1], 1))
        # x2_1 = self.conv2_1(torch.cat([x2_0, self.Up4(x3_0)], 1))
        x2_1_1 = self.head21_12(x2_1,output2)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, x2_1_1], 1))
        # x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up3(x2_1)], 1))
        x1_2_1 = self.head12_03(x1_2,output2)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, x1_2_1], 1))
        # x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up2(x1_2)], 1))

        output3 = self.final3(x0_3)
        # imsave('output/coutput3.png', (output3[0,0,:,:].cpu().detach().numpy()))

        x4_0 = self.conv4_0(self.Maxpool4(x3_0))
        x4_0_1 = self.head40_31(x4_0,output3)
        x3_1 = self.conv3_1(torch.cat([x3_0, x4_0_1], 1))
        # x3_1 = self.conv3_1(torch.cat([x3_0, self.Up5(x4_0)], 1))
        x3_1_1 = self.head31_22(x3_1,output3)
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, x3_1_1], 1))
        # x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up4(x3_1)], 1))
        x2_2_1 = self.head22_13(x2_2,output3)
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, x2_2_1], 1))
        # x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up3(x2_2)], 1))
        x1_3_1 = self.head13_04(x1_3,output3)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, x1_3_1], 1))
        # x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up2(x1_3)], 1))


        output4 = self.final4(x0_4)
        fv0= self.v0(x0_0)
        fv1= self.v1(x0_1)
        fv2= self.v2(x0_2)
        fv3= self.v3(x0_3)
        fv4= self.v4(x0_4)

        # imsave('feature visualization/x00.png', (fv0[0,0,:,:].cpu()))
        # imsave('feature visualization/x01.png', (fv1[0,0,:,:].cpu()))
        # imsave('feature visualization/x02.png', (fv2[0,0,:,:].cpu()))
        # imsave('feature visualization/x03.png', (fv3[0,0,:,:].cpu()))
        # imsave('feature visualization/x04.png', (fv4[0,0,:,:].cpu()))
        # imsave('output/coutput4.png', (output4[0,0,:,:].cpu().detach().numpy()))
        # print(output1[0,0,:,:])  #[B,1,256,256]
        return [output1,output2, output3, output4]#,[point_features1,point_features2,point_features3],[points1,points2,points3]

    # else:
        # output = self.final(x0_4)
        # imsave('output/coutput_final.png', (output4[0,0,:,:].cpu().detach().numpy()))

        # out = self.backbone.head(c4, c1)	#backbone is deeplabv3

        # result = {'res2': output1, 'coarse': c4}
        # result.update(self.head(x2_0, output1))
        #d1 = self.active(out)
        # if not self.training:
            # print('inference-result size',result['fine'])
        # return (output4)

        # return output

class PointHead(nn.Module):
    def __init__(self, input_channels=128, out_channels=1,stride=2, k=3, beta=0.75):        #in_c=275
        super().__init__()

        self.upp = nn.Sequential(
            # nn.Upsample(scale_factor=2),    #尺寸扩大2倍
            nn.Conv2d(input_channels, out_channels,  kernel_size=3, stride=1, padding=1, bias=True),
            # nn.ConvTranspose2d(nb_filter[2], nb_filter[1], 3),   #通道数改变,尺寸扩大2倍
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)

        )
        self.Maxpool = nn.MaxPool2d(kernel_size=stride, stride=stride)
        # self.convtrans = nn.ConvTranspose2d(input_channels, out_channels, 2, 2)
        self.k = k
        self.beta = beta
        # self.final = nn.Sequential(
        #         nn.Conv2d(num_classes, 1, 1, 1),
        #         # nn.ConvTranspose2d(1, 1, 2, 2),
        #         #nn.Upsample(scale_factor=2,mode='nearest'),
        #         nn.Sigmoid()
        #     )
    def forward(self, input, refer_ouput):    #x2_0,output1
        """
        1. Fine-grained features are interpolated from res2 for DeeplabV3
        2. During training we sample as many points as there are on a stride 16 feature map of the input
        3. To measure prediction uncertainty
           we use the same strategy during training and inference: the difference between the most
           confident and second most confident class probabilities.
        通过out作为粗糙coarse特征,输入sampling_points()得到TOPN的不确定点，针对每个点得到在精细fine特征与coarse特征中对应的特征，组合，得到feature_representation
        再将feature_representation输入mlp得到更准确的预测结果rend
        para：
        input: 输入图像的特征   []
        res2:是xception65的第一层输出 c1 [1,256,64,64]
        out：经过级联空洞卷积提取的特征，粗糙预测   [1,1,16,16]
        return：
        rend：更精准的预测结果，points：不确定点的位置
        """

        # if not self.training:
            # print('into inference!!!!!')
        return self.inference(input, refer_ouput)

        # device = out.device
		# #training
        # #获取不确定点的位置
        # # out [1,1,16,16] input [12,1,256,256] points[12,16,2]
        # points = sampling_points(out, input.shape[-1] // 8, self.k, self.beta)	#根据输入x进行点选取，转到sampling_points的inference部分，返回不确定点的坐标及索引
        # #print('input shape is ',input.shape,'points shape is',points.shape)
        # #根据不确定点的坐标，得到对应的coarse特征
        # coarse = point_sample(out, points, align_corners=False)	#coarse特征选取[12,1,16]
        # #根据不确定点的坐标，得到对应的fine特征
        # fine = point_sample(res2, points, align_corners=False)	#fine特征选取[12,256,16]
        # # print('coarse',coarse.shape,'fine',fine.shape)
        # #将二者合并
        # feature_representation = coarse #torch.cat([coarse, fine], dim=1)	#级联合并 [12，257，16]
        # print('feature_representation is ',feature_representation.shape)
        # # rend = self.final(feature_representation)	#通过MLP分类预测
        # #使用MLP进行细分预测
        # rend = self.mlp(feature_representation) #[B,1,16]

        # # print("rend size is ",rend.shape)
        # return {"rend": rend, "points": points}

    @torch.no_grad()
    def inference(self, input, refer_ouput):
        """
        During inference, subdivision uses N=8096
        (i.e., the number of points in the stride 16 map of a 1024×2048 image)
        通过out计算出前N=8096个不确定点，针对每个点针对每个点得到在精细fine特征与coarse特征中对应的特征，组合，得到rend
        再将rend输入mlp得到更准确的预测结果，一直迭代，直到rend的输出尺寸等于输入图像的尺寸
        para：
        input:本层特征xi_0  [B,in_c,,]
        out：经过级联空洞卷积提取的特征，粗糙预测output_i   [B,1,256,256]
        return：
        fine-out：上采样结果
        """
        num_points = 8096
        # print('input.shape',input.shape)
        # print('refer_ouput.shape',refer_ouput.shape)    #[B,1,256,256]
        # while input.shape[-1] != refer_ouput.shape[-1]:	#不断迭代，每次扩大2倍，直到输出out的分辨率 等于 目标x分辨率
        # print('input.shape is  ',input.shape) #[B,128,64,64]
        reduced_refer = self.Maxpool(refer_ouput) #[B,1,128,128]
        # num_points = reduced_refer.shape[-1] // 8
        points_idx, points = sampling_points(self, reduced_refer, num_points, training=self.training)	#挑选出num_points8096个最难分割的点
        # print('points_idx shape is',points_idx.shape)   #[B,128,N]

        input = F.interpolate(input, scale_factor=2, mode="bilinear", align_corners=True)	#2倍双线性插值，扩大分辨率
        # print('input.shape is  ',input.shape) #[B,128,128,128]

        coarse = point_sample(input, points, align_corners=False)	#计算coarse特征,表示8096个不确定点根据out特征（高级特征）判断的类别
        # fine = point_sample(refer_ouput, points, align_corners=False)	#计算fine特征,表示8096个不确定点根据res2特征（低级特征）判断的类别
        # print('coarse shape is',coarse.shape)   #[B,128,8096]

        # feature_representation = torch.cat([coarse, fine], dim=1)	#级联合并
        # print('feature_representation shape is',feature_representation.shape)   #[B,129,8096]
        # rend = self.mlp(coarse)	#通过MLP分类预测
        coarse[coarse>0.5]=1
        coarse[coarse<0.5]=0
        # print(coarse[0,0,:])
        # rend = self.up(feature_representation)	#通过MLP分类预测
        # rend = self.mlp(feature_representation)
        # print('rend.shape is  ',rend.shape) #[B,64,N]
        B, C, H, W = input.shape
        # _, C1, _ = rend.shape
        points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
        # print('points_idx shape is',points_idx.shape)   #[B,128,N]

        # scatter_(input, dim, index, src)将src中数据根据index中的索引按照dim的方向填进input中
        # 其实是将不确定点的特征替换掉原有特征
        # coarse : [B,128,N]
        # points_idx : [B,128,N]
        out1 = (input.reshape(B, C, -1)
                    .scatter_(2, points_idx, coarse)
                    .view(B, C, H, W))
        # imsave('output/reduced_refer-%s.png'%reduced_refer.shape[-1], (reduced_refer[0,0,:,:].cpu().detach().numpy()))

        # feature_representation = torch.cat([out1, out2], dim=1)	#级联合并
        # print('feature_representation.shape is  ',feature_representation.shape)   #[B,128,128,128]
            # rend = self.up(out)	#通过MLP分类预测
            # output = out[0]

        # output = out[0]
        # output = output.cpu().numpy()
        # for j in range(output.shape[0]):
        #     imsave('/var/www/nextcloud/data/dbc2017/files/output/'+str(j)+'.png', (output[j,:,:]))

        return self.upp(out1)#,coarse,points

class PointHead3d(nn.Module):
    def __init__(self, input_channels=128, out_channels=1,stride=2, k=3, beta=0.75):        #in_c=275
        super().__init__()

        self.upp = nn.Sequential(
            # nn.Upsample(scale_factor=2),    #尺寸扩大2倍
            nn.Conv3d(input_channels, out_channels,  kernel_size=3, stride=1, padding=1, bias=True),
            # nn.ConvTranspose2d(nb_filter[2], nb_filter[1], 3),   #通道数改变,尺寸扩大2倍
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)

        )
        self.Maxpool = nn.MaxPool3d(kernel_size=stride, stride=stride)
        # self.convtrans = nn.ConvTranspose2d(input_channels, out_channels, 2, 2)
        self.k = k
        self.beta = beta
        # self.final = nn.Sequential(
        #         nn.Conv2d(num_classes, 1, 1, 1),
        #         # nn.ConvTranspose2d(1, 1, 2, 2),
        #         #nn.Upsample(scale_factor=2,mode='nearest'),
        #         nn.Sigmoid()
        #     )
    def forward(self, input, refer_ouput):    #x2_0,output1
        """
        1. Fine-grained features are interpolated from res2 for DeeplabV3
        2. During training we sample as many points as there are on a stride 16 feature map of the input
        3. To measure prediction uncertainty
           we use the same strategy during training and inference: the difference between the most
           confident and second most confident class probabilities.
        通过out作为粗糙coarse特征,输入sampling_points()得到TOPN的不确定点，针对每个点得到在精细fine特征与coarse特征中对应的特征，组合，得到feature_representation
        再将feature_representation输入mlp得到更准确的预测结果rend
        para：
        input: 输入图像的特征   []
        res2:是xception65的第一层输出 c1 [1,256,64,64]
        out：经过级联空洞卷积提取的特征，粗糙预测   [1,1,16,16]
        return：
        rend：更精准的预测结果，points：不确定点的位置
        """
        # print(input.shape,refer_ouput.shape)
        # for i in range(input.shape[0]):
        #     Input=input[i]
        #     Refer_ouput = refer_ouput[i]
        #     Output = self.inference(Input, Refer_ouput)
        #     print(Output.shape)
        # if not self.training:
            # print('into inference!!!!!')
        return self.inference(input, refer_ouput)

        # device = out.device
		# #training
        # #获取不确定点的位置
        # # out [1,1,16,16] input [12,1,256,256] points[12,16,2]
        # points = sampling_points(out, input.shape[-1] // 8, self.k, self.beta)	#根据输入x进行点选取，转到sampling_points的inference部分，返回不确定点的坐标及索引
        # #print('input shape is ',input.shape,'points shape is',points.shape)
        # #根据不确定点的坐标，得到对应的coarse特征
        # coarse = point_sample(out, points, align_corners=False)	#coarse特征选取[12,1,16]
        # #根据不确定点的坐标，得到对应的fine特征
        # fine = point_sample(res2, points, align_corners=False)	#fine特征选取[12,256,16]
        # # print('coarse',coarse.shape,'fine',fine.shape)
        # #将二者合并
        # feature_representation = coarse #torch.cat([coarse, fine], dim=1)	#级联合并 [12，257，16]
        # print('feature_representation is ',feature_representation.shape)
        # # rend = self.final(feature_representation)	#通过MLP分类预测
        # #使用MLP进行细分预测
        # rend = self.mlp(feature_representation) #[B,1,16]

        # # print("rend size is ",rend.shape)
        # return {"rend": rend, "points": points}

    @torch.no_grad()
    def inference(self, input, refer_ouput):
        """
        During inference, subdivision uses N=8096
        (i.e., the number of points in the stride 16 map of a 1024×2048 image)
        通过out计算出前N=8096个不确定点，针对每个点针对每个点得到在精细fine特征与coarse特征中对应的特征，组合，得到rend
        再将rend输入mlp得到更准确的预测结果，一直迭代，直到rend的输出尺寸等于输入图像的尺寸
        para：
        input:本层特征xi_0  [B,in_c,,]
        out：经过级联空洞卷积提取的特征，粗糙预测output_i   [B,1,256,256]
        return：
        fine-out：上采样结果
        """
        num_points = 8096*16
        # print('input.shape',input.shape)
        # print('refer_ouput.shape',refer_ouput.shape)    #[B,1,256,256]
        # while input.shape[-1] != refer_ouput.shape[-1]:	#不断迭代，每次扩大2倍，直到输出out的分辨率 等于 目标x分辨率
        print('input.shape is  ',input.shape) #[B,128,64,64]
        reduced_refer = self.Maxpool(refer_ouput) #[B,1,128,128]
        print('reduced_refer.shape is  ',reduced_refer.shape) #[B,128,64,64]

        # reduced_refer = reduced_Refer[:,:,i,:,:]
        # num_points = reduced_refer.shape[-1] // 8
        points_idx, points = sampling_points_3D(self, reduced_refer, num_points, training=self.training)	#挑选出num_points8096个最难分割的点
        # print('points_idx shape is',points_idx.shape)   #[B,128,N]

        input = F.interpolate(input, scale_factor=2, mode="bilinear", align_corners=True)	#2倍双线性插值，扩大分辨率
        # print('input.shape is  ',input.shape) #[B,128,128,128]

        coarse = point_sample(input, points, align_corners=False)	#计算coarse特征,表示8096个不确定点根据out特征（高级特征）判断的类别
        # fine = point_sample(refer_ouput, points, align_corners=False)	#计算fine特征,表示8096个不确定点根据res2特征（低级特征）判断的类别
        # print('coarse shape is',coarse.shape)   #[B,128,8096]

        # feature_representation = torch.cat([coarse, fine], dim=1)	#级联合并
        # print('feature_representation shape is',feature_representation.shape)   #[B,129,8096]
        # rend = self.mlp(coarse)	#通过MLP分类预测
        coarse[coarse>0.5]=1
        coarse[coarse<0.5]=0
        # print(coarse[0,0,:])
        # rend = self.up(feature_representation)	#通过MLP分类预测
        # rend = self.mlp(feature_representation)
        # print('rend.shape is  ',rend.shape) #[B,64,N]
        B, C, L ,H, W = input.shape
        # _, C1, _ = rend.shape
        points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
        # print('points_idx shape is',points_idx.shape)   #[B,128,N]

        # scatter_(input, dim, index, src)将src中数据根据index中的索引按照dim的方向填进input中
        # 其实是将不确定点的特征替换掉原有特征
        # coarse : [B,128,N]
        # points_idx : [B,128,N]
        out1 = (input.reshape(B, C, -1)
                    .scatter_(2, points_idx, coarse)
                    .view(B, C, H, W))
        # imsave('output/reduced_refer-%s.png'%reduced_refer.shape[-1], (reduced_refer[0,0,:,:].cpu().detach().numpy()))

        # feature_representation = torch.cat([out1, out2], dim=1)	#级联合并
        # print('feature_representation.shape is  ',feature_representation.shape)   #[B,128,128,128]
            # rend = self.up(out)	#通过MLP分类预测
            # output = out[0]

        # output = out[0]
        # output = output.cpu().numpy()
        # for j in range(output.shape[0]):
        #     imsave('/var/www/nextcloud/data/dbc2017/files/output/'+str(j)+'.png', (output[j,:,:]))

        return self.upp(out1)#,coarse,points

class PointUNetPlusPlus2(nn.Module):
    """
    UNet++ - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        n1 = 32
        nb_filter = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#线性插值
        self.Up5 = up_conv(nb_filter[4], nb_filter[3])
        self.Up4 = up_conv(nb_filter[3], nb_filter[2])
        self.Up3 = up_conv(nb_filter[2], nb_filter[1])
        self.Up2 = up_conv(nb_filter[1], nb_filter[0])

        self.conv0_0 = Conv_Block(args.input_channels, nb_filter[0])
        self.conv1_0 = Conv_Block(nb_filter[0], nb_filter[1])
        self.conv2_0 = Conv_Block(nb_filter[1], nb_filter[2])
        self.conv3_0 = Conv_Block(nb_filter[2], nb_filter[3])
        self.conv4_0 = Conv_Block(nb_filter[3], nb_filter[4])

        self.conv0_1 = Conv_Block(nb_filter[0]+nb_filter[0], nb_filter[0])
        self.conv1_1 = Conv_Block(nb_filter[1]*3, nb_filter[1])
        self.conv2_1 = Conv_Block(nb_filter[2]*3, nb_filter[2])
        self.conv3_1 = Conv_Block(nb_filter[3]*3, nb_filter[3])

        self.conv0_2 = Conv_Block(nb_filter[0]*4, nb_filter[0])
        self.conv1_2 = Conv_Block(nb_filter[1]*4, nb_filter[1])
        self.conv2_2 = Conv_Block(nb_filter[2]*4, nb_filter[2])

        self.conv0_3 = Conv_Block(nb_filter[0]*5, nb_filter[0])
        self.conv1_3 = Conv_Block(nb_filter[1]*5, nb_filter[1])

        self.conv0_4 = Conv_Block(nb_filter[0]*6, nb_filter[0])

        # if self.args.deepsupervision:
        # self.final1 = nn.Sequential(
        #     nn.Conv2d(nb_filter[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     #nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final2 = nn.Sequential(
        #     nn.Conv2d(nb_filter[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     #nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final3 = nn.Sequential(
        #     nn.Conv2d(nb_filter[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     #nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final4 = nn.Sequential(
        #     nn.Conv2d(nb_filter[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     #nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # else:
        # self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        # self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        # self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        # self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

        self.final1 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final2 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final3 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final4 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )

        self.head20_11 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        self.head11_02 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)

        self.head30_21 = PointHead(input_channels=nb_filter[3], out_channels=nb_filter[2],stride=4)
        self.head21_12 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        self.head12_03 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)

        self.head40_31 = PointHead(input_channels=nb_filter[4], out_channels=nb_filter[3],stride=8)
        self.head31_22 = PointHead(input_channels=nb_filter[3], out_channels=nb_filter[2],stride=4)
        self.head22_13 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        self.head13_04 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)
        # self.v0 = nn.Conv2d(nb_filter[0], 1, 1, 1)
        # self.v1 = nn.Conv2d(nb_filter[0], 1, 1, 1)
        # self.v2 = nn.Conv2d(nb_filter[0], 1, 1, 1)
        # self.v3 = nn.Conv2d(nb_filter[0], 1, 1, 1)
        # self.v4 = nn.Conv2d(nb_filter[0], 1, 1, 1)
       # self.active = torch.nn.Sigmoid()

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.Maxpool1(x0_0))    #[B,64,128,128]
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up2(x1_0)], 1))

        x2_0 = self.conv2_0(self.Maxpool2(x1_0))

        output1 = self.final1(x0_1)
        # imsave('output/coutput1.png', (output1[0,0,:,:].cpu().detach().numpy()))

        # result = {'res2': , 'coarse': }
        x2_0_1 = self.head20_11(x2_0, output1)
        # print('x2_0_1 shape',x2_0_1.shape)  #[B,128,128,128]
        # x2_0_1 = result.update(self.head(x2_0, output1))
        # x2_0_2 = self.upp(x2_0_1)	#通过MLP分类预测
        # print(x1_0.shape,self.Up3(x2_0).shape,x2_0_1.shape)
        # x1_1 = self.conv1_1(torch.cat([x1_0, x2_0_1], 1))
        # x1_1 = self.conv1_1(torch.cat([x1_0, self.Up3(x2_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up3(x2_0),x2_0_1], 1))

        x1_1_1 = self.head11_02(x1_1, output1)
        # x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, x1_1_1], 1))
        # x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up2(x1_1)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up2(x1_1),x1_1_1], 1))

        output2 = self.final2(x0_2)
        # imsave('output/coutput2.png', (output2[0,0,:,:].cpu().detach().numpy()))

        x3_0 = self.conv3_0(self.Maxpool3(x2_0))
        x3_0_1 = self.head30_21(x3_0,output2)
        # x2_1 = self.conv2_1(torch.cat([x2_0, x3_0_1], 1))
        # x2_1 = self.conv2_1(torch.cat([x2_0, self.Up4(x3_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up4(x3_0),x3_0_1], 1))
        x2_1_1 = self.head21_12(x2_1,output2)
        # x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, x2_1_1], 1))
        # x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up3(x2_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up3(x2_1),x2_1_1], 1))
        x1_2_1 = self.head12_03(x1_2,output2)
        # x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, x1_2_1], 1))
        # x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up2(x1_2)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up2(x1_2),x1_2_1], 1))

        output3 = self.final3(x0_3)
        # imsave('output/coutput3.png', (output3[0,0,:,:].cpu().detach().numpy()))

        x4_0 = self.conv4_0(self.Maxpool4(x3_0))
        x4_0_1 = self.head40_31(x4_0,output3)
        # x3_1 = self.conv3_1(torch.cat([x3_0, x4_0_1], 1))
        # x3_1 = self.conv3_1(torch.cat([x3_0, self.Up5(x4_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up5(x4_0),x4_0_1], 1))
        x3_1_1 = self.head31_22(x3_1,output3)
        # x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, x3_1_1], 1))
        # x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up4(x3_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up4(x3_1),x3_1_1], 1))
        x2_2_1 = self.head22_13(x2_2,output3)
        # x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, x2_2_1], 1))
        # x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up3(x2_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up3(x2_2),x2_2_1], 1))
        x1_3_1 = self.head13_04(x1_3,output3)
        # x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, x1_3_1], 1))
        # x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up2(x1_3)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up2(x1_3),x1_3_1], 1))


        output4 = self.final4(x0_4)
        # fv0= self.v0(x0_0)
        # fv1= self.v1(x0_1)
        # fv2= self.v2(x0_2)
        # fv3= self.v3(x0_3)
        # fv4= self.v4(x0_4)
        # imsave('feature visualization/x00.png', (fv0[0,0,:,:].cpu()))
        # imsave('feature visualization/x01.png', (fv1[0,0,:,:].cpu()))
        # imsave('feature visualization/x02.png', (fv2[0,0,:,:].cpu()))
        # imsave('feature visualization/x03.png', (fv3[0,0,:,:].cpu()))
        # imsave('feature visualization/x04.png', (fv4[0,0,:,:].cpu()))
        # imsave('output/coutput4.png', (output4[0,0,:,:].cpu().detach().numpy()))
        # print(output1[0,0,:,:])  #[B,1,256,256]
        return [output1,output2, output3, output4]#,[point_features1,point_features2,point_features3],[points1,points2,points3]

    # else:
        # output = self.final(x0_4)
        # imsave('output/coutput_final.png', (output4[0,0,:,:].cpu().detach().numpy()))

        # out = self.backbone.head(c4, c1)	#backbone is deeplabv3

        # result = {'res2': output1, 'coarse': c4}
        # result.update(self.head(x2_0, output1))
        #d1 = self.active(out)
        # if not self.training:
            # print('inference-result size',result['fine'])
        # return (output4)

        # return output

class AttentionPointUNetPlusPlus(nn.Module):
    # without CGM,with ds
    def __init__(self, args):
        super().__init__()

        self.args = args

        nb_filter = [32,64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Up5 = up_conv(nb_filter[4], nb_filter[3])
        self.Up4 = up_conv(nb_filter[3], nb_filter[2])
        self.Up3 = up_conv(nb_filter[2], nb_filter[1])
        self.Up2 = up_conv(nb_filter[1], nb_filter[0])

        self.conv0_0 = Conv_Block(args.input_channels, nb_filter[0])
        self.conv1_0 = Conv_Block(nb_filter[0], nb_filter[1])
        self.conv2_0 = Conv_Block(nb_filter[1], nb_filter[2])
        self.conv3_0 = Conv_Block(nb_filter[2], nb_filter[3])
        self.conv4_0 = Conv_Block(nb_filter[3], nb_filter[4])

        self.conv0_1 = Conv_Block(nb_filter[0]+nb_filter[0], nb_filter[0])
        self.conv1_1 = Conv_Block(nb_filter[1]*3, nb_filter[1])
        self.conv2_1 = Conv_Block(nb_filter[2]*3, nb_filter[2])
        self.conv3_1 = Conv_Block(nb_filter[3]*3, nb_filter[3])

        self.conv0_2 = Conv_Block(nb_filter[0]*4, nb_filter[0])
        self.conv1_2 = Conv_Block(nb_filter[1]*4, nb_filter[1])
        self.conv2_2 = Conv_Block(nb_filter[2]*4, nb_filter[2])

        self.conv0_3 = Conv_Block(nb_filter[0]*5, nb_filter[0])
        self.conv1_3 = Conv_Block(nb_filter[1]*5, nb_filter[1])

        self.conv0_4 = Conv_Block(nb_filter[0]*6, nb_filter[0])

        # self.Att4_2 = Attention_block(F_g=nb_filter[3], F_l=nb_filter[2]*2, F_int=nb_filter[1])
        # self.Att3_3 = Attention_block(F_g=nb_filter[2], F_l=nb_filter[1]*3, F_int=nb_filter[0])
        # self.Att2_4 = Attention_block(F_g=nb_filter[1], F_l=nb_filter[0]*4, F_int=32)

        self.Att0_1 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_1 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Att2_1 = Attention_block4(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])
        self.Att3_1 = Attention_block4(F_g=nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[2])

        self.Att0_2 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_2 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Att2_2 = Attention_block4(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])

        self.Att0_3 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_3 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])

        self.Att0_4 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)

        self.head20_11 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        self.head11_02 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)

        self.head30_21 = PointHead(input_channels=nb_filter[3], out_channels=nb_filter[2],stride=4)
        self.head21_12 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        self.head12_03 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)

        self.head40_31 = PointHead(input_channels=nb_filter[4], out_channels=nb_filter[3],stride=8)
        self.head31_22 = PointHead(input_channels=nb_filter[3], out_channels=nb_filter[2],stride=4)
        self.head22_13 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        self.head13_04 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)
        # self.Up_conv5 = Conv_Block(nb_filter[4], nb_filters[3])
        # self.Up_conv4 = Conv_Block(nb_filter[3], nb_filters[2])
        # self.Up_conv3 = Conv_Block(nb_filter[2], nb_filters[1])
        # self.Up_conv2 = Conv_Block(nb_filters[1], nb_filters[0])

        self.final1 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final2 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final3 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final4 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.v0 = nn.Conv2d(nb_filter[0], 1, 1, 1)
        self.v1 = nn.Conv2d(nb_filter[0], 1, 1, 1)
        self.v2 = nn.Conv2d(nb_filter[0], 1, 1, 1)
        self.v3 = nn.Conv2d(nb_filter[0], 1, 1, 1)
        self.v4 = nn.Conv2d(nb_filter[0], 1, 1, 1)
    def forward(self, input):
        # encoding path
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        tmp0_1 = self.Up2(x1_0)
        att0_1=self.Att0_1(g=tmp0_1,x=x0_0)
        x0_1 = torch.cat([att0_1,tmp0_1], 1)
        x0_1 = self.conv0_1(x0_1)

        output1 = self.final1(x0_1)

        x2_0 = self.conv2_0(self.pool(x1_0))
        # print('x2_0 shape:',x2_0.shape) #[B,128,64,64]
        x2_0_1 = self.head20_11(x2_0, output1)

        tmp1_1 = self.Up3(x2_0)
        att1_1=self.Att1_1(g=x2_0_1,x=x1_0) ####
        x1_1 = torch.cat([att1_1,tmp1_1,x2_0_1], 1)
        x1_1 = self.conv1_1(x1_1)

        x1_1_1 = self.head11_02(x1_1, output1)
        tmp0_2 = self.Up2(x1_1)
        att0_2=self.Att0_2(g=x1_1_1,x=x0_1)
        x0_2 = torch.cat([x0_0,att0_2,tmp0_2,x1_1_1], 1)
        x0_2 = self.conv0_2(x0_2)

        output2 = self.final2(x0_2)

        x3_0 = self.conv3_0(self.pool(x2_0))
        x3_0_1 = self.head30_21(x3_0,output2)
        tmp2_1 = self.Up4(x3_0)
        att2_1=self.Att2_1(g=x3_0_1,x=x2_0)
        x2_1 = torch.cat([att2_1,tmp2_1,x3_0_1], 1)
        x2_1 = self.conv2_1(x2_1)
        x2_1_1 = self.head21_12(x2_1,output2)

        tmp1_2 = self.Up3(x2_1)
        att1_2=self.Att1_2(g=x2_1_1,x=x1_1)
        x1_2 = torch.cat([x1_0,att1_2,tmp1_2,x2_1_1], 1)
        x1_2 = self.conv1_2(x1_2)
        x1_2_1 = self.head12_03(x1_2,output2)
        tmp0_3 = self.Up2(x1_2)
        att0_3=self.Att0_3(g=x1_2_1,x=x0_2)
        x0_3 = torch.cat([x0_0,x0_1,att0_3,tmp0_3,x1_2_1], 1)
        x0_3 = self.conv0_3(x0_3)

        output3 = self.final3(x0_3)

        x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0_1 = self.head40_31(x4_0,output3)
        tmp3_1=self.Up5(x4_0)
        att3_1=self.Att3_1(g=x4_0_1,x=x3_0)
        x3_1 = torch.cat([att3_1,tmp3_1,x4_0_1], 1)
        x3_1 = self.conv3_1(x3_1)
        x3_1_1 = self.head31_22(x3_1,output3)
        tmp2_2=self.Up4(x3_1)
        att2_2=self.Att2_2(g=x3_1_1,x=x2_1)
        x2_2 = torch.cat([x2_0,att2_2, tmp2_2, x3_1_1], 1)
        x2_2 = self.conv2_2(x2_2)
        x2_2_1 = self.head22_13(x2_2,output3)
        tmp1_3=self.Up3(x2_2)
        att1_3=self.Att1_3(g=x2_2_1,x=x1_2)
        x1_3 = torch.cat([x1_0 ,x1_1,att1_3, tmp1_3 ,x2_2_1], 1)
        x1_3 = self.conv1_3(x1_3)
        x1_3_1 = self.head13_04(x1_3,output3)
        tmp0_4=self.Up2(x1_3)
        att0_4 = self.Att0_4(g=x1_3_1,x=x0_3)
        x0_4 = torch.cat([x0_1, x0_2,att0_4, tmp0_4,x0_0, x1_3_1], 1)
        x0_4 = self.conv0_4(x0_4)

        output4 = self.final4(x0_4)

        # fv0= self.v0(x0_0)
        # fv1= self.v1(x0_1)
        # fv2= self.v2(x0_2)
        # fv3= self.v3(x0_3)
        # fv4= self.v4(x0_4)
        # # print('x0_4.shape',x0_4.shape)
        # imsave('feature visualization/x00.png', (fv0[0,0,:,:].cpu()))
        # imsave('feature visualization/x01.png', (fv1[0,0,:,:].cpu()))
        # imsave('feature visualization/x02.png', (fv2[0,0,:,:].cpu()))
        # imsave('feature visualization/x03.png', (fv3[0,0,:,:].cpu()))
        # imsave('feature visualization/x04.png', (fv4[0,0,:,:].cpu()))
        return [output1, output2, output3, output4]

        # else:
        #     output = self.final(x0_4)
        #     return output

class AttentionPointUNetPlusPlus2(nn.Module):
    #with CGM,ds
    def __init__(self, args):
        super().__init__()

        self.args = args

        nb_filter = [32,64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Up5 = up_conv(nb_filter[4], nb_filter[3])
        self.Up4 = up_conv(nb_filter[3], nb_filter[2])
        self.Up3 = up_conv(nb_filter[2], nb_filter[1])
        self.Up2 = up_conv(nb_filter[1], nb_filter[0])

        self.conv0_0 = Conv_Block(args.input_channels, nb_filter[0])
        self.conv1_0 = Conv_Block(nb_filter[0], nb_filter[1])
        self.conv2_0 = Conv_Block(nb_filter[1], nb_filter[2])
        self.conv3_0 = Conv_Block(nb_filter[2], nb_filter[3])
        self.conv4_0 = Conv_Block(nb_filter[3], nb_filter[4])

        self.conv0_1 = Conv_Block(nb_filter[0]+nb_filter[0], nb_filter[0])
        self.conv1_1 = Conv_Block(nb_filter[1]*3, nb_filter[1])
        self.conv2_1 = Conv_Block(nb_filter[2]*3, nb_filter[2])
        self.conv3_1 = Conv_Block(nb_filter[3]*3, nb_filter[3])

        self.conv0_2 = Conv_Block(nb_filter[0]*4, nb_filter[0])
        self.conv1_2 = Conv_Block(nb_filter[1]*4, nb_filter[1])
        self.conv2_2 = Conv_Block(nb_filter[2]*4, nb_filter[2])

        self.conv0_3 = Conv_Block(nb_filter[0]*5, nb_filter[0])
        self.conv1_3 = Conv_Block(nb_filter[1]*5, nb_filter[1])

        self.conv0_4 = Conv_Block(nb_filter[0]*6, nb_filter[0])

        # self.Att4_2 = Attention_block(F_g=nb_filter[3], F_l=nb_filter[2]*2, F_int=nb_filter[1])
        # self.Att3_3 = Attention_block(F_g=nb_filter[2], F_l=nb_filter[1]*3, F_int=nb_filter[0])
        # self.Att2_4 = Attention_block(F_g=nb_filter[1], F_l=nb_filter[0]*4, F_int=32)

        self.Att0_1 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_1 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Att2_1 = Attention_block4(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])
        self.Att3_1 = Attention_block4(F_g=nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[2])

        self.Att0_2 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_2 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Att2_2 = Attention_block4(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])

        self.Att0_3 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_3 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])

        self.Att0_4 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)

        self.head20_11 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        self.head11_02 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)

        self.head30_21 = PointHead(input_channels=nb_filter[3], out_channels=nb_filter[2],stride=4)
        self.head21_12 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        self.head12_03 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)

        self.head40_31 = PointHead(input_channels=nb_filter[4], out_channels=nb_filter[3],stride=8)
        self.head31_22 = PointHead(input_channels=nb_filter[3], out_channels=nb_filter[2],stride=4)
        self.head22_13 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        self.head13_04 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)
        # self.Up_conv5 = Conv_Block(nb_filters[4], nb_filters[3])
        # self.Up_conv4 = Conv_Block(nb_filters[3], nb_filters[2])
        # self.Up_conv3 = Conv_Block(nb_filters[2], nb_filters[1])
        # self.Up_conv2 = Conv_Block(nb_filters[1], nb_filters[0])

        self.final1 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final2 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final3 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final4 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.cls = nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Conv2d(nb_filter[4], 2, 1),
                    nn.AdaptiveMaxPool2d(1),
                    nn.Sigmoid())

    def dotProduct(self,seg,cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final

    def forward(self, input):
        # encoding path
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        tmp0_1 = self.Up2(x1_0)
        att0_1=self.Att0_1(g=tmp0_1,x=x0_0)
        x0_1 = torch.cat([att0_1,tmp0_1], 1)
        x0_1 = self.conv0_1(x0_1)

        output1 = self.final1(x0_1)

        x2_0 = self.conv2_0(self.pool(x1_0))
        # print('x2_0 shape:',x2_0.shape) #[B,128,64,64]
        x2_0_1 = self.head20_11(x2_0, output1)

        tmp1_1 = self.Up3(x2_0)
        att1_1=self.Att1_1(g=x2_0_1,x=x1_0) ####
        x1_1 = torch.cat([att1_1,tmp1_1,x2_0_1], 1)
        x1_1 = self.conv1_1(x1_1)

        x1_1_1 = self.head11_02(x1_1, output1)
        tmp0_2 = self.Up2(x1_1)
        att0_2=self.Att0_2(g=x1_1_1,x=x0_1)
        x0_2 = torch.cat([x0_0,att0_2,tmp0_2,x1_1_1], 1)
        x0_2 = self.conv0_2(x0_2)

        output2 = self.final2(x0_2)

        x3_0 = self.conv3_0(self.pool(x2_0))
        x3_0_1 = self.head30_21(x3_0,output2)
        tmp2_1 = self.Up4(x3_0)
        att2_1=self.Att2_1(g=x3_0_1,x=x2_0)
        x2_1 = torch.cat([att2_1,tmp2_1,x3_0_1], 1)
        x2_1 = self.conv2_1(x2_1)
        x2_1_1 = self.head21_12(x2_1,output2)

        tmp1_2 = self.Up3(x2_1)
        att1_2=self.Att1_2(g=x2_1_1,x=x1_1)
        x1_2 = torch.cat([x1_0,att1_2,tmp1_2,x2_1_1], 1)
        x1_2 = self.conv1_2(x1_2)
        x1_2_1 = self.head12_03(x1_2,output2)
        tmp0_3 = self.Up2(x1_2)
        att0_3=self.Att0_3(g=x1_2_1,x=x0_2)
        x0_3 = torch.cat([x0_0,x0_1,att0_3,tmp0_3,x1_2_1], 1)
        x0_3 = self.conv0_3(x0_3)

        output3 = self.final3(x0_3)

        x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0_1 = self.head40_31(x4_0,output3)
        tmp3_1=self.Up5(x4_0)
        att3_1=self.Att3_1(g=x4_0_1,x=x3_0)
        x3_1 = torch.cat([att3_1,tmp3_1,x4_0_1], 1)
        x3_1 = self.conv3_1(x3_1)
        x3_1_1 = self.head31_22(x3_1,output3)
        tmp2_2=self.Up4(x3_1)
        att2_2=self.Att2_2(g=x3_1_1,x=x2_1)
        x2_2 = torch.cat([x2_0,att2_2, tmp2_2, x3_1_1], 1)
        x2_2 = self.conv2_2(x2_2)
        x2_2_1 = self.head22_13(x2_2,output3)
        tmp1_3=self.Up3(x2_2)
        att1_3=self.Att1_3(g=x2_2_1,x=x1_2)
        x1_3 = torch.cat([x1_0 ,x1_1,att1_3, tmp1_3 ,x2_2_1], 1)
        x1_3 = self.conv1_3(x1_3)
        x1_3_1 = self.head13_04(x1_3,output3)
        tmp0_4=self.Up2(x1_3)
        att0_4 = self.Att0_4(g=x1_3_1,x=x0_3)
        x0_4 = torch.cat([x0_1, x0_2,att0_4, tmp0_4,x0_0, x1_3_1], 1)
        x0_4 = self.conv0_4(x0_4)

        output4 = self.final4(x0_4)

        # -------------Classification-------------
        # print('x4_0 shape',x4_0.shape)  #[B,512,16,16]
        cls_branch = self.cls(x4_0).squeeze(3).squeeze(2)  # (B,N,1,1)->(B,N)
        # print('cls_branch is',cls_branch)
        cls_branch_max = cls_branch.argmax(dim=1)
        cls_branch_max = cls_branch_max[:, np.newaxis].float()
        # print('cls_branch_max is',cls_branch_max)   #[B]
        output1 = self.dotProduct(output1, cls_branch_max)
        output2 = self.dotProduct(output2, cls_branch_max)
        output3 = self.dotProduct(output3, cls_branch_max)
        output4 = self.dotProduct(output4, cls_branch_max)
        return [output1, output2, output3, output4]

        # else:
        #     output = self.final(x0_4)
        #     return output

class AttentionPointUNetPlusPlus3(nn.Module):
    # without CGM , ds
    def __init__(self, args):
        super().__init__()

        self.args = args

        nb_filter = [32,64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Up5 = up_conv(nb_filter[4], nb_filter[3])
        self.Up4 = up_conv(nb_filter[3], nb_filter[2])
        self.Up3 = up_conv(nb_filter[2], nb_filter[1])
        self.Up2 = up_conv(nb_filter[1], nb_filter[0])

        self.conv0_0 = Conv_Block(args.input_channels, nb_filter[0])
        self.conv1_0 = Conv_Block(nb_filter[0], nb_filter[1])
        self.conv2_0 = Conv_Block(nb_filter[1], nb_filter[2])
        self.conv3_0 = Conv_Block(nb_filter[2], nb_filter[3])
        self.conv4_0 = Conv_Block(nb_filter[3], nb_filter[4])

        self.conv0_1 = Conv_Block(nb_filter[0]+nb_filter[0], nb_filter[0])
        self.conv1_1 = Conv_Block(nb_filter[1]*3, nb_filter[1])
        self.conv2_1 = Conv_Block(nb_filter[2]*3, nb_filter[2])
        self.conv3_1 = Conv_Block(nb_filter[3]*3, nb_filter[3])

        self.conv0_2 = Conv_Block(nb_filter[0]*4, nb_filter[0])
        self.conv1_2 = Conv_Block(nb_filter[1]*4, nb_filter[1])
        self.conv2_2 = Conv_Block(nb_filter[2]*4, nb_filter[2])

        self.conv0_3 = Conv_Block(nb_filter[0]*5, nb_filter[0])
        self.conv1_3 = Conv_Block(nb_filter[1]*5, nb_filter[1])

        self.conv0_4 = Conv_Block(nb_filter[0]*6, nb_filter[0])

        # self.Att4_2 = Attention_block(F_g=nb_filter[3], F_l=nb_filter[2]*2, F_int=nb_filter[1])
        # self.Att3_3 = Attention_block(F_g=nb_filter[2], F_l=nb_filter[1]*3, F_int=nb_filter[0])
        # self.Att2_4 = Attention_block(F_g=nb_filter[1], F_l=nb_filter[0]*4, F_int=32)

        self.Att0_1 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_1 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Att2_1 = Attention_block4(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])
        self.Att3_1 = Attention_block4(F_g=nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[2])

        self.Att0_2 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_2 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Att2_2 = Attention_block4(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])

        self.Att0_3 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_3 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])

        self.Att0_4 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)

        self.head20_11 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        self.head11_02 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)

        self.head30_21 = PointHead(input_channels=nb_filter[3], out_channels=nb_filter[2],stride=4)
        self.head21_12 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        self.head12_03 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)

        self.head40_31 = PointHead(input_channels=nb_filter[4], out_channels=nb_filter[3],stride=8)
        self.head31_22 = PointHead(input_channels=nb_filter[3], out_channels=nb_filter[2],stride=4)
        self.head22_13 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        self.head13_04 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)
        # self.Up_conv5 = Conv_Block(nb_filters[4], nb_filters[3])
        # self.Up_conv4 = Conv_Block(nb_filters[3], nb_filters[2])
        # self.Up_conv3 = Conv_Block(nb_filters[2], nb_filters[1])
        # self.Up_conv2 = Conv_Block(nb_filters[1], nb_filters[0])

        self.final1 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final2 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final3 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final4 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )

    def forward(self, input):
        # encoding path
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        tmp0_1 = self.Up2(x1_0)
        att0_1=self.Att0_1(g=tmp0_1,x=x0_0)
        x0_1 = torch.cat([att0_1,tmp0_1], 1)
        x0_1 = self.conv0_1(x0_1)

        output1 = self.final1(x0_1)

        x2_0 = self.conv2_0(self.pool(x1_0))
        # print('x2_0 shape:',x2_0.shape) #[B,128,64,64]
        x2_0_1 = self.head20_11(x2_0, output1)

        tmp1_1 = self.Up3(x2_0)
        att1_1=self.Att1_1(g=x2_0_1,x=x1_0) ####
        x1_1 = torch.cat([att1_1,tmp1_1,x2_0_1], 1)
        x1_1 = self.conv1_1(x1_1)

        x1_1_1 = self.head11_02(x1_1, output1)
        tmp0_2 = self.Up2(x1_1)
        att0_2=self.Att0_2(g=x1_1_1,x=x0_1)
        x0_2 = torch.cat([x0_0,att0_2,tmp0_2,x1_1_1], 1)
        x0_2 = self.conv0_2(x0_2)

        output2 = self.final2(x0_2)

        x3_0 = self.conv3_0(self.pool(x2_0))
        x3_0_1 = self.head30_21(x3_0,output2)
        tmp2_1 = self.Up4(x3_0)
        att2_1=self.Att2_1(g=x3_0_1,x=x2_0)
        x2_1 = torch.cat([att2_1,tmp2_1,x3_0_1], 1)
        x2_1 = self.conv2_1(x2_1)
        x2_1_1 = self.head21_12(x2_1,output2)

        tmp1_2 = self.Up3(x2_1)
        att1_2=self.Att1_2(g=x2_1_1,x=x1_1)
        x1_2 = torch.cat([x1_0,att1_2,tmp1_2,x2_1_1], 1)
        x1_2 = self.conv1_2(x1_2)
        x1_2_1 = self.head12_03(x1_2,output2)
        tmp0_3 = self.Up2(x1_2)
        att0_3=self.Att0_3(g=x1_2_1,x=x0_2)
        x0_3 = torch.cat([x0_0,x0_1,att0_3,tmp0_3,x1_2_1], 1)
        x0_3 = self.conv0_3(x0_3)

        output3 = self.final3(x0_3)

        x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0_1 = self.head40_31(x4_0,output3)
        tmp3_1=self.Up5(x4_0)
        att3_1=self.Att3_1(g=x4_0_1,x=x3_0)
        x3_1 = torch.cat([att3_1,tmp3_1,x4_0_1], 1)
        x3_1 = self.conv3_1(x3_1)
        x3_1_1 = self.head31_22(x3_1,output3)
        tmp2_2=self.Up4(x3_1)
        att2_2=self.Att2_2(g=x3_1_1,x=x2_1)
        x2_2 = torch.cat([x2_0,att2_2, tmp2_2, x3_1_1], 1)
        x2_2 = self.conv2_2(x2_2)
        x2_2_1 = self.head22_13(x2_2,output3)
        tmp1_3=self.Up3(x2_2)
        att1_3=self.Att1_3(g=x2_2_1,x=x1_2)
        x1_3 = torch.cat([x1_0 ,x1_1,att1_3, tmp1_3 ,x2_2_1], 1)
        x1_3 = self.conv1_3(x1_3)
        x1_3_1 = self.head13_04(x1_3,output3)
        tmp0_4=self.Up2(x1_3)
        att0_4 = self.Att0_4(g=x1_3_1,x=x0_3)
        x0_4 = torch.cat([x0_1, x0_2,att0_4, tmp0_4,x0_0, x1_3_1], 1)
        x0_4 = self.conv0_4(x0_4)

        output4 = self.final4(x0_4)
        return output4

        # else:
        #     output = self.final(x0_4)
        #     return output

class AttentionPointUNetPlusPlusL3(nn.Module):
    # without CGM,with ds
    def __init__(self, args):
        super().__init__()

        self.args = args

        nb_filter = [32,64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        # self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.Up5 = up_conv(nb_filter[4], nb_filter[3])
        self.Up4 = up_conv(nb_filter[3], nb_filter[2])
        self.Up3 = up_conv(nb_filter[2], nb_filter[1])
        self.Up2 = up_conv(nb_filter[1], nb_filter[0])

        self.conv0_0 = Conv_Block(args.input_channels, nb_filter[0])
        self.conv1_0 = Conv_Block(nb_filter[0], nb_filter[1])
        self.conv2_0 = Conv_Block(nb_filter[1], nb_filter[2])
        self.conv3_0 = Conv_Block(nb_filter[2], nb_filter[3])
        # self.conv4_0 = Conv_Block(nb_filter[3], nb_filter[4])

        self.conv0_1 = Conv_Block(nb_filter[0]+nb_filter[0], nb_filter[0])
        self.conv1_1 = Conv_Block(nb_filter[1]*3, nb_filter[1])
        self.conv2_1 = Conv_Block(nb_filter[2]*3, nb_filter[2])
        # self.conv3_1 = Conv_Block(nb_filter[3]*3, nb_filter[3])

        self.conv0_2 = Conv_Block(nb_filter[0]*4, nb_filter[0])
        self.conv1_2 = Conv_Block(nb_filter[1]*4, nb_filter[1])
        # self.conv2_2 = Conv_Block(nb_filter[2]*4, nb_filter[2])

        self.conv0_3 = Conv_Block(nb_filter[0]*5, nb_filter[0])
        # self.conv1_3 = Conv_Block(nb_filter[1]*5, nb_filter[1])

        # self.conv0_4 = Conv_Block(nb_filter[0]*6, nb_filter[0])

        # self.Att4_2 = Attention_block(F_g=nb_filter[3], F_l=nb_filter[2]*2, F_int=nb_filter[1])
        # self.Att3_3 = Attention_block(F_g=nb_filter[2], F_l=nb_filter[1]*3, F_int=nb_filter[0])
        # self.Att2_4 = Attention_block(F_g=nb_filter[1], F_l=nb_filter[0]*4, F_int=32)

        self.Att0_1 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_1 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Att2_1 = Attention_block4(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])
        # self.Att3_1 = Attention_block4(F_g=nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[2])

        self.Att0_2 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_2 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        # self.Att2_2 = Attention_block4(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])

        self.Att0_3 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        # self.Att1_3 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])

        # self.Att0_4 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)

        self.head20_11 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        self.head11_02 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)

        self.head30_21 = PointHead(input_channels=nb_filter[3], out_channels=nb_filter[2],stride=4)
        self.head21_12 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        self.head12_03 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)

        # self.head40_31 = PointHead(input_channels=nb_filter[4], out_channels=nb_filter[3],stride=8)
        # self.head31_22 = PointHead(input_channels=nb_filter[3], out_channels=nb_filter[2],stride=4)
        # self.head22_13 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        # self.head13_04 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)

        # self.Up_conv5 = Conv_Block(nb_filters[4], nb_filters[3])
        # self.Up_conv4 = Conv_Block(nb_filters[3], nb_filters[2])
        # self.Up_conv3 = Conv_Block(nb_filters[2], nb_filters[1])
        # self.Up_conv2 = Conv_Block(nb_filters[1], nb_filters[0])

        self.final1 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final2 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final3 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        # self.final4 = nn.Sequential(
        #     nn.Conv2d(nb_filter[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     # nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )

    def forward(self, input):
        # encoding path
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        tmp0_1 = self.Up2(x1_0)
        att0_1=self.Att0_1(g=tmp0_1,x=x0_0)
        x0_1 = torch.cat([att0_1,tmp0_1], 1)
        x0_1 = self.conv0_1(x0_1)

        output1 = self.final1(x0_1)

        x2_0 = self.conv2_0(self.pool(x1_0))
        # print('x2_0 shape:',x2_0.shape) #[B,128,64,64]
        x2_0_1 = self.head20_11(x2_0, output1)

        tmp1_1 = self.Up3(x2_0)
        att1_1=self.Att1_1(g=x2_0_1,x=x1_0) ####
        x1_1 = torch.cat([att1_1,tmp1_1,x2_0_1], 1)
        x1_1 = self.conv1_1(x1_1)

        x1_1_1 = self.head11_02(x1_1, output1)
        tmp0_2 = self.Up2(x1_1)
        att0_2=self.Att0_2(g=x1_1_1,x=x0_1)
        x0_2 = torch.cat([x0_0,att0_2,tmp0_2,x1_1_1], 1)
        x0_2 = self.conv0_2(x0_2)

        output2 = self.final2(x0_2)

        x3_0 = self.conv3_0(self.pool(x2_0))
        x3_0_1 = self.head30_21(x3_0,output2)
        tmp2_1 = self.Up4(x3_0)
        att2_1=self.Att2_1(g=x3_0_1,x=x2_0)
        x2_1 = torch.cat([att2_1,tmp2_1,x3_0_1], 1)
        x2_1 = self.conv2_1(x2_1)
        x2_1_1 = self.head21_12(x2_1,output2)

        tmp1_2 = self.Up3(x2_1)
        att1_2=self.Att1_2(g=x2_1_1,x=x1_1)
        x1_2 = torch.cat([x1_0,att1_2,tmp1_2,x2_1_1], 1)
        x1_2 = self.conv1_2(x1_2)
        x1_2_1 = self.head12_03(x1_2,output2)
        tmp0_3 = self.Up2(x1_2)
        att0_3=self.Att0_3(g=x1_2_1,x=x0_2)
        x0_3 = torch.cat([x0_0,x0_1,att0_3,tmp0_3,x1_2_1], 1)
        x0_3 = self.conv0_3(x0_3)

        output3 = self.final3(x0_3)

        # x4_0 = self.conv4_0(self.pool(x3_0))
        # x4_0_1 = self.head40_31(x4_0,output3)
        # tmp3_1=self.Up5(x4_0)
        # att3_1=self.Att3_1(g=x4_0_1,x=x3_0)
        # x3_1 = torch.cat([att3_1,tmp3_1,x4_0_1], 1)
        # x3_1 = self.conv3_1(x3_1)
        # x3_1_1 = self.head31_22(x3_1,output3)
        # tmp2_2=self.Up4(x3_1)
        # att2_2=self.Att2_2(g=x3_1_1,x=x2_1)
        # x2_2 = torch.cat([x2_0,att2_2, tmp2_2, x3_1_1], 1)
        # x2_2 = self.conv2_2(x2_2)
        # x2_2_1 = self.head22_13(x2_2,output3)
        # tmp1_3=self.Up3(x2_2)
        # att1_3=self.Att1_3(g=x2_2_1,x=x1_2)
        # x1_3 = torch.cat([x1_0 ,x1_1,att1_3, tmp1_3 ,x2_2_1], 1)
        # x1_3 = self.conv1_3(x1_3)
        # x1_3_1 = self.head13_04(x1_3,output3)
        # tmp0_4=self.Up2(x1_3)
        # att0_4 = self.Att0_4(g=x1_3_1,x=x0_3)
        # x0_4 = torch.cat([x0_1, x0_2,att0_4, tmp0_4,x0_0, x1_3_1], 1)
        # x0_4 = self.conv0_4(x0_4)
        #
        # output4 = self.final4(x0_4)
        return [output1, output2, output3]#, output4]

        # else:
        #     output = self.final(x0_4)
        #     return output

class AttentionPointUNetPlusPlusL2(nn.Module):
    # without CGM,with ds
    def __init__(self, args):
        super().__init__()

        self.args = args

        nb_filter = [32,64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        # self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.Up5 = up_conv(nb_filter[4], nb_filter[3])
        # self.Up4 = up_conv(nb_filter[3], nb_filter[2])
        self.Up3 = up_conv(nb_filter[2], nb_filter[1])
        self.Up2 = up_conv(nb_filter[1], nb_filter[0])

        self.conv0_0 = Conv_Block(args.input_channels, nb_filter[0])
        self.conv1_0 = Conv_Block(nb_filter[0], nb_filter[1])
        self.conv2_0 = Conv_Block(nb_filter[1], nb_filter[2])
        # self.conv3_0 = Conv_Block(nb_filter[2], nb_filter[3])
        # self.conv4_0 = Conv_Block(nb_filter[3], nb_filter[4])

        self.conv0_1 = Conv_Block(nb_filter[0]+nb_filter[0], nb_filter[0])
        self.conv1_1 = Conv_Block(nb_filter[1]*3, nb_filter[1])
        # self.conv2_1 = Conv_Block(nb_filter[2]*3, nb_filter[2])
        # self.conv3_1 = Conv_Block(nb_filter[3]*3, nb_filter[3])

        self.conv0_2 = Conv_Block(nb_filter[0]*4, nb_filter[0])
        # self.conv1_2 = Conv_Block(nb_filter[1]*4, nb_filter[1])
        # self.conv2_2 = Conv_Block(nb_filter[2]*4, nb_filter[2])

        # self.conv0_3 = Conv_Block(nb_filter[0]*5, nb_filter[0])
        # self.conv1_3 = Conv_Block(nb_filter[1]*5, nb_filter[1])

        # self.conv0_4 = Conv_Block(nb_filter[0]*6, nb_filter[0])

        # self.Att4_2 = Attention_block(F_g=nb_filter[3], F_l=nb_filter[2]*2, F_int=nb_filter[1])
        # self.Att3_3 = Attention_block(F_g=nb_filter[2], F_l=nb_filter[1]*3, F_int=nb_filter[0])
        # self.Att2_4 = Attention_block(F_g=nb_filter[1], F_l=nb_filter[0]*4, F_int=32)

        self.Att0_1 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_1 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        # self.Att2_1 = Attention_block4(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])
        # self.Att3_1 = Attention_block4(F_g=nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[2])

        self.Att0_2 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        # self.Att1_2 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        # self.Att2_2 = Attention_block4(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])

        # self.Att0_3 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        # self.Att1_3 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])

        # self.Att0_4 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)

        self.head20_11 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        self.head11_02 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)

        # self.head30_21 = PointHead(input_channels=nb_filter[3], out_channels=nb_filter[2],stride=4)
        # self.head21_12 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        # self.head12_03 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)

        # self.head40_31 = PointHead(input_channels=nb_filter[4], out_channels=nb_filter[3],stride=8)
        # self.head31_22 = PointHead(input_channels=nb_filter[3], out_channels=nb_filter[2],stride=4)
        # self.head22_13 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        # self.head13_04 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)

        # self.Up_conv5 = Conv_Block(nb_filters[4], nb_filters[3])
        # self.Up_conv4 = Conv_Block(nb_filters[3], nb_filters[2])
        # self.Up_conv3 = Conv_Block(nb_filters[2], nb_filters[1])
        # self.Up_conv2 = Conv_Block(nb_filters[1], nb_filters[0])

        self.final1 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final2 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        # self.final3 = nn.Sequential(
        #     nn.Conv2d(nb_filter[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     # nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final4 = nn.Sequential(
        #     nn.Conv2d(nb_filter[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     # nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )

    def forward(self, input):
        # encoding path
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        tmp0_1 = self.Up2(x1_0)
        att0_1=self.Att0_1(g=tmp0_1,x=x0_0)
        x0_1 = torch.cat([att0_1,tmp0_1], 1)
        x0_1 = self.conv0_1(x0_1)

        output1 = self.final1(x0_1)

        x2_0 = self.conv2_0(self.pool(x1_0))
        # print('x2_0 shape:',x2_0.shape) #[B,128,64,64]
        x2_0_1 = self.head20_11(x2_0, output1)

        tmp1_1 = self.Up3(x2_0)
        att1_1=self.Att1_1(g=x2_0_1,x=x1_0) ####
        x1_1 = torch.cat([att1_1,tmp1_1,x2_0_1], 1)
        x1_1 = self.conv1_1(x1_1)

        x1_1_1 = self.head11_02(x1_1, output1)
        tmp0_2 = self.Up2(x1_1)
        att0_2=self.Att0_2(g=x1_1_1,x=x0_1)
        x0_2 = torch.cat([x0_0,att0_2,tmp0_2,x1_1_1], 1)
        x0_2 = self.conv0_2(x0_2)

        output2 = self.final2(x0_2)

        # x3_0 = self.conv3_0(self.pool(x2_0))
        # x3_0_1 = self.head30_21(x3_0,output2)
        # tmp2_1 = self.Up4(x3_0)
        # att2_1=self.Att2_1(g=x3_0_1,x=x2_0)
        # x2_1 = torch.cat([att2_1,tmp2_1,x3_0_1], 1)
        # x2_1 = self.conv2_1(x2_1)
        # x2_1_1 = self.head21_12(x2_1,output2)
        #
        # tmp1_2 = self.Up3(x2_1)
        # att1_2=self.Att1_2(g=x2_1_1,x=x1_1)
        # x1_2 = torch.cat([x1_0,att1_2,tmp1_2,x2_1_1], 1)
        # x1_2 = self.conv1_2(x1_2)
        # x1_2_1 = self.head12_03(x1_2,output2)
        # tmp0_3 = self.Up2(x1_2)
        # att0_3=self.Att0_3(g=x1_2_1,x=x0_2)
        # x0_3 = torch.cat([x0_0,x0_1,att0_3,tmp0_3,x1_2_1], 1)
        # x0_3 = self.conv0_3(x0_3)
        #
        # output3 = self.final3(x0_3)

        # x4_0 = self.conv4_0(self.pool(x3_0))
        # x4_0_1 = self.head40_31(x4_0,output3)
        # tmp3_1=self.Up5(x4_0)
        # att3_1=self.Att3_1(g=x4_0_1,x=x3_0)
        # x3_1 = torch.cat([att3_1,tmp3_1,x4_0_1], 1)
        # x3_1 = self.conv3_1(x3_1)
        # x3_1_1 = self.head31_22(x3_1,output3)
        # tmp2_2=self.Up4(x3_1)
        # att2_2=self.Att2_2(g=x3_1_1,x=x2_1)
        # x2_2 = torch.cat([x2_0,att2_2, tmp2_2, x3_1_1], 1)
        # x2_2 = self.conv2_2(x2_2)
        # x2_2_1 = self.head22_13(x2_2,output3)
        # tmp1_3=self.Up3(x2_2)
        # att1_3=self.Att1_3(g=x2_2_1,x=x1_2)
        # x1_3 = torch.cat([x1_0 ,x1_1,att1_3, tmp1_3 ,x2_2_1], 1)
        # x1_3 = self.conv1_3(x1_3)
        # x1_3_1 = self.head13_04(x1_3,output3)
        # tmp0_4=self.Up2(x1_3)
        # att0_4 = self.Att0_4(g=x1_3_1,x=x0_3)
        # x0_4 = torch.cat([x0_1, x0_2,att0_4, tmp0_4,x0_0, x1_3_1], 1)
        # x0_4 = self.conv0_4(x0_4)
        #
        # output4 = self.final4(x0_4)
        return [output1, output2]#, output3]#, output4]

        # else:
        #     output = self.final(x0_4)
        #     return output

class AttentionPointUNetPlusPlusL1(nn.Module):
    # without CGM,with ds
    def __init__(self, args):
        super().__init__()

        self.args = args

        nb_filter = [32,64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        # self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.Up5 = up_conv(nb_filter[4], nb_filter[3])
        # self.Up4 = up_conv(nb_filter[3], nb_filter[2])
        # self.Up3 = up_conv(nb_filter[2], nb_filter[1])
        self.Up2 = up_conv(nb_filter[1], nb_filter[0])

        self.conv0_0 = Conv_Block(args.input_channels, nb_filter[0])
        self.conv1_0 = Conv_Block(nb_filter[0], nb_filter[1])
        # self.conv2_0 = Conv_Block(nb_filter[1], nb_filter[2])
        # self.conv3_0 = Conv_Block(nb_filter[2], nb_filter[3])
        # self.conv4_0 = Conv_Block(nb_filter[3], nb_filter[4])

        self.conv0_1 = Conv_Block(nb_filter[0]+nb_filter[0], nb_filter[0])
        # self.conv1_1 = Conv_Block(nb_filter[1]*3, nb_filter[1])
        # self.conv2_1 = Conv_Block(nb_filter[2]*3, nb_filter[2])
        # self.conv3_1 = Conv_Block(nb_filter[3]*3, nb_filter[3])

        # self.conv0_2 = Conv_Block(nb_filter[0]*4, nb_filter[0])
        # self.conv1_2 = Conv_Block(nb_filter[1]*4, nb_filter[1])
        # self.conv2_2 = Conv_Block(nb_filter[2]*4, nb_filter[2])

        # self.conv0_3 = Conv_Block(nb_filter[0]*5, nb_filter[0])
        # self.conv1_3 = Conv_Block(nb_filter[1]*5, nb_filter[1])

        # self.conv0_4 = Conv_Block(nb_filter[0]*6, nb_filter[0])

        # self.Att4_2 = Attention_block(F_g=nb_filter[3], F_l=nb_filter[2]*2, F_int=nb_filter[1])
        # self.Att3_3 = Attention_block(F_g=nb_filter[2], F_l=nb_filter[1]*3, F_int=nb_filter[0])
        # self.Att2_4 = Attention_block(F_g=nb_filter[1], F_l=nb_filter[0]*4, F_int=32)

        self.Att0_1 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        # self.Att1_1 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        # self.Att2_1 = Attention_block4(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])
        # self.Att3_1 = Attention_block4(F_g=nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[2])

        # self.Att0_2 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        # self.Att1_2 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        # self.Att2_2 = Attention_block4(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])

        # self.Att0_3 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        # self.Att1_3 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])

        # self.Att0_4 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)

        # self.head20_11 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        # self.head11_02 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)

        # self.head30_21 = PointHead(input_channels=nb_filter[3], out_channels=nb_filter[2],stride=4)
        # self.head21_12 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        # self.head12_03 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)

        # self.head40_31 = PointHead(input_channels=nb_filter[4], out_channels=nb_filter[3],stride=8)
        # self.head31_22 = PointHead(input_channels=nb_filter[3], out_channels=nb_filter[2],stride=4)
        # self.head22_13 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        # self.head13_04 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)

        # self.Up_conv5 = Conv_Block(nb_filters[4], nb_filters[3])
        # self.Up_conv4 = Conv_Block(nb_filters[3], nb_filters[2])
        # self.Up_conv3 = Conv_Block(nb_filters[2], nb_filters[1])
        # self.Up_conv2 = Conv_Block(nb_filters[1], nb_filters[0])

        self.final1 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        # self.final2 = nn.Sequential(
        #     nn.Conv2d(nb_filter[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     # nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final3 = nn.Sequential(
        #     nn.Conv2d(nb_filter[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     # nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final4 = nn.Sequential(
        #     nn.Conv2d(nb_filter[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     # nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )

    def forward(self, input):
        # encoding path
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        tmp0_1 = self.Up2(x1_0)
        att0_1=self.Att0_1(g=tmp0_1,x=x0_0)
        x0_1 = torch.cat([att0_1,tmp0_1], 1)
        x0_1 = self.conv0_1(x0_1)

        output1 = self.final1(x0_1)

        # x2_0 = self.conv2_0(self.pool(x1_0))
        # # print('x2_0 shape:',x2_0.shape) #[B,128,64,64]
        # x2_0_1 = self.head20_11(x2_0, output1)
        #
        # tmp1_1 = self.Up3(x2_0)
        # att1_1=self.Att1_1(g=x2_0_1,x=x1_0) ####
        # x1_1 = torch.cat([att1_1,tmp1_1,x2_0_1], 1)
        # x1_1 = self.conv1_1(x1_1)
        #
        # x1_1_1 = self.head11_02(x1_1, output1)
        # tmp0_2 = self.Up2(x1_1)
        # att0_2=self.Att0_2(g=x1_1_1,x=x0_1)
        # x0_2 = torch.cat([x0_0,att0_2,tmp0_2,x1_1_1], 1)
        # x0_2 = self.conv0_2(x0_2)
        #
        # output2 = self.final2(x0_2)

        # x3_0 = self.conv3_0(self.pool(x2_0))
        # x3_0_1 = self.head30_21(x3_0,output2)
        # tmp2_1 = self.Up4(x3_0)
        # att2_1=self.Att2_1(g=x3_0_1,x=x2_0)
        # x2_1 = torch.cat([att2_1,tmp2_1,x3_0_1], 1)
        # x2_1 = self.conv2_1(x2_1)
        # x2_1_1 = self.head21_12(x2_1,output2)
        #
        # tmp1_2 = self.Up3(x2_1)
        # att1_2=self.Att1_2(g=x2_1_1,x=x1_1)
        # x1_2 = torch.cat([x1_0,att1_2,tmp1_2,x2_1_1], 1)
        # x1_2 = self.conv1_2(x1_2)
        # x1_2_1 = self.head12_03(x1_2,output2)
        # tmp0_3 = self.Up2(x1_2)
        # att0_3=self.Att0_3(g=x1_2_1,x=x0_2)
        # x0_3 = torch.cat([x0_0,x0_1,att0_3,tmp0_3,x1_2_1], 1)
        # x0_3 = self.conv0_3(x0_3)
        #
        # output3 = self.final3(x0_3)

        # x4_0 = self.conv4_0(self.pool(x3_0))
        # x4_0_1 = self.head40_31(x4_0,output3)
        # tmp3_1=self.Up5(x4_0)
        # att3_1=self.Att3_1(g=x4_0_1,x=x3_0)
        # x3_1 = torch.cat([att3_1,tmp3_1,x4_0_1], 1)
        # x3_1 = self.conv3_1(x3_1)
        # x3_1_1 = self.head31_22(x3_1,output3)
        # tmp2_2=self.Up4(x3_1)
        # att2_2=self.Att2_2(g=x3_1_1,x=x2_1)
        # x2_2 = torch.cat([x2_0,att2_2, tmp2_2, x3_1_1], 1)
        # x2_2 = self.conv2_2(x2_2)
        # x2_2_1 = self.head22_13(x2_2,output3)
        # tmp1_3=self.Up3(x2_2)
        # att1_3=self.Att1_3(g=x2_2_1,x=x1_2)
        # x1_3 = torch.cat([x1_0 ,x1_1,att1_3, tmp1_3 ,x2_2_1], 1)
        # x1_3 = self.conv1_3(x1_3)
        # x1_3_1 = self.head13_04(x1_3,output3)
        # tmp0_4=self.Up2(x1_3)
        # att0_4 = self.Att0_4(g=x1_3_1,x=x0_3)
        # x0_4 = torch.cat([x0_1, x0_2,att0_4, tmp0_4,x0_0, x1_3_1], 1)
        # x0_4 = self.conv0_4(x0_4)
        #
        # output4 = self.final4(x0_4)
        return [output1]#, output2]#, output3]#, output4]

        # else:
        #     output = self.final(x0_4)
        #     return output


def point_sample(input, point_coords, **kwargs):
    """
    主要思路：通过不确定像素点的位置信息，得到不确定像素点在input特征层上的对应特征
    :param input: 图片提取的特征（res2、out） eg.[2, 19, 48, 48]
    :param point_coords: 不确定像素点的位置信息 eg.[2, 48, 2], 2:batch_size, 48:不确定点的数量，2:空间相对坐标
    :return: 不确定像素点在input特征层上的对应特征 eg.[2, 19, 48]
    """
    """
    From Detectron2, point_features.py#19
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    #print('point_coords.dim() = ',point_coords.dim())   #3
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)    #在第2维（也就是第3维）增加一个维度
        #print('after unsqueeze',point_coords.shape)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0)#, **kwargs)
    #print(output.shape)

    if add_dim:
        output = output.squeeze(3)
        #print(output.shape)#12,1,48
    return output


@torch.no_grad()
def sampling_points(self, mask, N, k=3, beta=0.75, training=True):
    """
    Follows 3.1. Point Selection for Inference and Training
    In Train:, `The sampling strategy selects N points on a feature map to train on.`
    In Inference, `then selects the N most uncertain points`
    Args:
        mask(Tensor): [B, C, H, W] 粗糙的预测结果
        N(int): `During training we sample as many points as there are on a stride 16 feature map of the input`不确定点的个数，train中：N=图像size/16，infer中N=8096
        k(int): Over generation multiplier
        beta(float): ratio of importance points
        training(bool): flag
    Return:
        selected_point(Tensor) : flattened indexing points [B, num_points, 2] 不确定点的索引及坐标
    """
    # print(mask.shape)
    assert mask.dim() == 4, "Dim must be B(Batch)CHW"
    device = mask.device
    B, C, H, W = mask.shape #[B,1,H,W]
    # print(B,C,H,W)
    # mask = mask.cpu()

    # mask, _ = mask.cpu().sort(1, descending=True) #按照num_class中，每一类的总体得分进行排序
    # mask = mask.to(device)
    # if not training:
    H_step, W_step = 1 / H, 1 / W
    N = min(H * W, N)
    #print('N is ',N)
    # print('mask shape is ',mask.shape)  #[B,1,32,32],,,[B,1,256,256]

    # print(mask[0,0,:,:])
    over_generation = torch.rand(B, k * N, 2, device=device)
    over_generation_map = point_sample(mask, over_generation, align_corners=False)  #input, point_coords
    uncertainty_map = -1 * abs(mask-0.5)    #mask[:,0]是每个像素最大可能的分类，mask[:,1]是第二可能分类，如果一个像素预测处于既是最有可能的类别，又是第二可能的类别，那它是难分类的点，也就是mask0-mask1的值小，说明它难分，不确定度大
    # print(uncertainty_map[:,0,20,20])
    num_uncertain_points = int(beta * N)
    pp, idx = uncertainty_map.view(B, -1).topk(num_uncertain_points, dim=1) #选出最不好分的前N个点
    # print(pp[:,0],'idx is',idx)
    # print('idx shape is ',idx.shape)

    points = torch.zeros(B, num_uncertain_points, 2, dtype=torch.float, device=device)
    points[:, :, 0] = W_step / 2.0 + (idx  % W).to(torch.float) * W_step    #点的横坐标
    # print('point_x is',points[:, :, 0]* W)
    points[:, :, 1] = H_step / 2.0 + (idx // W).to(torch.float) * H_step    #点的纵坐标
    # print('point_y is',points[:, :, 1]* H)
    points_map = np.zeros([H,W])
    for i in range(H):
        index_x = int(points[0,i,0]*W-0.5)
        index_y = int(points[0,i,1]*H-0.5)
        points_map[index_y,index_x]=255
    # ct_dir = glob('/home/lc/学习/code-0524/output/point-maps/*')
    # # print(ct_dir)
    # if not self.training:
    #     length = len(ct_dir)
    #     # print(length)
    #     tmp = mask[0,0,:,:].cpu()
    #     imsave('output/point-maps/mask-%s-%s.png'%(length,H),tmp)
    #     imsave('output/point-maps/%s-%s.png'%(length,H),points_map)
    return idx, points  #

    # Official Comment : point_features.py#92
    # It is crucial to calculate uncertanty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to worse results. To illustrate the difference: a sampled point between two coarse predictions
    # with -1 and 1 logits has 0 logit prediction and therefore 0 uncertainty value, however, if one
    # calculates uncertainties for the coarse predictions first (-1 and -1) and sampe it for the
    # center point, they will get -1 unceratinty.

    # mask = mask.to(device)
	# k means oversample_ratio , should bigger than 1
    # over_generation = torch.rand(B, k * N, 2, device=device)
    # #print('over_generation size is ',over_generation.shape) #12,48,2
    # over_generation_map = point_sample(mask, over_generation, align_corners=False)  #input, point_coords
    # print('over_generation_map size is ',over_generation_map.shape) #B,1,48
    # print(over_generation_map)
    # uncertainty_map = -1 * abs(over_generation_map[:,0]-0.5)
    # #contains uncertainty scores with the most uncertain locations having the highest uncertainty score.
    # # print(uncertainty_map.shape)    #B,48
    # num_uncertain_points = int(beta * N)
    # num_random_points = N - num_uncertain_points
    # # values, idx = uncertainty_map.topk(num_uncertain_points, -1)   #每行中前N个最大的数
    # values, idx = torch.topk(uncertainty_map, k=num_uncertain_points,dim=1)   #每行中前N个最大的数

    # #print(values,idx)
    # shift = (k * N) * torch.arange(B, dtype=torch.long, device=device)

    # idx += shift[:, None]

    # selected_point = over_generation.view(-1, 2)[idx.view(-1), :].view(B, num_uncertain_points, 2)
    # if num_random_points>0:
    #     selected_point = torch.cat([
    #         selected_point,
    #         torch.rand(B, num_random_points, 2, device=device)
    #     ],
    #     dim=1
    #     )
    # # print(selected_point.shape,' selected_points :',selected_point)# [B,16,2]
    # return selected_point

def sampling_points_3D(self, mask, N, k=3, beta=0.75, training=True):
    """
    Follows 3.1. Point Selection for Inference and Training
    In Train:, `The sampling strategy selects N points on a feature map to train on.`
    In Inference, `then selects the N most uncertain points`
    Args:
        mask(Tensor): [B, C, H, W] 粗糙的预测结果
        N(int): `During training we sample as many points as there are on a stride 16 feature map of the input`不确定点的个数，train中：N=图像size/16，infer中N=8096
        k(int): Over generation multiplier
        beta(float): ratio of importance points
        training(bool): flag
    Return:
        selected_point(Tensor) : flattened indexing points [B, num_points, 2] 不确定点的索引及坐标
    """
    # print(mask.shape)
    assert mask.dim() == 5, "Dim must be B(Batch)CHW"
    device = mask.device
    B, C, L, H, W = mask.shape #[B,1,H,W]
    # print(B,C,H,W)
    # mask = mask.cpu()

    # mask, _ = mask.cpu().sort(1, descending=True) #按照num_class中，每一类的总体得分进行排序
    # mask = mask.to(device)
    # if not training:
    H_step, W_step ,L_step= 1 / H, 1 / W, 1/L
    N = min(H * W * L, N)
    #print('N is ',N)
    # print('mask shape is ',mask.shape)  #[B,1,32,32],,,[B,1,256,256]

    # print(mask[0,0,:,:])
    over_generation = torch.rand(B, k * N, 2, device=device)
    over_generation_map = point_sample(mask, over_generation, align_corners=False)  #input, point_coords
    uncertainty_map = -1 * abs(mask-0.5)    #mask[:,0]是每个像素最大可能的分类，mask[:,1]是第二可能分类，如果一个像素预测处于既是最有可能的类别，又是第二可能的类别，那它是难分类的点，也就是mask0-mask1的值小，说明它难分，不确定度大
    # print(uncertainty_map[:,0,20,20])
    num_uncertain_points = int(beta * N)
    pp, idx = uncertainty_map.view(B, -1).topk(num_uncertain_points, dim=1) #选出最不好分的前N个点
    # print(pp[:,0],'idx is',idx)
    # print('idx shape is ',idx.shape)

    points = torch.zeros(B, num_uncertain_points, 3, dtype=torch.float, device=device)
    points[:, :, 0] = W_step / 2.0 + (idx  % W).to(torch.float) * W_step    #点的横坐标
    # print('point_x is',points[:, :, 0]* W)
    points[:, :, 1] = H_step / 2.0 + (idx // W).to(torch.float) * H_step    #点的纵坐标
    points[:, :, 2] = L_step / 2.0 + (idx // L).to(torch.float) * L_step    #点的纵坐标
    # print('point_y is',points[:, :, 1]* H)
    points_map = np.zeros([L,H,W])
    for i in range(H):
        index_x = int(points[0,i,0]*W-0.5)
        index_y = int(points[0,i,1]*H-0.5)
        index_z = int(points[0,i,2]*L-0.5)
        points_map[index_z,index_y,index_x]=255
    # ct_dir = glob('/home/lc/学习/code-0524/output/point-maps/*')
    # # print(ct_dir)
    # if not self.training:
    #     length = len(ct_dir)
    #     # print(length)
    #     tmp = mask[0,0,:,:].cpu()
    #     imsave('output/point-maps/mask-%s-%s.png'%(length,H),tmp)
    #     imsave('output/point-maps/%s-%s.png'%(length,H),points_map)
    return idx, points  #

    # Official Comment : point_features.py#92
    # It is crucial to calculate uncertanty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to worse results. To illustrate the difference: a sampled point between two coarse predictions
    # with -1 and 1 logits has 0 logit prediction and therefore 0 uncertainty value, however, if one
    # calculates uncertainties for the coarse predictions first (-1 and -1) and sampe it for the
    # center point, they will get -1 unceratinty.

    # mask = mask.to(device)
	# k means oversample_ratio , should bigger than 1
    # over_generation = torch.rand(B, k * N, 2, device=device)
    # #print('over_generation size is ',over_generation.shape) #12,48,2
    # over_generation_map = point_sample(mask, over_generation, align_corners=False)  #input, point_coords
    # print('over_generation_map size is ',over_generation_map.shape) #B,1,48
    # print(over_generation_map)
    # uncertainty_map = -1 * abs(over_generation_map[:,0]-0.5)
    # #contains uncertainty scores with the most uncertain locations having the highest uncertainty score.
    # # print(uncertainty_map.shape)    #B,48
    # num_uncertain_points = int(beta * N)
    # num_random_points = N - num_uncertain_points
    # # values, idx = uncertainty_map.topk(num_uncertain_points, -1)   #每行中前N个最大的数
    # values, idx = torch.topk(uncertainty_map, k=num_uncertain_points,dim=1)   #每行中前N个最大的数

    # #print(values,idx)
    # shift = (k * N) * torch.arange(B, dtype=torch.long, device=device)

    # idx += shift[:, None]

    # selected_point = over_generation.view(-1, 2)[idx.view(-1), :].view(B, num_uncertain_points, 2)
    # if num_random_points>0:
    #     selected_point = torch.cat([
    #         selected_point,
    #         torch.rand(B, num_random_points, 2, device=device)
    #     ],
    #     dim=1
    #     )
    # # print(selected_point.shape,' selected_points :',selected_point)# [B,16,2]
    # return selected_point

class NestedUNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        nb_filter = [64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = Conv_Block(args.input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = Conv_Block(nb_filter[0],  nb_filter[1])
        self.conv2_0 = Conv_Block(nb_filter[1],  nb_filter[2])
        self.conv3_0 = Conv_Block(nb_filter[2],  nb_filter[3])
        self.conv4_0 = Conv_Block(nb_filter[3],  nb_filter[4])

        self.conv0_1 = Conv_Block(nb_filter[0]+nb_filter[1],  nb_filter[0])
        self.conv1_1 = Conv_Block(nb_filter[1]+nb_filter[2],  nb_filter[1])
        self.conv2_1 = Conv_Block(nb_filter[2]+nb_filter[3],  nb_filter[2])
        self.conv3_1 = Conv_Block(nb_filter[3]+nb_filter[4],  nb_filter[3])

        self.conv0_2 = Conv_Block(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_2 = Conv_Block(nb_filter[1]*2+nb_filter[2], nb_filter[1])
        self.conv2_2 = Conv_Block(nb_filter[2]*2+nb_filter[3],  nb_filter[2])

        self.conv0_3 = Conv_Block(nb_filter[0]*3+nb_filter[1], nb_filter[0])
        self.conv1_3 = Conv_Block(nb_filter[1]*3+nb_filter[2],  nb_filter[1])

        self.conv0_4 = Conv_Block(nb_filter[0]*4+nb_filter[1],  nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.args.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

class AttentionUNetPlusPlus(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        nb_filter = [64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.Up5 = up_conv(nb_filter[4], nb_filter[3])
        self.Up4 = up_conv(nb_filter[3], nb_filter[2])
        self.Up3 = up_conv(nb_filter[2], nb_filter[1])
        self.Up2 = up_conv(nb_filter[1], nb_filter[0])

        self.conv0_0 = Conv_Block(args.input_channels, nb_filter[0])
        self.conv1_0 = Conv_Block(nb_filter[0], nb_filter[1])
        self.conv2_0 = Conv_Block(nb_filter[1], nb_filter[2])
        self.conv3_0 = Conv_Block(nb_filter[2], nb_filter[3])
        self.conv4_0 = Conv_Block(nb_filter[3], nb_filter[4])

        self.conv0_1 = Conv_Block(nb_filter[0]+nb_filter[1], nb_filter[0])
        self.conv1_1 = Conv_Block(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv2_1 = Conv_Block(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv3_1 = Conv_Block(nb_filter[3]+nb_filter[4], nb_filter[3])

        self.conv0_2 = Conv_Block(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_2 = Conv_Block(nb_filter[1]*2+nb_filter[2], nb_filter[1])
        self.conv2_2 = Conv_Block(nb_filter[2]*2+nb_filter[3], nb_filter[2])

        self.conv0_3 = Conv_Block(nb_filter[0]*3+nb_filter[1], nb_filter[0])
        self.conv1_3 = Conv_Block(nb_filter[1]*3+nb_filter[2], nb_filter[1])

        self.conv0_4 = Conv_Block(nb_filter[0]*4+nb_filter[1], nb_filter[0])

        self.Att5 = Attention_block(F_g=nb_filter[4], F_l=nb_filter[3], F_int=nb_filter[2])
        self.Att4 = Attention_block(F_g=nb_filter[3], F_l=nb_filter[2]*2, F_int=nb_filter[1])
        self.Att3 = Attention_block(F_g=nb_filter[2], F_l=nb_filter[1]*3, F_int=nb_filter[0])
        self.Att2 = Attention_block(F_g=nb_filter[1], F_l=nb_filter[0]*4, F_int=32)

        # self.Up_conv5 = Conv_Block(nb_filters[4], nb_filters[3])
        # self.Up_conv4 = Conv_Block(nb_filters[3], nb_filters[2])
        # self.Up_conv3 = Conv_Block(nb_filters[2], nb_filters[1])
        # self.Up_conv2 = Conv_Block(nb_filters[1], nb_filters[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1) # 卷积个数，输出通道数，kernel尺寸，stride，pad


    def forward(self, input):
        # encoding path
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # decoding + concat path
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        tmp4=self.Up(x4_0)
        # print('x4_0:',x4_0.shape)

        att4=self.Att5(g=tmp4,x=x3_0)
        # print("att4 : ",att4.shape)
        x3_1 = torch.cat([att4,tmp4], 1)
        # print("x3_1 : ",x3_1.shape)
        x3_1 = self.conv3_1(x3_1)
        # print('x0_0:',x0_0.shape)
        # print('x0_1:',x0_1.shape)
        # print('x1_1:',x1_1.shape)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        tmp3=self.Up(x3_1)
        # print('tmp3:',tmp3.shape)
        att3=self.Att4(g=tmp3,x=torch.cat([x2_0, x2_1], 1))
        x2_2 = torch.cat([att3, tmp3], 1)
        # print('att3:',att3.shape)
        # print('tmp3:',tmp3.shape)
        x2_2 = self.conv2_2(x2_2)

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))
        tmp2=self.Up(x2_2)
        att2=self.Att3(g=tmp2,x=torch.cat([x1_0, x1_1, x1_2], 1))
        x1_3 = torch.cat([att2, tmp2], 1)
        x1_3 = self.conv1_3(x1_3)

        tmp1=self.Up(x1_3)
        att1 = self.Att2(g=tmp1,x=torch.cat([x0_0, x0_1, x0_2, x0_3], 1))
        x0_4 = torch.cat([att1, tmp1], 1)
        x0_4 = self.conv0_4(x0_4)

        if self.args.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

class AttentionUNetPlusPlus2(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        nb_filter = [32,64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Up5 = up_conv(nb_filter[4], nb_filter[3])
        self.Up4 = up_conv(nb_filter[3], nb_filter[2])
        self.Up3 = up_conv(nb_filter[2], nb_filter[1])
        self.Up2 = up_conv(nb_filter[1], nb_filter[0])

        self.conv0_0 = Conv_Block(args.input_channels, nb_filter[0])
        self.conv1_0 = Conv_Block(nb_filter[0], nb_filter[1])
        self.conv2_0 = Conv_Block(nb_filter[1], nb_filter[2])
        self.conv3_0 = Conv_Block(nb_filter[2], nb_filter[3])
        self.conv4_0 = Conv_Block(nb_filter[3], nb_filter[4])

        self.conv0_1 = Conv_Block(nb_filter[1], nb_filter[0])
        self.conv1_1 = Conv_Block(nb_filter[2], nb_filter[1])
        self.conv2_1 = Conv_Block(nb_filter[3], nb_filter[2])
        self.conv3_1 = Conv_Block(nb_filter[4], nb_filter[3])

        self.conv0_2 = Conv_Block(nb_filter[0]*1+nb_filter[1], nb_filter[0])
        self.conv1_2 = Conv_Block(nb_filter[1]*1+nb_filter[2], nb_filter[1])
        self.conv2_2 = Conv_Block(nb_filter[2]*1+nb_filter[3], nb_filter[2])

        self.conv0_3 = Conv_Block(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_3 = Conv_Block(nb_filter[1]*2+nb_filter[2], nb_filter[1])

        self.conv0_4 = Conv_Block(nb_filter[0]*3+nb_filter[1], nb_filter[0])

        # self.Att4_2 = Attention_block(F_g=nb_filter[3], F_l=nb_filter[2]*2, F_int=nb_filter[1])
        # self.Att3_3 = Attention_block(F_g=nb_filter[2], F_l=nb_filter[1]*3, F_int=nb_filter[0])
        # self.Att2_4 = Attention_block(F_g=nb_filter[1], F_l=nb_filter[0]*4, F_int=32)

        self.Att0_1 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_1 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Att2_1 = Attention_block4(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])
        self.Att3_1 = Attention_block4(F_g=nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[2])

        self.Att0_2 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_2 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Att2_2 = Attention_block4(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])

        self.Att0_3 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_3 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])

        self.Att0_4 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)

        # self.Up_conv5 = Conv_Block(nb_filters[4], nb_filters[3])
        # self.Up_conv4 = Conv_Block(nb_filters[3], nb_filters[2])
        # self.Up_conv3 = Conv_Block(nb_filters[2], nb_filters[1])
        # self.Up_conv2 = Conv_Block(nb_filters[1], nb_filters[0])

        if self.args.deepsupervision:
            self.final1 = nn.Sequential(
                nn.Conv2d(nb_filter[0], 1, 1, 1),
                # nn.ConvTranspose2d(1, 1, 2, 2),
                # nn.Upsample(scale_factor=2,mode='nearest'),
                nn.Sigmoid()
            )
            self.final2 = nn.Sequential(
                nn.Conv2d(nb_filter[0], 1, 1, 1),
                # nn.ConvTranspose2d(1, 1, 2, 2),
                # nn.Upsample(scale_factor=2,mode='nearest'),
                nn.Sigmoid()
            )
            self.final3 = nn.Sequential(
                nn.Conv2d(nb_filter[0], 1, 1, 1),
                # nn.ConvTranspose2d(1, 1, 2, 2),
                # nn.Upsample(scale_factor=2,mode='nearest'),
                nn.Sigmoid()
            )
            self.final4 = nn.Sequential(
                nn.Conv2d(nb_filter[0], 1, 1, 1),
                # nn.ConvTranspose2d(1, 1, 2, 2),
                # nn.Upsample(scale_factor=2,mode='nearest'),
                nn.Sigmoid()
            )
        else:
            self.final = nn.Sequential(
                nn.Conv2d(nb_filter[0], 1, 1, 1),
                # nn.ConvTranspose2d(1, 1, 2, 2),
                # nn.Upsample(scale_factor=2,mode='nearest'),
                nn.Sigmoid()
            ) # 卷积个数，输出通道数，kernel尺寸，stride，pad


    def forward(self, input):
        # encoding path
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # decoding + concat path
        # print(x1_0.shape)
        tmp0_1 = self.Up2(x1_0)
        # print("tmp0_1",tmp0_1.shape)
        att0_1=self.Att0_1(g=tmp0_1,x=x0_0)
        # print("x0_0",x0_0.shape)
        # print("att0_1",att0_1.shape)
        x0_1 = torch.cat([att0_1,tmp0_1], 1)
        # print(x0_1.shape)
        x0_1 = self.conv0_1(x0_1)

        # x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        tmp1_1 = self.Up3(x2_0)
        att1_1=self.Att1_1(g=tmp1_1,x=x1_0)
        x1_1 = torch.cat([att1_1,tmp1_1], 1)
        x1_1 = self.conv1_1(x1_1)

        # x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        tmp2_1 = self.Up4(x3_0)
        att2_1=self.Att2_1(g=tmp2_1,x=x2_0)
        x2_1 = torch.cat([att2_1,tmp2_1], 1)
        x2_1 = self.conv2_1(x2_1)

        tmp3_1=self.Up5(x4_0)
        att3_1=self.Att3_1(g=tmp3_1,x=x3_0)
        x3_1 = torch.cat([att3_1,tmp3_1], 1)
        x3_1 = self.conv3_1(x3_1)

        # x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))
        tmp0_2 = self.Up2(x1_1)
        att0_2=self.Att0_2(g=tmp0_2,x=x0_1)
        x0_2 = torch.cat([att0_2,tmp0_2,x0_0], 1)
        x0_2 = self.conv0_2(x0_2)

        # x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        tmp1_2 = self.Up3(x2_1)
        att1_2=self.Att1_2(g=tmp1_2,x=x1_1)
        x1_2 = torch.cat([att1_2,tmp1_2,x1_0], 1)
        x1_2 = self.conv1_2(x1_2)

        tmp2_2=self.Up4(x3_1)
        att2_2=self.Att2_2(g=tmp2_2,x=x2_1)
        x2_2 = torch.cat([att2_2, tmp2_2, x2_0], 1)
        x2_2 = self.conv2_2(x2_2)

        # x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))
        tmp0_3 = self.Up2(x1_2)
        att0_3=self.Att0_3(g=tmp0_3,x=x0_2)
        x0_3 = torch.cat([att0_3,tmp0_3,x0_0,x0_1], 1)
        x0_3 = self.conv0_3(x0_3)

        tmp1_3=self.Up3(x2_2)
        att1_3=self.Att1_3(g=tmp1_3,x=x1_2)
        x1_3 = torch.cat([att1_3, tmp1_3 ,x1_0 ,x1_1], 1)
        x1_3 = self.conv1_3(x1_3)

        tmp0_4=self.Up2(x1_3)
        att0_4 = self.Att0_4(g=tmp0_4,x=x0_3)
        x0_4 = torch.cat([att0_4, tmp0_4,x0_0, x0_1, x0_2], 1)
        x0_4 = self.conv0_4(x0_4)

        if self.args.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

class AttentionUNetPlusPlusL3(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        nb_filter = [32,64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.Up4 = up_conv(nb_filter[3], nb_filter[2])
        self.Up3 = up_conv(nb_filter[2], nb_filter[1])
        self.Up2 = up_conv(nb_filter[1], nb_filter[0])

        self.conv0_0 = Conv_Block(args.input_channels, nb_filter[0])
        self.conv1_0 = Conv_Block(nb_filter[0], nb_filter[1])
        self.conv2_0 = Conv_Block(nb_filter[1], nb_filter[2])
        self.conv3_0 = Conv_Block(nb_filter[2], nb_filter[3])

        self.conv0_1 = Conv_Block(nb_filter[1], nb_filter[0])
        self.conv1_1 = Conv_Block(nb_filter[2], nb_filter[1])
        self.conv2_1 = Conv_Block(nb_filter[3], nb_filter[2])

        self.conv0_2 = Conv_Block(nb_filter[0]*1+nb_filter[1], nb_filter[0])
        self.conv1_2 = Conv_Block(nb_filter[1]*1+nb_filter[2], nb_filter[1])

        self.conv0_3 = Conv_Block(nb_filter[0]*2+nb_filter[1], nb_filter[0])


        # self.Att4_2 = Attention_block(F_g=nb_filter[3], F_l=nb_filter[2]*2, F_int=nb_filter[1])
        # self.Att3_3 = Attention_block(F_g=nb_filter[2], F_l=nb_filter[1]*3, F_int=nb_filter[0])
        # self.Att2_4 = Attention_block(F_g=nb_filter[1], F_l=nb_filter[0]*4, F_int=32)

        self.Att0_1 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_1 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Att2_1 = Attention_block4(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])

        self.Att0_2 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_2 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])

        self.Att0_3 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)


        # self.Up_conv5 = Conv_Block(nb_filters[4], nb_filters[3])
        # self.Up_conv4 = Conv_Block(nb_filters[3], nb_filters[2])
        # self.Up_conv3 = Conv_Block(nb_filters[2], nb_filters[1])
        # self.Up_conv2 = Conv_Block(nb_filters[1], nb_filters[0])

        self.final1 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final2 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final3 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )


    def forward(self, input):
        # encoding path
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        # decoding + concat path
        # print(x1_0.shape)
        tmp0_1 = self.Up2(x1_0)
        # print("tmp0_1",tmp0_1.shape)
        att0_1=self.Att0_1(g=tmp0_1,x=x0_0)
        # print("x0_0",x0_0.shape)
        # print("att0_1",att0_1.shape)
        x0_1 = torch.cat([att0_1,tmp0_1], 1)
        # print(x0_1.shape)
        x0_1 = self.conv0_1(x0_1)

        # x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        tmp1_1 = self.Up3(x2_0)
        att1_1=self.Att1_1(g=tmp1_1,x=x1_0)
        x1_1 = torch.cat([att1_1,tmp1_1], 1)
        x1_1 = self.conv1_1(x1_1)

        # x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        tmp2_1 = self.Up4(x3_0)
        att2_1=self.Att2_1(g=tmp2_1,x=x2_0)
        x2_1 = torch.cat([att2_1,tmp2_1], 1)
        x2_1 = self.conv2_1(x2_1)



        # x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))
        tmp0_2 = self.Up2(x1_1)
        att0_2=self.Att0_2(g=tmp0_2,x=x0_1)
        x0_2 = torch.cat([att0_2,tmp0_2,x0_0], 1)
        x0_2 = self.conv0_2(x0_2)

        # x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        tmp1_2 = self.Up3(x2_1)
        att1_2=self.Att1_2(g=tmp1_2,x=x1_1)
        x1_2 = torch.cat([att1_2,tmp1_2,x1_0], 1)
        x1_2 = self.conv1_2(x1_2)



        # x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))
        tmp0_3 = self.Up2(x1_2)
        att0_3=self.Att0_3(g=tmp0_3,x=x0_2)
        x0_3 = torch.cat([att0_3,tmp0_3,x0_0,x0_1], 1)
        x0_3 = self.conv0_3(x0_3)





        output1 = self.final1(x0_1)
        output2 = self.final2(x0_2)
        output3 = self.final3(x0_3)
        return output3#[output1, output2, output3]

class AttentionUNetPlusPlusL2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        nb_filter = [32,64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Up5 = up_conv(nb_filter[4], nb_filter[3])
        self.Up4 = up_conv(nb_filter[3], nb_filter[2])
        self.Up3 = up_conv(nb_filter[2], nb_filter[1])
        self.Up2 = up_conv(nb_filter[1], nb_filter[0])

        self.conv0_0 = Conv_Block(args.input_channels, nb_filter[0])
        self.conv1_0 = Conv_Block(nb_filter[0], nb_filter[1])
        self.conv2_0 = Conv_Block(nb_filter[1], nb_filter[2])
        self.conv3_0 = Conv_Block(nb_filter[2], nb_filter[3])
        self.conv4_0 = Conv_Block(nb_filter[3], nb_filter[4])

        self.conv0_1 = Conv_Block(nb_filter[1], nb_filter[0])
        self.conv1_1 = Conv_Block(nb_filter[2], nb_filter[1])
        self.conv2_1 = Conv_Block(nb_filter[3], nb_filter[2])
        self.conv3_1 = Conv_Block(nb_filter[4], nb_filter[3])

        self.conv0_2 = Conv_Block(nb_filter[0]*1+nb_filter[1], nb_filter[0])
        self.conv1_2 = Conv_Block(nb_filter[1]*1+nb_filter[2], nb_filter[1])
        self.conv2_2 = Conv_Block(nb_filter[2]*1+nb_filter[3], nb_filter[2])

        self.conv0_3 = Conv_Block(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_3 = Conv_Block(nb_filter[1]*2+nb_filter[2], nb_filter[1])

        self.conv0_4 = Conv_Block(nb_filter[0]*3+nb_filter[1], nb_filter[0])

        # self.Att4_2 = Attention_block(F_g=nb_filter[3], F_l=nb_filter[2]*2, F_int=nb_filter[1])
        # self.Att3_3 = Attention_block(F_g=nb_filter[2], F_l=nb_filter[1]*3, F_int=nb_filter[0])
        # self.Att2_4 = Attention_block(F_g=nb_filter[1], F_l=nb_filter[0]*4, F_int=32)

        self.Att0_1 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_1 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Att2_1 = Attention_block4(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])
        self.Att3_1 = Attention_block4(F_g=nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[2])

        self.Att0_2 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_2 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Att2_2 = Attention_block4(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])

        self.Att0_3 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_3 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])

        self.Att0_4 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)

        # self.Up_conv5 = Conv_Block(nb_filters[4], nb_filters[3])
        # self.Up_conv4 = Conv_Block(nb_filters[3], nb_filters[2])
        # self.Up_conv3 = Conv_Block(nb_filters[2], nb_filters[1])
        # self.Up_conv2 = Conv_Block(nb_filters[1], nb_filters[0])

        if self.args.deepsupervision:
            self.final1 = nn.Sequential(
                nn.Conv2d(nb_filter[0], 1, 1, 1),
                # nn.ConvTranspose2d(1, 1, 2, 2),
                # nn.Upsample(scale_factor=2,mode='nearest'),
                nn.Sigmoid()
            )
            self.final2 = nn.Sequential(
                nn.Conv2d(nb_filter[0], 1, 1, 1),
                # nn.ConvTranspose2d(1, 1, 2, 2),
                # nn.Upsample(scale_factor=2,mode='nearest'),
                nn.Sigmoid()
            )
            self.final3 = nn.Sequential(
                nn.Conv2d(nb_filter[0], 1, 1, 1),
                # nn.ConvTranspose2d(1, 1, 2, 2),
                # nn.Upsample(scale_factor=2,mode='nearest'),
                nn.Sigmoid()
            )
            self.final4 = nn.Sequential(
                nn.Conv2d(nb_filter[0], 1, 1, 1),
                # nn.ConvTranspose2d(1, 1, 2, 2),
                # nn.Upsample(scale_factor=2,mode='nearest'),
                nn.Sigmoid()
            )
        else:
            self.final1 = nn.Sequential(
                nn.Conv2d(nb_filter[0], 1, 1, 1),
                # nn.ConvTranspose2d(1, 1, 2, 2),
                # nn.Upsample(scale_factor=2,mode='nearest'),
                nn.Sigmoid()
            ) # 卷积个数，输出通道数，kernel尺寸，stride，pad



    def forward(self, input):
        # encoding path
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))

        # decoding + concat path
        # print(x1_0.shape)
        tmp0_1 = self.Up2(x1_0)
        # print("tmp0_1",tmp0_1.shape)
        att0_1=self.Att0_1(g=tmp0_1,x=x0_0)
        # print("x0_0",x0_0.shape)
        # print("att0_1",att0_1.shape)
        x0_1 = torch.cat([att0_1,tmp0_1], 1)
        # print(x0_1.shape)
        x0_1 = self.conv0_1(x0_1)

        # x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        tmp1_1 = self.Up3(x2_0)
        att1_1=self.Att1_1(g=tmp1_1,x=x1_0)
        x1_1 = torch.cat([att1_1,tmp1_1], 1)
        x1_1 = self.conv1_1(x1_1)

        # x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))
        tmp0_2 = self.Up2(x1_1)
        att0_2=self.Att0_2(g=tmp0_2,x=x0_1)
        x0_2 = torch.cat([att0_2,tmp0_2,x0_0], 1)
        x0_2 = self.conv0_2(x0_2)


        # output1 = self.final1(x0_1)
        output1 = self.final1(x0_2)
        return output1#[output1, output2]

class AttentionUNetPlusPlusL1(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        nb_filter = [32,64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Up5 = up_conv(nb_filter[4], nb_filter[3])
        self.Up4 = up_conv(nb_filter[3], nb_filter[2])
        self.Up3 = up_conv(nb_filter[2], nb_filter[1])
        self.Up2 = up_conv(nb_filter[1], nb_filter[0])

        self.conv0_0 = Conv_Block(args.input_channels, nb_filter[0])
        self.conv1_0 = Conv_Block(nb_filter[0], nb_filter[1])
        self.conv2_0 = Conv_Block(nb_filter[1], nb_filter[2])
        self.conv3_0 = Conv_Block(nb_filter[2], nb_filter[3])
        self.conv4_0 = Conv_Block(nb_filter[3], nb_filter[4])

        self.conv0_1 = Conv_Block(nb_filter[1], nb_filter[0])
        self.conv1_1 = Conv_Block(nb_filter[2], nb_filter[1])
        self.conv2_1 = Conv_Block(nb_filter[3], nb_filter[2])
        self.conv3_1 = Conv_Block(nb_filter[4], nb_filter[3])

        self.conv0_2 = Conv_Block(nb_filter[0]*1+nb_filter[1], nb_filter[0])
        self.conv1_2 = Conv_Block(nb_filter[1]*1+nb_filter[2], nb_filter[1])
        self.conv2_2 = Conv_Block(nb_filter[2]*1+nb_filter[3], nb_filter[2])

        self.conv0_3 = Conv_Block(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_3 = Conv_Block(nb_filter[1]*2+nb_filter[2], nb_filter[1])

        self.conv0_4 = Conv_Block(nb_filter[0]*3+nb_filter[1], nb_filter[0])

        # self.Att4_2 = Attention_block(F_g=nb_filter[3], F_l=nb_filter[2]*2, F_int=nb_filter[1])
        # self.Att3_3 = Attention_block(F_g=nb_filter[2], F_l=nb_filter[1]*3, F_int=nb_filter[0])
        # self.Att2_4 = Attention_block(F_g=nb_filter[1], F_l=nb_filter[0]*4, F_int=32)

        self.Att0_1 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_1 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Att2_1 = Attention_block4(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])
        self.Att3_1 = Attention_block4(F_g=nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[2])

        self.Att0_2 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_2 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Att2_2 = Attention_block4(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])

        self.Att0_3 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_3 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])

        self.Att0_4 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)

        # self.Up_conv5 = Conv_Block(nb_filters[4], nb_filters[3])
        # self.Up_conv4 = Conv_Block(nb_filters[3], nb_filters[2])
        # self.Up_conv3 = Conv_Block(nb_filters[2], nb_filters[1])
        # self.Up_conv2 = Conv_Block(nb_filters[1], nb_filters[0])

        if self.args.deepsupervision:
            self.final1 = nn.Sequential(
                nn.Conv2d(nb_filter[0], 1, 1, 1),
                # nn.ConvTranspose2d(1, 1, 2, 2),
                # nn.Upsample(scale_factor=2,mode='nearest'),
                nn.Sigmoid()
            )
            self.final2 = nn.Sequential(
                nn.Conv2d(nb_filter[0], 1, 1, 1),
                # nn.ConvTranspose2d(1, 1, 2, 2),
                # nn.Upsample(scale_factor=2,mode='nearest'),
                nn.Sigmoid()
            )
            self.final3 = nn.Sequential(
                nn.Conv2d(nb_filter[0], 1, 1, 1),
                # nn.ConvTranspose2d(1, 1, 2, 2),
                # nn.Upsample(scale_factor=2,mode='nearest'),
                nn.Sigmoid()
            )
            self.final4 = nn.Sequential(
                nn.Conv2d(nb_filter[0], 1, 1, 1),
                # nn.ConvTranspose2d(1, 1, 2, 2),
                # nn.Upsample(scale_factor=2,mode='nearest'),
                nn.Sigmoid()
            )
        else:
            self.final1 = nn.Sequential(
                nn.Conv2d(nb_filter[0], 1, 1, 1),
                # nn.ConvTranspose2d(1, 1, 2, 2),
                # nn.Upsample(scale_factor=2,mode='nearest'),
                nn.Sigmoid()
            ) # 卷积个数，输出通道数，kernel尺寸，stride，pad




    def forward(self, input):
        # encoding path
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))

        # decoding + concat path
        # print(x1_0.shape)
        tmp0_1 = self.Up2(x1_0)
        # print("tmp0_1",tmp0_1.shape)
        att0_1=self.Att0_1(g=tmp0_1,x=x0_0)
        # print("x0_0",x0_0.shape)
        # print("att0_1",att0_1.shape)
        x0_1 = torch.cat([att0_1,tmp0_1], 1)
        # print(x0_1.shape)
        x0_1 = self.conv0_1(x0_1)


        output1 = self.final1(x0_1)
        return output1

class VNet(nn.Module):
    """

    共9498260个可训练的参数, 接近九百五十万
    """
    def __init__(self, training):
        super().__init__()

        self.training = training

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(1, 16, 3, 1, padding=1),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5),
            nn.PReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.PReLU(32)
        )

        # 最后大尺度下的映射（256*256），下面的尺度依次递减
        self.map4 = nn.Sequential(
            nn.Conv3d(32, 1, 1, 1),
            # nn.Upsample(scale_factor=(1, 2, 2)),
            nn.Sigmoid()
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, 1, 1, 1),
            # nn.Upsample(scale_factor=(2, 4, 4)),
            nn.Sigmoid()
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, 1, 1, 1),
            # nn.Upsample(scale_factor=(4, 8, 8)),
            nn.Sigmoid()
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, 1, 1, 1),
            # nn.Upsample(scale_factor=(8, 16, 16)),
            nn.Sigmoid()
        )

    def forward(self, inputs):

        long_range1 = self.encoder_stage1(inputs) + inputs

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        # long_range2 = F.dropout(long_range2, 0.3, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        # long_range3 = F.dropout(long_range3, 0.3, self.training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        # long_range4 = F.dropout(long_range4, 0.3, self.training)

        short_range4 = self.down_conv4(long_range4)

        outputs = self.decoder_stage1(long_range4) + short_range4
        # outputs = F.dropout(outputs, 0.3, self.training)

        output1 = self.map1(outputs)

        short_range6 = self.up_conv2(outputs)

        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        # outputs = F.dropout(outputs, 0.3, self.training)

        output2 = self.map2(outputs)

        short_range7 = self.up_conv3(outputs)

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        # outputs = F.dropout(outputs, 0.3, self.training)

        output3 = self.map3(outputs)

        short_range8 = self.up_conv4(outputs)

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        output4 = self.map4(outputs)
        # print(type(output4))
        # if self.training is True:
        #     return output1, output2, output3, output4
        # else:
        return output4

class VNet2(nn.Module):

    def __init__(self, training):
        super().__init__()

        self.training = training

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(1, 16, 3, 1, padding=1),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5),
            nn.PReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.PReLU(32)
        )

        # 最后大尺度下的映射（256*256），下面的尺度依次递减
        self.map4 = nn.Sequential(
            nn.Conv3d(32, 1, 1, 1),
            # nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Sigmoid()
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, 1, 1, 1),
            # nn.Upsample(scale_factor=(2, 4, 4), mode='trilinear'),
            nn.Sigmoid()
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, 1, 1, 1),
            # nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'),
            nn.Sigmoid()
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, 1, 1, 1),
            # nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear'),
            nn.Sigmoid()
        )

    def forward(self, inputs):

        long_range1 = self.encoder_stage1(inputs) + inputs

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, 0.3, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, 0.3, self.training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, 0.3, self.training)

        short_range4 = self.down_conv4(long_range4)

        outputs = self.decoder_stage1(long_range4) + short_range4
        outputs = F.dropout(outputs, 0.3, self.training)

        output1 = self.map1(outputs)

        short_range6 = self.up_conv2(outputs)

        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = F.dropout(outputs, 0.3, self.training)

        output2 = self.map2(outputs)

        short_range7 = self.up_conv3(outputs)

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, 0.3, self.training)

        output3 = self.map3(outputs)

        short_range8 = self.up_conv4(outputs)

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        output4 = self.map4(outputs)

        return output4

class conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation_func=nn.ReLU):
        """
        + Instantiate modules: conv-relu-norm
        + Assign them as member variables
        """
        super(conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=1)
        self.relu = activation_func()
        # with learnable parameters
        # self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class conv3d_x3(nn.Module):
    """Three serial convs with a residual connection.
    Structure:
        inputs --> ① --> ② --> ③ --> outputs
                   ↓ --> add--> ↑
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(conv3d_x3, self).__init__()
        self.conv_1 = conv3d(in_channels, out_channels, kernel_size)
        self.conv_2 = conv3d(out_channels, out_channels, kernel_size)
        self.conv_3 = conv3d(out_channels, out_channels, kernel_size)

    def forward(self, x):
        z_1 = self.conv_1(x)
        z_3 = self.conv_3(self.conv_2(z_1))
        return z_1 + z_3


class deconv3d_x3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, activation_func=nn.ReLU):
        super(deconv3d_x3, self).__init__()
        self.up = deconv3d_as_up(in_channels, out_channels, kernel_size, stride)
        self.lhs_conv = conv3d(out_channels // 2, out_channels, kernel_size)
        self.conv_x3 = conv3d_x3(out_channels, out_channels, kernel_size)

    def forward(self, lhs, rhs):
        rhs_up = self.up(rhs)
        lhs_conv = self.lhs_conv(lhs)
        rhs_add = crop(rhs_up, lhs_conv) + lhs_conv
        return self.conv_x3(rhs_add)


def crop(large, small):
    """large / small with shape [batch_size, channels, depth, height, width]"""

    l, s = large.size(), small.size()
    offset = [0, 0, (l[2] - s[2]) // 2, (l[3] - s[3]) // 2, (l[4] - s[4]) // 2]
    return large[..., offset[2]: offset[2] + s[2], offset[3]: offset[3] + s[3], offset[4]: offset[4] + s[4]]


def conv3d_as_pool(in_channels, out_channels, kernel_size=3, stride=2, activation_func=nn.ReLU):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=1),
        activation_func())


def deconv3d_as_up(in_channels, out_channels, kernel_size=3, stride=2, activation_func=nn.ReLU):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride),
        activation_func()
    )


class softmax_out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(softmax_out, self).__init__()
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0)
        # if criterion == 'nll':
        #     self.softmax = F.log_softmax
        # else:
        #     assert criterion == 'dice', "Expect `dice` (dice loss) or `nll` (negative log likelihood loss)."
        self.softmax = F.sigmoid
        self.Up = nn.Upsample(scale_factor=(1,2,2), mode='nearest')

    def forward(self, x):
        """Output with shape [batch_size, 1, depth, height, width]."""
        # Do NOT add normalize layer, or its values vanish.
        y_conv = self.conv_2(self.conv_1(x))
        # Put channel axis in the last dim for softmax.
        # print("y_conv: ",y_conv.shape)
        # y_conv = self.Up(y_conv)#.view(-1, 2)
        # print("y_up: ",y_conv.shape)
        y_perm = y_conv.permute(0, 2, 3, 4, 1).contiguous()
        # y_perm = y_perm.squeeze(dim=4)
        # print("y_perm: ",y_perm.shape)
        y_flat = y_perm[:,:,:,:,0]
        # y_flat2 = y_perm[:,:,:,:,1]
        print('0:',F.sigmoid(y_flat[0,10,100,100]))
        # print('1:',F.sigmoid(y_flat2[0,10,100,100]))
        # print(self.softmax(y_perm).shape)
        return self.softmax(y_flat)


class VNet3(nn.Module):
    def __init__(self, arg):
        super(VNet3, self).__init__()
        self.conv_1 = conv3d_x3(1, 16)
        self.pool_1 = conv3d_as_pool(16, 32)
        self.conv_2 = conv3d_x3(32, 32)
        self.pool_2 = conv3d_as_pool(32, 64)
        self.conv_3 = conv3d_x3(64, 64)
        self.pool_3 = conv3d_as_pool(64, 128)
        self.conv_4 = conv3d_x3(128, 128)
        self.pool_4 = conv3d_as_pool(128, 256)

        self.bottom = conv3d_x3(256, 256)

        self.deconv_4 = deconv3d_x3(256, 256)
        self.deconv_3 = deconv3d_x3(256, 128)
        self.deconv_2 = deconv3d_x3(128, 64)
        self.deconv_1 = deconv3d_x3(64, 32)

        self.out = softmax_out(32, 1)

    def forward(self, x):
        conv_1 = self.conv_1(x)
        pool = self.pool_1(conv_1)
        conv_2 = self.conv_2(pool)
        pool = self.pool_2(conv_2)
        conv_3 = self.conv_3(pool)
        pool = self.pool_3(conv_3)
        conv_4 = self.conv_4(pool)
        pool = self.pool_4(conv_4)
        bottom = self.bottom(pool)
        deconv = self.deconv_4(conv_4, bottom)
        deconv = self.deconv_3(conv_3, deconv)
        deconv = self.deconv_2(conv_2, deconv)
        deconv = self.deconv_1(conv_1, deconv)
        return self.out(deconv)

def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        # super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), 0)
        out = self.relu1(out)   #torch.add(out, x16)
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 1, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(2)
        self.conv2 = nn.Conv3d(1, 1, kernel_size=1)
        self.relu1 = ELUCons(elu, 2)
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        # make channels the last axis
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        out = out.squeeze(dim=4)
        # flatten
        # out = out.view(out.numel() // 2, 2)
        out = self.softmax(out)
        # treat channel 0 as the predicted output
        return out


class VNet4(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self,args):
        super(VNet4, self).__init__()
        elu=True
        nll=False
        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, elu, nll)

    # The network topology as described in the diagram
    # in the VNet paper
    # def __init__(self):
    #     super(VNet, self).__init__()
    #     self.in_tr =  InputTransition(16)
    #     # the number of convolutions in each layer corresponds
    #     # to what is in the actual prototxt, not the intent
    #     self.down_tr32 = DownTransition(16, 2)
    #     self.down_tr64 = DownTransition(32, 3)
    #     self.down_tr128 = DownTransition(64, 3)
    #     self.down_tr256 = DownTransition(128, 3)
    #     self.up_tr256 = UpTransition(256, 3)
    #     self.up_tr128 = UpTransition(128, 3)
    #     self.up_tr64 = UpTransition(64, 2)
    #     self.up_tr32 = UpTransition(32, 1)
    #     self.out_tr = OutputTransition(16)
    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out

class DIYNet(nn.Module):#3dunet
    """
    3d unet

    """
    def __init__(self, training):
        super().__init__()

        self.training = training
        self.in_channel =1
        n_classes=1
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.encoder_stage1 = self.encoder(self.in_channel, filters[0], bias=False, batchnorm=False)

        self.encoder_stage2 = self.encoder(filters[0], filters[1], bias=False, batchnorm=False)

        self.encoder_stage3 = self.encoder(filters[1], filters[1], bias=False, batchnorm=False)

        self.encoder_stage4 = self.encoder(filters[1], filters[2], bias=False, batchnorm=False)

        self.encoder_stage5 = self.encoder(filters[2], filters[2], bias=False, batchnorm=False)

        self.encoder_stage6 = self.encoder(filters[2], filters[3], bias=False, batchnorm=False)

        self.encoder_stage7 = self.encoder(filters[3], filters[3], bias=False, batchnorm=False)

        self.encoder_stage8 = self.encoder(filters[3], filters[4], bias=False, batchnorm=False)

        self.decoder_stage10 = self.decoder(filters[4], filters[4], kernel_size=2, stride=2, bias=False)

        self.decoder_stage9 = self.decoder(filters[3] + filters[4], filters[3], kernel_size=3, stride=1, padding=1, bias=False)

        self.decoder_stage8 = self.decoder(filters[3], filters[3], kernel_size=3, stride=1, padding=1, bias=False)

        self.decoder_stage7 = self.decoder(filters[3], filters[3], kernel_size=2, stride=2, bias=False)

        self.decoder_stage6 = self.decoder(filters[2] + filters[3], filters[2], kernel_size=3, stride=1, padding=1, bias=False)

        self.decoder_stage5 = self.decoder(filters[2], filters[2], kernel_size=3, stride=1, padding=1, bias=False)

        self.decoder_stage4 = self.decoder(filters[2], filters[2], kernel_size=2, stride=2, bias=False)

        self.decoder_stage3 = self.decoder(filters[1] + filters[2], filters[1], kernel_size=3, stride=1, padding=1, bias=False)

        self.decoder_stage2 = self.decoder(filters[1], filters[1], kernel_size=3, stride=1, padding=1, bias=False)

        self.decoder_stage1 = self.decoder(filters[1], n_classes, kernel_size=1, stride=1, bias=False)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool3d(2)

        # self.down_conv1 = nn.Sequential(
        #     nn.Conv3d(16, 32, 2, 2),
        #     nn.ReLU(32)
        # )
        #
        # self.down_conv2 = nn.Sequential(
        #     nn.Conv3d(32, 64, 2, 2),
        #     nn.ReLU(64)
        # )
        #
        # self.down_conv3 = nn.Sequential(
        #     nn.Conv3d(64, 128, 2, 2),
        #     nn.ReLU(128)
        # )
        #
        # self.down_conv4 = nn.Sequential(
        #     nn.Conv3d(128, 256, 3, 1, padding=1),
        #     nn.ReLU(256)
        # )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(filters[4], filters[4], 2, 2),
            nn.ReLU()
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(filters[3], filters[3], 2, 2),
            nn.ReLU()
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(filters[2], filters[2], 2, 2),
            nn.ReLU()
        )

        # 最后大尺度下的映射（256*256），下面的尺度依次递减
        self.map4 = nn.Sequential(
            nn.Conv3d(filters[1], 1, 1, 1),
            # nn.Upsample(scale_factor=(1, 2, 2)),
            nn.Sigmoid()
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(filters[2], 1, 1, 1),
            # nn.Upsample(scale_factor=(2, 4, 4)),
            nn.Sigmoid()
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(filters[3], 1, 1, 1),
            # nn.Upsample(scale_factor=(4, 8, 8)),
            nn.Sigmoid()
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(filters[4], 1, 1, 1),
            # nn.Upsample(scale_factor=(8, 16, 16)),
            nn.Sigmoid()
        )
        self.v0 = nn.Conv3d(filters[1], 1, 1, 1)
        # self.v1 = nn.Conv3d(nb_filter[0], 1, 1, 1)
        # self.v2 = nn.Conv3d(nb_filter[0], 1, 1, 1)
        # self.v3 = nn.Conv3d(nb_filter[0], 1, 1, 1)
        self.v4 = nn.Conv3d(filters[1], 1, 1, 1)
    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=1,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            # nn.Upsample(scale_factor=2),    #尺寸扩大2倍
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),    #通道数改变
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
            # nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
            #                    padding=padding, output_padding=output_padding, bias=bias),
            # nn.ReLU()
            )
        return layer

    def forward(self, inputs):

        short_range1 = self.encoder_stage1(inputs)
        long_range1 = self.encoder_stage2(short_range1)
        long_range2 = self.pool0(long_range1)
        # long_range2 = F.dropout(long_range2, 0.3, self.training)

        short_range2 = self.encoder_stage3(long_range2)
        long_range3 = self.encoder_stage4(short_range2)
        long_range4 = self.pool1(long_range3)


        # long_range3 = F.dropout(long_range3, 0.3, self.training)
        short_range3 = self.encoder_stage5(long_range4)
        long_range5 = self.encoder_stage6(short_range3)
        long_range6 = self.pool2(long_range5)

        short_range4 = self.encoder_stage7(long_range6)
        long_range7 = self.encoder_stage8(short_range4)
        # long_range4 = F.dropout(long_range4, 0.3, self.training)

        short_range6 = self.up_conv2(long_range7)
        # print(short_range6.shape, long_range5.shape)
        long_range8 = self.decoder_stage9(torch.cat([short_range6, long_range5], dim=1))
        long_range9 = self.decoder_stage8(long_range8)
        # outputs = F.dropout(outputs, 0.3, self.training)

        short_range7 = self.up_conv3(long_range9)
        # print(short_range7.shape, long_range3.shape)
        long_range10 = self.decoder_stage6(torch.cat([short_range7, long_range3], dim=1))
        long_range11 = self.decoder_stage5(long_range10)

        short_range8 = self.up_conv4(long_range11)
        # print(short_range8.shape, long_range1.shape)
        long_range12 = self.decoder_stage3(torch.cat([short_range8, long_range1], dim=1))
        long_range13 = self.decoder_stage2(long_range12)

        # output1 = self.map1(outputs)
        #
        #
        #
        #
        # # outputs = F.dropout(outputs, 0.3, self.training)
        #
        # output2 = self.map2(outputs)
        #
        # short_range7 = self.up_conv3(outputs)
        #
        # outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1))
        # # outputs = F.dropout(outputs, 0.3, self.training)
        #
        # output3 = self.map3(outputs)
        #
        # short_range8 = self.up_conv4(outputs)
        #
        # outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1))
        #
        # fv0= self.v0(long_range1)
        # # fv1= self.v1(x0_1)
        # # fv2= self.v2(x0_2)
        # # fv3= self.v3(x0_3)
        # fv4= self.v4(long_range13)

        # imsave('feature visualization/'+str(i)+'-x00.png', (fv0[0,0,0,:,:].cpu()))
        # # imsave('feature visualization/'+str(i)+'-x01.png', (fv1[0,0,0,:,:].cpu()))
        # # imsave('feature visualization/'+str(i)+'-x02.png', (fv2[0,0,0,:,:].cpu()))
        # # imsave('feature visualization/'+str(i)+'-x03.png', (fv3[0,0,0,:,:].cpu()))
        # imsave('feature visualization/'+str(i)+'-x04.png', (fv4[0,0,0,:,:].cpu()))

        output4 = self.map4(long_range13)
        # print(type(output4))
        # if self.training is True:
        #     return output1, output2, output3, output4
        # else:
        return output4

class ResUNet(nn.Module):
    """
    共9498260个可训练的参数, 接近九百五十万
    """
    def __init__(self, training):
        super().__init__()

        self.training = training

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(1, 16, 3, 1, padding=1),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5),
            nn.PReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.PReLU(32)
        )

        # 最后大尺度下的映射（256*256），下面的尺度依次递减
        self.map4 = nn.Sequential(
            nn.Conv3d(32, 1, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Sigmoid()
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, 1, 1, 1),
            nn.Upsample(scale_factor=(2, 4, 4), mode='trilinear'),
            nn.Sigmoid()
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, 1, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'),
            nn.Sigmoid()
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, 1, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear'),
            nn.Sigmoid()
        )

    def forward(self, inputs):

        long_range1 = self.encoder_stage1(inputs) + inputs

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + trainingshort_range1
        long_range2 = F.dropout(long_range2, para.drop_rate, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, para.drop_rate, self.training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, para.drop_rate, self.training)

        short_range4 = self.down_conv4(long_range4)

        outputs = self.decoder_stage1(long_range4) + short_range4
        outputs = F.dropout(outputs, para.drop_rate, self.training)

        output1 = self.map1(outputs)

        short_range6 = self.up_conv2(outputs)

        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = F.dropout(outputs, 0.3, self.training)

        output2 = self.map2(outputs)

        short_range7 = self.up_conv3(outputs)

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, 0.3, self.training)

        output3 = self.map3(outputs)

        short_range8 = self.up_conv4(outputs)

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        output4 = self.map4(outputs)

        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4

class Recurrent_block(nn.Module):
    def __init__(self,in_ch,out_ch,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv3d(out_ch,out_ch,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm3d(out_ch),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        # print(x.shape)
        for i in range(self.t):
            if i==0:
                x1 = self.conv(x)
            # print(x1.shape)
            x1 = self.conv(x+x1)
        return x1

class RRCNN_block(nn.Module):
    def __init__(self,in_ch,out_ch,t=3):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(out_ch,out_ch,t=t),
            Recurrent_block(out_ch,out_ch,t=t)
        )
        self.Conv_1x1 = nn.Conv3d(in_ch,out_ch,kernel_size=1,stride=1,padding=0)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.Conv_1x1(x)
        x = self.bn1(x)
        # print("x",x.shape,x[1,1,100,100])
        x1 = self.RCNN(x)
        # print("x1",x1.shape,x1[1,1,100,100])
        result = x+x1
        # print("result",result.shape,result[1,1,100,100])
        return result


class single_conv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv2(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(up_conv2,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_ch,out_ch,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm3d(out_ch),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class R2U_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1,t=2):
        super(R2U_Net,self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)
        # self.Upsample = nn.Upsample(scale_factor=2)
        n1 = 16#32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.RRCNN1 = RRCNN_block(in_ch=1,out_ch=filters[0],t=2)

        self.RRCNN2 = RRCNN_block(in_ch=filters[0],out_ch=filters[1],t=t)

        self.RRCNN3 = RRCNN_block(in_ch=filters[1],out_ch=filters[2],t=t)

        self.RRCNN4 = RRCNN_block(in_ch=filters[2],out_ch=filters[3],t=t)

        self.RRCNN5 = RRCNN_block(in_ch=filters[3],out_ch=filters[4],t=t)


        self.Up5 = up_conv2(in_ch=filters[4],out_ch=filters[3])
        self.Up_RRCNN5 = RRCNN_block(in_ch=filters[4], out_ch=filters[3],t=t)

        self.Up4 = up_conv2(in_ch=filters[3],out_ch=filters[2])
        self.Up_RRCNN4 = RRCNN_block(in_ch=filters[3], out_ch=filters[2],t=t)

        self.Up3 = up_conv2(in_ch=filters[2],out_ch=filters[1])
        self.Up_RRCNN3 = RRCNN_block(in_ch=filters[2], out_ch=filters[1],t=t)

        self.Up2 = up_conv2(in_ch=filters[1],out_ch=filters[0])
        self.Up_RRCNN2 = RRCNN_block(in_ch=filters[1], out_ch=filters[0],t=t)

        self.Conv_1x1 = nn.Conv3d(filters[0],output_ch,kernel_size=1,stride=1,padding=0)

        self.final = nn.Sequential(
            nn.Conv3d(filters[0], 1, 1, 1,padding=0),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        ) # 卷积个数，输出通道数，kernel尺寸，stride，pad

    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.final(d2)#Conv_1x1(d2)#

        return d1

'''
    UNet 3+ with deep supervision
'''
class UNet3Plus(nn.Module):
    def __init__(self, args):
        # def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet3Plus, self).__init__()
        self.args = args
        self.is_deconv = True
        self.feature_scale = 4
        n_classes=1
        filters = [32,64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        self.conv1 = Conv_Block(args.input_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = Conv_Block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = Conv_Block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = Conv_Block(filters[2], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = Conv_Block(filters[3], filters[4])

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32,mode='bilinear')###
        self.upscore5 = nn.Upsample(scale_factor=16,mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8,mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4,mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        # DeepSup
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv5 = nn.Conv2d(filters[4], n_classes, 3, padding=1)

        # initialise weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init_weights(m, init_type='kaiming')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5) # 16->256

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4) # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3) # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2) # 128->256

        d1 = self.outconv1(hd1) # 256
        # imsave('feature visualization/x00.png', (d1[0,0,:,:].cpu()))
        # imsave('feature visualization/x01.png', (d2[0,0,:,:].cpu()))
        # imsave('feature visualization/x02.png', (d3[0,0,:,:].cpu()))
        # imsave('feature visualization/x03.png', (d4[0,0,:,:].cpu()))
        # imsave('feature visualization/x04.png', (d5[0,0,:,:].cpu()))
        return F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4)#, F.sigmoid(d5)

class UNet3Plus3D(nn.Module):
    def __init__(self, args):
        # def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet3Plus3D, self).__init__()
        self.args = args
        self.is_deconv = True
        self.feature_scale = 4
        n_classes=1
        filters = [8,16,32,64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        self.conv1 = Conv_Block_3d(args.input_channels, filters[0])
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = Conv_Block_3d(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = Conv_Block_3d(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)

        self.conv4 = Conv_Block_3d(filters[2], filters[3])
        self.maxpool4 = nn.MaxPool3d(kernel_size=2)

        self.conv5 = Conv_Block_3d(filters[3], filters[4])

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool3d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool3d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv3d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv3d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv3d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='nearest')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv3d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool3d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv3d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv3d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='nearest')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='nearest')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv3d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv3d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='nearest')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='nearest')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='nearest')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv3d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='nearest')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='nearest')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='nearest')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='nearest')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv3d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32,mode='nearest')###
        self.upscore5 = nn.Upsample(scale_factor=16,mode='nearest')
        self.upscore4 = nn.Upsample(scale_factor=8,mode='nearest')
        self.upscore3 = nn.Upsample(scale_factor=4,mode='nearest')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='nearest')

        # DeepSup
        self.outconv1 = nn.Conv3d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv3d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv3d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv3d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv5 = nn.Conv3d(filters[4], n_classes, 3, padding=1)


        # initialise weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init_weights(m, init_type='kaiming')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        a = self.hd5_UT_hd4(hd5)
        b = self.hd5_UT_hd4_conv(a)
        c = self.hd5_UT_hd4_bn(b)
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(c)
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5) # 16->256

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4) # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3) # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2) # 128->256

        d1 = self.outconv1(hd1) # 256

        # fv0= self.v0(d5)
        # fv1= self.v1(d4)
        # fv2= self.v2(d3)
        # fv3= self.v3(d2)
        # fv4= self.v4(d1)
        # imsave('feature visualization/'+str(i)+'-x00.png', (d5[0,0,0,:,:].cpu()))
        # imsave('feature visualization/'+str(i)+'-x01.png', (d4[0,0,0,:,:].cpu()))
        # imsave('feature visualization/'+str(i)+'-x02.png', (d3[0,0,0,:,:].cpu()))
        # imsave('feature visualization/'+str(i)+'-x03.png', (d2[0,0,0,:,:].cpu()))
        # imsave('feature visualization/'+str(i)+'-x04.png', (d1[0,0,0,:,:].cpu()))

        # imsave('feature visualization/x00.png', (d1[0,0,:,:].cpu()))
        # imsave('feature visualization/x01.png', (d2[0,0,:,:].cpu()))
        # imsave('feature visualization/x02.png', (d3[0,0,:,:].cpu()))
        # imsave('feature visualization/x03.png', (d4[0,0,:,:].cpu()))
        # imsave('feature visualization/x04.png', (d5[0,0,:,:].cpu()))
        return F.sigmoid(d4), F.sigmoid(d3), F.sigmoid(d2), F.sigmoid(d1), #, F.sigmoid(d5)

class UNetPlusPlus3D(nn.Module):
    """
    UNet++ - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        n1 = 16#32
        nb_filter = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#线性插值
        self.Up5 = self.up_conv3d(nb_filter[4], nb_filter[3])
        self.Up4 = self.up_conv3d(nb_filter[3], nb_filter[2])
        self.Up3 = self.up_conv3d(nb_filter[2], nb_filter[1])
        self.Up2 = self.up_conv3d(nb_filter[1], nb_filter[0])

        self.conv0_0 = self.Conv_Block3d(args.input_channels, nb_filter[0])
        self.conv1_0 = self.Conv_Block3d(nb_filter[0], nb_filter[1])
        self.conv2_0 = self.Conv_Block3d(nb_filter[1], nb_filter[2])
        self.conv3_0 = self.Conv_Block3d(nb_filter[2], nb_filter[3])
        self.conv4_0 = self.Conv_Block3d(nb_filter[3], nb_filter[4])

        self.conv0_1 = self.Conv_Block3d(nb_filter[0]+nb_filter[0], nb_filter[0])
        self.conv1_1 = self.Conv_Block3d(nb_filter[1]+nb_filter[1], nb_filter[1])
        self.conv2_1 = self.Conv_Block3d(nb_filter[2]+nb_filter[2], nb_filter[2])
        self.conv3_1 = self.Conv_Block3d(nb_filter[3]+nb_filter[3], nb_filter[3])

        self.conv0_2 = self.Conv_Block3d(nb_filter[0]*3, nb_filter[0])
        self.conv1_2 = self.Conv_Block3d(nb_filter[1]*3, nb_filter[1])
        self.conv2_2 = self.Conv_Block3d(nb_filter[2]*3, nb_filter[2])

        self.conv0_3 = self.Conv_Block3d(nb_filter[0]*4, nb_filter[0])
        self.conv1_3 = self.Conv_Block3d(nb_filter[1]*4, nb_filter[1])

        self.conv0_4 = self.Conv_Block3d(nb_filter[0]*5, nb_filter[0])


        # if self.args.deepsupervision:
        # self.final1 = nn.Sequential(
        #     nn.Conv2d(nb_filter[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     #nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final2 = nn.Sequential(
        #     nn.Conv2d(nb_filter[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     #nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final3 = nn.Sequential(
        #     nn.Conv2d(nb_filter[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     #nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # self.final4 = nn.Sequential(
        #     nn.Conv2d(nb_filter[0], 1, 1, 1),
        #     # nn.ConvTranspose2d(1, 1, 2, 2),
        #     #nn.Upsample(scale_factor=2,mode='nearest'),
        #     nn.Sigmoid()
        # )
        # else:
        # self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        # self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        # self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        # self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

        self.final1 = nn.Sequential(
            nn.Conv3d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            #nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final2 = nn.Sequential(
            nn.Conv3d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            #nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final3 = nn.Sequential(
            nn.Conv3d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            #nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final4 = nn.Sequential(
            nn.Conv3d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            #nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
       # self.active = torch.nn.Sigmoid()
    def Conv_Block3d(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            # nn.Upsample(scale_factor=2),    #尺寸扩大2倍
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias),    #通道数改变
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias),    #通道数改变
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
            )
            # nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
            #                    padding=padding, output_padding=output_padding, bias=bias),
            # nn.ReLU())
        return layer
    def up_conv3d(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            # nn.Upsample(scale_factor=(2,2,2))#,    #尺寸扩大2倍
            # nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, bias=bias),    #通道数改变
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True)

            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU()
            )
        return layer

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.Maxpool1(x0_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up2(x1_0)], 1))

        x2_0 = self.conv2_0(self.Maxpool2(x1_0))
        # print('before up3 ',x2_0.shape)#[B,128,64,64]
        # print('after up3 ',self.Up3(x2_0).shape)    #[B,64,128,128]
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up3(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up2(x1_1)], 1))

        x3_0 = self.conv3_0(self.Maxpool3(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up4(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up3(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up2(x1_2)], 1))

        x4_0 = self.conv4_0(self.Maxpool4(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up5(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up4(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up3(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up2(x1_3)], 1))

        # if self.args.deepsupervision:
        output1 = self.final1(x0_1)
        # print(output1[0,0,:,:])
        output2 = self.final2(x0_2)
        output3 = self.final3(x0_3)
        output4 = self.final4(x0_4)

        # fv0= self.v0(x0_0)
        # fv1= self.v1(x0_1)
        # fv2= self.v2(x0_2)
        # fv3= self.v3(x0_3)
        # fv4= self.v4(x0_4)
        #
        # imsave('feature visualization/'+str(i)+'-x00.png', (fv0[0,0,0,:,:].cpu()))
        # imsave('feature visualization/'+str(i)+'-x01.png', (fv1[0,0,0,:,:].cpu()))
        # imsave('feature visualization/'+str(i)+'-x02.png', (fv2[0,0,0,:,:].cpu()))
        # imsave('feature visualization/'+str(i)+'-x03.png', (fv3[0,0,0,:,:].cpu()))
        # imsave('feature visualization/'+str(i)+'-x04.png', (fv4[0,0,0,:,:].cpu()))

        # imsave('output/coutput1.png', (output1[0,0,:,:].cpu().detach().numpy()))
        # imsave('output/coutput2.png', (output2[0,0,:,:].cpu().detach().numpy()))
        # imsave('output/coutput3.png', (output3[0,0,:,:].cpu().detach().numpy()))
        # imsave('output/coutput4.png', (output4[0,0,:,:].cpu().detach().numpy()))
        # print(output4.shape)  #[B,1,256,256]
        return [output1, output2, output3, output4]

    # else:
        # output = self.final4(x0_4)
        # imsave('output/coutput_final.png', (output[0,0,:,:].cpu().detach().numpy()))
        # return output

class AttentionUNet3D(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        filters = [32, 64, 128, 256, 512, 1024]

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(2, 2)
        self.Maxpool3 = nn.MaxPool3d(2, 2)
        self.Maxpool4 = nn.MaxPool3d(2, 2)

        self.Up5 = self.up_conv(filters[4], filters[3])
        self.Up4 = self.up_conv(filters[3], filters[2])
        self.Up3 = self.up_conv(filters[2], filters[1])
        self.Up2 = self.up_conv(filters[1], filters[0])

        self.conv0_0 = self.Conv_Block(args.input_channels, filters[0])
        self.conv1_0 = self.Conv_Block(filters[0], filters[1])
        self.conv2_0 = self.Conv_Block(filters[1], filters[2])
        self.conv3_0 = self.Conv_Block(filters[2], filters[3])
        self.conv4_0 = self.Conv_Block(filters[3], filters[4])


        # self.final = nn.Conv2d(filters[0], 1, kernel_size=1, stride=1, padding=0)
        self.final = nn.Sequential(
            nn.Conv3d(filters[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            #nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.Att5 = Attention_block_3d(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Att4 = Attention_block_3d(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Att3 = Attention_block_3d(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Att2 = Attention_block_3d(F_g=filters[0], F_l=filters[0], F_int=32)

        self.Up_conv5 = self.Conv_Block(filters[3]+filters[3], filters[3])
        self.Up_conv4 = self.Conv_Block(filters[2]+filters[2], filters[2])
        self.Up_conv3 = self.Conv_Block(filters[1]+filters[1], filters[1])
        self.Up_conv2 = self.Conv_Block(filters[0]+filters[0], filters[0])

    def up_conv(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            # nn.Upsample(scale_factor=(2,2,2))#,    #尺寸扩大2倍
            # nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, bias=bias),    #通道数改变
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True)

            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU()
            )
        return layer
    def Conv_Block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            # nn.Upsample(scale_factor=2),    #尺寸扩大2倍
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias),    #通道数改变
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias),    #通道数改变
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
            )
            # nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
            #                    padding=padding, output_padding=output_padding, bias=bias),
            # nn.ReLU())
        return layer
    def forward(self, input):
        # encoding path
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.Maxpool1(x0_0))
        x2_0 = self.conv2_0(self.Maxpool2(x1_0))
        x3_0 = self.conv3_0(self.Maxpool3(x2_0))
        x4_0 = self.conv4_0(self.Maxpool4(x3_0))

        # decoding + concat path
        tmp4=self.Up5(x4_0)
        att4=self.Att5(g=tmp4,x=x3_0)
        # print('att4:',att4.shape)
        # print('tmp4:',tmp4.shape)
        x3_1 = torch.cat([att4,tmp4], 1)
        x3_1 = self.Up_conv5(x3_1)

        tmp3=self.Up4(x3_1)
        att3=self.Att4(g=tmp3,x=x2_0)
        x2_2 = torch.cat([att3, tmp3], 1)
        x2_2 = self.Up_conv4(x2_2)

        tmp2=self.Up3(x2_2)
        att2=self.Att3(g=tmp2,x=x1_0)
        x1_3 = torch.cat([att2, tmp2], 1)
        x1_3 = self.Up_conv3(x1_3)

        tmp1=self.Up2(x1_3)
        att1=self.Att2(g=tmp1,x=x0_0)
        x0_4 = torch.cat([att1, tmp1], 1)
        x0_4 = self.Up_conv2(x0_4)

        output = self.final(x0_4)
        # self.epoch+=1
        # print(epoch)
        return output

class AttentionUNetPlusPlus3D(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        nb_filter = [8,16,32,64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool3d(2, 2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Up5 = self.up_conv(nb_filter[4], nb_filter[3])
        self.Up4 = self.up_conv(nb_filter[3], nb_filter[2])
        self.Up3 = self.up_conv(nb_filter[2], nb_filter[1])
        self.Up2 = self.up_conv(nb_filter[1], nb_filter[0])

        self.conv0_0 = self.Conv_Block(args.input_channels, nb_filter[0])
        self.conv1_0 = self.Conv_Block(nb_filter[0], nb_filter[1])
        self.conv2_0 = self.Conv_Block(nb_filter[1], nb_filter[2])
        self.conv3_0 = self.Conv_Block(nb_filter[2], nb_filter[3])
        self.conv4_0 = self.Conv_Block(nb_filter[3], nb_filter[4])

        self.conv0_1 = self.Conv_Block(nb_filter[1], nb_filter[0])
        self.conv1_1 = self.Conv_Block(nb_filter[2], nb_filter[1])
        self.conv2_1 = self.Conv_Block(nb_filter[3], nb_filter[2])
        self.conv3_1 = self.Conv_Block(nb_filter[4], nb_filter[3])

        self.conv0_2 = self.Conv_Block(nb_filter[0]*1+nb_filter[1], nb_filter[0])
        self.conv1_2 = self.Conv_Block(nb_filter[1]*1+nb_filter[2], nb_filter[1])
        self.conv2_2 = self.Conv_Block(nb_filter[2]*1+nb_filter[3], nb_filter[2])

        self.conv0_3 = self.Conv_Block(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_3 = self.Conv_Block(nb_filter[1]*2+nb_filter[2], nb_filter[1])

        self.conv0_4 = self.Conv_Block(nb_filter[0]*3+nb_filter[1], nb_filter[0])

        # self.Att4_2 = Attention_block(F_g=nb_filter[3], F_l=nb_filter[2]*2, F_int=nb_filter[1])
        # self.Att3_3 = Attention_block(F_g=nb_filter[2], F_l=nb_filter[1]*3, F_int=nb_filter[0])
        # self.Att2_4 = Attention_block(F_g=nb_filter[1], F_l=nb_filter[0]*4, F_int=32)

        self.Att0_1 = Attention_block_3d(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_1 = Attention_block_3d(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Att2_1 = Attention_block_3d(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])
        self.Att3_1 = Attention_block_3d(F_g=nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[2])

        self.Att0_2 = Attention_block_3d(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_2 = Attention_block_3d(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Att2_2 = Attention_block_3d(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])

        self.Att0_3 = Attention_block_3d(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_3 = Attention_block_3d(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])

        self.Att0_4 = Attention_block_3d(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)

        # self.Up_conv5 = Conv_Block(nb_filters[4], nb_filters[3])
        # self.Up_conv4 = Conv_Block(nb_filters[3], nb_filters[2])
        # self.Up_conv3 = Conv_Block(nb_filters[2], nb_filters[1])
        # self.Up_conv2 = Conv_Block(nb_filters[1], nb_filters[0])

        if self.args.deepsupervision:
            self.final1 = nn.Sequential(
                nn.Conv3d(nb_filter[0], 1, 1, 1),
                # nn.ConvTranspose2d(1, 1, 2, 2),
                # nn.Upsample(scale_factor=2,mode='nearest'),
                nn.Sigmoid()
            )
            self.final2 = nn.Sequential(
                nn.Conv3d(nb_filter[0], 1, 1, 1),
                # nn.ConvTranspose2d(1, 1, 2, 2),
                # nn.Upsample(scale_factor=2,mode='nearest'),
                nn.Sigmoid()
            )
            self.final3 = nn.Sequential(
                nn.Conv3d(nb_filter[0], 1, 1, 1),
                # nn.ConvTranspose2d(1, 1, 2, 2),
                # nn.Upsample(scale_factor=2,mode='nearest'),
                nn.Sigmoid()
            )
            self.final4 = nn.Sequential(
                nn.Conv3d(nb_filter[0], 1, 1, 1),
                # nn.ConvTranspose2d(1, 1, 2, 2),
                # nn.Upsample(scale_factor=2,mode='nearest'),
                nn.Sigmoid()
            )
        else:
            self.final = nn.Sequential(
                nn.Conv3d(nb_filter[0], 1, 1, 1),
                # nn.ConvTranspose2d(1, 1, 2, 2),
                # nn.Upsample(scale_factor=2,mode='nearest'),
                nn.Sigmoid()
            ) # 卷积个数，输出通道数，kernel尺寸，stride，pad

        self.v0 = nn.Conv3d(nb_filter[0], 1, 1, 1)
        self.v1 = nn.Conv3d(nb_filter[0], 1, 1, 1)
        self.v2 = nn.Conv3d(nb_filter[0], 1, 1, 1)
        self.v3 = nn.Conv3d(nb_filter[0], 1, 1, 1)
        self.v4 = nn.Conv3d(nb_filter[0], 1, 1, 1)
    def up_conv(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            # nn.Upsample(scale_factor=(2,2,2))#,    #尺寸扩大2倍
            # nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, bias=bias),    #通道数改变
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True)

            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU()
            )
        return layer
    def Conv_Block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            # nn.Upsample(scale_factor=2),    #尺寸扩大2倍
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias),    #通道数改变
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias),    #通道数改变
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
            )
            # nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
            #                    padding=padding, output_padding=output_padding, bias=bias),
            # nn.ReLU())
        return layer
    def forward(self, input):
        # encoding path
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # decoding + concat path
        # print(x1_0.shape)
        tmp0_1 = self.Up2(x1_0)
        # print("tmp0_1",tmp0_1.shape)
        att0_1=self.Att0_1(g=tmp0_1,x=x0_0)
        # print("x0_0",x0_0.shape)
        # print("att0_1",att0_1.shape)
        x0_1 = torch.cat([att0_1,tmp0_1], 1)
        # print(x0_1.shape)
        x0_1 = self.conv0_1(x0_1)

        # x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        tmp1_1 = self.Up3(x2_0)
        att1_1=self.Att1_1(g=tmp1_1,x=x1_0)
        x1_1 = torch.cat([att1_1,tmp1_1], 1)
        x1_1 = self.conv1_1(x1_1)

        # x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        tmp2_1 = self.Up4(x3_0)
        att2_1=self.Att2_1(g=tmp2_1,x=x2_0)
        x2_1 = torch.cat([att2_1,tmp2_1], 1)
        x2_1 = self.conv2_1(x2_1)

        tmp3_1=self.Up5(x4_0)
        att3_1=self.Att3_1(g=tmp3_1,x=x3_0)
        x3_1 = torch.cat([att3_1,tmp3_1], 1)
        x3_1 = self.conv3_1(x3_1)

        # x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))
        tmp0_2 = self.Up2(x1_1)
        att0_2=self.Att0_2(g=tmp0_2,x=x0_1)
        x0_2 = torch.cat([att0_2,tmp0_2,x0_0], 1)
        x0_2 = self.conv0_2(x0_2)

        # x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        tmp1_2 = self.Up3(x2_1)
        att1_2=self.Att1_2(g=tmp1_2,x=x1_1)
        x1_2 = torch.cat([att1_2,tmp1_2,x1_0], 1)
        x1_2 = self.conv1_2(x1_2)

        tmp2_2=self.Up4(x3_1)
        att2_2=self.Att2_2(g=tmp2_2,x=x2_1)
        x2_2 = torch.cat([att2_2, tmp2_2, x2_0], 1)
        x2_2 = self.conv2_2(x2_2)

        # x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))
        tmp0_3 = self.Up2(x1_2)
        att0_3=self.Att0_3(g=tmp0_3,x=x0_2)
        x0_3 = torch.cat([att0_3,tmp0_3,x0_0,x0_1], 1)
        x0_3 = self.conv0_3(x0_3)

        tmp1_3=self.Up3(x2_2)
        att1_3=self.Att1_3(g=tmp1_3,x=x1_2)
        x1_3 = torch.cat([att1_3, tmp1_3 ,x1_0 ,x1_1], 1)
        x1_3 = self.conv1_3(x1_3)

        tmp0_4=self.Up2(x1_3)
        att0_4 = self.Att0_4(g=tmp0_4,x=x0_3)
        x0_4 = torch.cat([att0_4, tmp0_4,x0_0, x0_1, x0_2], 1)
        x0_4 = self.conv0_4(x0_4)

        # fv0= self.v0(x0_0)
        # fv1= self.v1(x0_1)
        # fv2= self.v2(x0_2)
        # fv3= self.v3(x0_3)
        # fv4= self.v4(x0_4)

        # imsave('feature visualization/'+str(i)+'-x00.png', (fv0[0,0,0,:,:].cpu()))
        # imsave('feature visualization/'+str(i)+'-x01.png', (fv1[0,0,0,:,:].cpu()))
        # imsave('feature visualization/'+str(i)+'-x02.png', (fv2[0,0,0,:,:].cpu()))
        # imsave('feature visualization/'+str(i)+'-x03.png', (fv3[0,0,0,:,:].cpu()))
        # imsave('feature visualization/'+str(i)+'-x04.png', (fv4[0,0,0,:,:].cpu()))

        if self.args.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

class AttentionPointUNetPlusPlus3D(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        nb_filter = [32,64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Up5 = up_conv(nb_filter[4], nb_filter[3])
        self.Up4 = up_conv(nb_filter[3], nb_filter[2])
        self.Up3 = up_conv(nb_filter[2], nb_filter[1])
        self.Up2 = up_conv(nb_filter[1], nb_filter[0])

        self.conv0_0 = Conv_Block(args.input_channels, nb_filter[0])
        self.conv1_0 = Conv_Block(nb_filter[0], nb_filter[1])
        self.conv2_0 = Conv_Block(nb_filter[1], nb_filter[2])
        self.conv3_0 = Conv_Block(nb_filter[2], nb_filter[3])
        self.conv4_0 = Conv_Block(nb_filter[3], nb_filter[4])

        self.conv0_1 = Conv_Block(nb_filter[0]+nb_filter[0], nb_filter[0])
        self.conv1_1 = Conv_Block(nb_filter[1]*3, nb_filter[1])
        self.conv2_1 = Conv_Block(nb_filter[2]*3, nb_filter[2])
        self.conv3_1 = Conv_Block(nb_filter[3]*3, nb_filter[3])

        self.conv0_2 = Conv_Block(nb_filter[0]*4, nb_filter[0])
        self.conv1_2 = Conv_Block(nb_filter[1]*4, nb_filter[1])
        self.conv2_2 = Conv_Block(nb_filter[2]*4, nb_filter[2])

        self.conv0_3 = Conv_Block(nb_filter[0]*5, nb_filter[0])
        self.conv1_3 = Conv_Block(nb_filter[1]*5, nb_filter[1])

        self.conv0_4 = Conv_Block(nb_filter[0]*6, nb_filter[0])

        # self.Att4_2 = Attention_block(F_g=nb_filter[3], F_l=nb_filter[2]*2, F_int=nb_filter[1])
        # self.Att3_3 = Attention_block(F_g=nb_filter[2], F_l=nb_filter[1]*3, F_int=nb_filter[0])
        # self.Att2_4 = Attention_block(F_g=nb_filter[1], F_l=nb_filter[0]*4, F_int=32)

        self.Att0_1 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_1 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Att2_1 = Attention_block4(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])
        self.Att3_1 = Attention_block4(F_g=nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[2])

        self.Att0_2 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_2 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Att2_2 = Attention_block4(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])

        self.Att0_3 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)
        self.Att1_3 = Attention_block4(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])

        self.Att0_4 = Attention_block4(F_g=nb_filter[0], F_l=nb_filter[0], F_int=32)

        self.head20_11 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        self.head11_02 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)

        self.head30_21 = PointHead(input_channels=nb_filter[3], out_channels=nb_filter[2],stride=4)
        self.head21_12 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        self.head12_03 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)

        self.head40_31 = PointHead(input_channels=nb_filter[4], out_channels=nb_filter[3],stride=8)
        self.head31_22 = PointHead(input_channels=nb_filter[3], out_channels=nb_filter[2],stride=4)
        self.head22_13 = PointHead(input_channels=nb_filter[2], out_channels=nb_filter[1],stride=2)
        self.head13_04 = PointHead(input_channels=nb_filter[1], out_channels=nb_filter[0],stride=1)
        # self.Up_conv5 = Conv_Block(nb_filter[4], nb_filters[3])
        # self.Up_conv4 = Conv_Block(nb_filter[3], nb_filters[2])
        # self.Up_conv3 = Conv_Block(nb_filter[2], nb_filters[1])
        # self.Up_conv2 = Conv_Block(nb_filters[1], nb_filters[0])

        self.final1 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final2 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final3 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        self.final4 = nn.Sequential(
            nn.Conv2d(nb_filter[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        )
        # self.v0 = nn.Conv2d(nb_filter[0], 1, 1, 1)
        # self.v1 = nn.Conv2d(nb_filter[0], 1, 1, 1)
        # self.v2 = nn.Conv2d(nb_filter[0], 1, 1, 1)
        # self.v3 = nn.Conv2d(nb_filter[0], 1, 1, 1)
        # self.v4 = nn.Conv2d(nb_filter[0], 1, 1, 1)
    def forward(self, input):
        # encoding path
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        tmp0_1 = self.Up2(x1_0)
        att0_1=self.Att0_1(g=tmp0_1,x=x0_0)
        x0_1 = torch.cat([att0_1,tmp0_1], 1)
        x0_1 = self.conv0_1(x0_1)

        output1 = self.final1(x0_1)

        x2_0 = self.conv2_0(self.pool(x1_0))
        # print('x2_0 shape:',x2_0.shape) #[B,128,64,64]
        x2_0_1 = self.head20_11(x2_0, output1)

        tmp1_1 = self.Up3(x2_0)
        att1_1=self.Att1_1(g=x2_0_1,x=x1_0) ####
        x1_1 = torch.cat([att1_1,tmp1_1,x2_0_1], 1)
        x1_1 = self.conv1_1(x1_1)

        x1_1_1 = self.head11_02(x1_1, output1)
        tmp0_2 = self.Up2(x1_1)
        att0_2=self.Att0_2(g=x1_1_1,x=x0_1)
        x0_2 = torch.cat([x0_0,att0_2,tmp0_2,x1_1_1], 1)
        x0_2 = self.conv0_2(x0_2)

        output2 = self.final2(x0_2)

        x3_0 = self.conv3_0(self.pool(x2_0))
        x3_0_1 = self.head30_21(x3_0,output2)
        tmp2_1 = self.Up4(x3_0)
        att2_1=self.Att2_1(g=x3_0_1,x=x2_0)
        x2_1 = torch.cat([att2_1,tmp2_1,x3_0_1], 1)
        x2_1 = self.conv2_1(x2_1)
        x2_1_1 = self.head21_12(x2_1,output2)

        tmp1_2 = self.Up3(x2_1)
        att1_2=self.Att1_2(g=x2_1_1,x=x1_1)
        x1_2 = torch.cat([x1_0,att1_2,tmp1_2,x2_1_1], 1)
        x1_2 = self.conv1_2(x1_2)
        x1_2_1 = self.head12_03(x1_2,output2)
        tmp0_3 = self.Up2(x1_2)
        att0_3=self.Att0_3(g=x1_2_1,x=x0_2)
        x0_3 = torch.cat([x0_0,x0_1,att0_3,tmp0_3,x1_2_1], 1)
        x0_3 = self.conv0_3(x0_3)

        output3 = self.final3(x0_3)

        x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0_1 = self.head40_31(x4_0,output3)
        tmp3_1=self.Up5(x4_0)
        att3_1=self.Att3_1(g=x4_0_1,x=x3_0)
        x3_1 = torch.cat([att3_1,tmp3_1,x4_0_1], 1)
        x3_1 = self.conv3_1(x3_1)
        x3_1_1 = self.head31_22(x3_1,output3)
        tmp2_2=self.Up4(x3_1)
        att2_2=self.Att2_2(g=x3_1_1,x=x2_1)
        x2_2 = torch.cat([x2_0,att2_2, tmp2_2, x3_1_1], 1)
        x2_2 = self.conv2_2(x2_2)
        x2_2_1 = self.head22_13(x2_2,output3)
        tmp1_3=self.Up3(x2_2)
        att1_3=self.Att1_3(g=x2_2_1,x=x1_2)
        x1_3 = torch.cat([x1_0 ,x1_1,att1_3, tmp1_3 ,x2_2_1], 1)
        x1_3 = self.conv1_3(x1_3)
        x1_3_1 = self.head13_04(x1_3,output3)
        tmp0_4=self.Up2(x1_3)
        att0_4 = self.Att0_4(g=x1_3_1,x=x0_3)
        x0_4 = torch.cat([x0_1, x0_2,att0_4, tmp0_4,x0_0, x1_3_1], 1)
        x0_4 = self.conv0_4(x0_4)

        output4 = self.final4(x0_4)

        # fv0= self.v0(x0_0)
        # fv1= self.v1(x0_1)
        # fv2= self.v2(x0_2)
        # fv3= self.v3(x0_3)
        # fv4= self.v4(x0_4)
        # # print('x0_4.shape',x0_4.shape)
        # imsave('feature visualization/x00.png', (fv0[0,0,:,:].cpu()))
        # imsave('feature visualization/x01.png', (fv1[0,0,:,:].cpu()))
        # imsave('feature visualization/x02.png', (fv2[0,0,:,:].cpu()))
        # imsave('feature visualization/x03.png', (fv3[0,0,:,:].cpu()))
        # imsave('feature visualization/x04.png', (fv4[0,0,:,:].cpu()))
        return [output1, output2, output3, output4]

class R2AttU_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1,t=2):
        super(R2AttU_Net,self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)
        # self.Upsample = nn.Upsample(scale_factor=2)
        n1 = 16#32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.RRCNN1 = RRCNN_block(in_ch=1,out_ch=filters[0],t=t)

        self.RRCNN2 = RRCNN_block(in_ch=filters[0],out_ch=filters[1],t=t)

        self.RRCNN3 = RRCNN_block(in_ch=filters[1],out_ch=filters[2],t=t)

        self.RRCNN4 = RRCNN_block(in_ch=filters[2],out_ch=filters[3],t=t)

        self.RRCNN5 = RRCNN_block(in_ch=filters[3],out_ch=filters[4],t=t)


        self.Up5 = up_conv2(in_ch=filters[4],out_ch=filters[3])
        self.Att5 = Attention_block_3d(F_g=filters[3],F_l=filters[3],F_int=filters[2])
        self.Up_RRCNN5 = RRCNN_block(in_ch=filters[4], out_ch=filters[3],t=t)

        self.Up4 = up_conv2(in_ch=filters[3],out_ch=filters[2])
        self.Att4 = Attention_block_3d(F_g=filters[2],F_l=filters[2],F_int=filters[1])
        self.Up_RRCNN4 = RRCNN_block(in_ch=filters[3], out_ch=filters[2],t=t)

        self.Up3 = up_conv2(in_ch=filters[2],out_ch=filters[1])
        self.Att3 = Attention_block_3d(F_g=filters[1],F_l=filters[1],F_int=filters[0])
        self.Up_RRCNN3 = RRCNN_block(in_ch=filters[2], out_ch=filters[1],t=t)

        self.Up2 = up_conv2(in_ch=filters[1],out_ch=filters[0])
        self.Att2 = Attention_block_3d(F_g=filters[0],F_l=filters[0],F_int=32)
        self.Up_RRCNN2 = RRCNN_block(in_ch=filters[1], out_ch=filters[0],t=t)

        self.Conv_1x1 = nn.Conv2d(filters[0],output_ch,kernel_size=1,stride=1,padding=0)

        self.final = nn.Sequential(
            nn.Conv3d(filters[0], 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2),
            # nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Sigmoid()
        ) # 卷积个数，输出通道数，kernel尺寸，stride，pad

    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        # d1 = self.Conv_1x1(d2)
        d1 = self.final(d2)#Conv_1x1(d2)#
        return d1

def plotNNFilterOverlay(units, figure_id, interp='bilinear',
                        colormap=plt.cm.jet, colormap_lim=None, alpha=0.8):
    # plt.ion()

    filters = units.shape[0]
    size=units.shape[3]
    fig = plt.figure(figure_id, figsize=(5,5))
    plt.clf() #重置画布
    index=0
    # for i in range(filters):
        # print("AttentionUNet3D!!")

        # print("AttentionUNet3D!00!")
    directory='/home/lc/Study/DataBase/LITS(1)/map/{}-{}/'.format(figure_id, size)

    index=len(glob(directory+'*'))
    # print(index)
    if not os.path.exists(directory):
        # print("AttentionUNet3D!22!")
        os.mkdir(directory)

    plt.imshow(units[0,0,0,:,:],interpolation=interp,  cmap=colormap) #, alpha=alpha
    # misc.imsave('/home/lc/学习/DataBase/CHAOS_Png_DataSets/att_images1/lena_new_sz.png', units[i,0,:,:])
    plt.axis('off')

    plt.savefig(directory+str(index)+'.png',dpi=200,bbox_inches='tight')    #
    # print("AttentionUNet3D111!!")
    # plt.colorbar()
    # plt.title(title, fontsize='small')
    if colormap_lim:
        plt.clim(colormap_lim[0],colormap_lim[1])
        # plt.close()
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.tight_layout()
    # plt.show()
