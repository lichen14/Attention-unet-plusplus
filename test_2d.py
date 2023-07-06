# -*- coding: utf-8 -*-

import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave
import matplotlib.pyplot as plt # plt 用于显示图片
import torch
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
from dataset.trans_dataset_2d import test_fix_ds
import xlsxwriter as xw
import new_archs
from metrics import dice_coef, batch_iou, mean_iou, iou_score,precision_and_recall_and_F1,Hausdorff_Distance
import losses
from utils import str2bool, count_params


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    test_args = parse_args()
    # print(test_args.name)
    args = joblib.load('./models/CHAOS/%s/args.pkl' %test_args.name)
    # args = joblib.load('models/None_NestedUNet_woDS/args.pkl')

    if not os.path.exists('./new_output/%s' %args.name):
        os.makedirs('./new_output/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    joblib.dump(args,'./models/CHAOS/%s/args.pkl' %args.name)
    # create model
    print("=> creating model %s" %args.arch)
    model = new_archs.__dict__[args.arch](args)

    model = model.cuda()

    # Data loading code
    # test_img_paths = glob('/home/lc/学习/DataBase/' + args.dataset + '/y_images/*')
    # test_mask_paths = glob('/home/lc/学习/DataBase/' + args.dataset + '/y_masks/*')

    # test_img_paths = glob('/home/lc/学习/DataBase/LITS(1)/new_train/Whole Test PNG_256/CT256/*')
    # test_mask_paths = glob('/home/lc/学习/DataBase/LITS(1)/new_train/Whole Test PNG_256-512/seg/*')
    test_img_paths = glob('/home/lc/Study/code-0524/dataset/dataset-liver/test/translated_image/*')
    # test_mask_paths = glob('/home/lc/学习/DataBase/Unet-master/deform/y_mask/*')
    # train_img_paths, test_img_paths, train_mask_paths, test_mask_paths = \
    #     train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)

    model=torch.nn.DataParallel(model,device_ids=[0])

    # state_dict = torch.load('models/%s/model.pth' %args.name)
    # # create new OrderedDict that does not contain `module.`
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:] # remove `module.`
    #     new_state_dict[name] = v
    # # load params
    # model.load_state_dict(new_state_dict)

    model.load_state_dict(torch.load('./models/CHAOS/%s/model.pth' %args.name))
    model.eval()

    # test_dataset = Dataset(args, test_img_paths, test_mask_paths)
    # test_dataset = Dataset(test_img_paths, test_mask_paths)
    test_loader = torch.utils.data.DataLoader(
        test_fix_ds,
        batch_size=1,#args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ious = []
        Dice = []
        precision = []
        recall = []
        F1score = []
        time_list = []
        Dist = []
        workbook = xw.Workbook('./new_output/%s/result.xlsx' %args.name)
        if not os.path.exists('./new_output/%s' %args.name):
            os.mkdir('./new_output/%s' %args.name)
        worksheet = workbook.add_worksheet('result')

        # 设置单元格格式
        bold = workbook.add_format()
        bold.set_bold()

        center = workbook.add_format()
        center.set_align('center')

        center_bold = workbook.add_format()
        center_bold.set_bold()
        center_bold.set_align('center')

        worksheet.set_column(1, len(test_img_paths), width=15)
        worksheet.set_column(0, 0, width=30, cell_format=center_bold)
        worksheet.set_row(0, 20, center_bold)

        # 写入文件名称
        worksheet.write(0, 0, 'file name')
        for index, file_name in enumerate(test_img_paths, start=1):
            worksheet.write( index, 0, file_name)
        # 写入各项评价指标名称
        worksheet.write(0, 1, 'organ:dice')
        worksheet.write(0, 2, 'organ:iou')
        worksheet.write(0, 3, 'organ:precision')
        worksheet.write(0, 4, 'organ:recall')
        worksheet.write(0, 5, 'organ:area-diameter')
        worksheet.write(0, 6, 'organ:speed')
        worksheet.write(0, 7, 'organ:fpr')
        worksheet.write(0, 8, 'organ:distance')
        casetime = 0
        with torch.no_grad():
            for i, (orgin, input, target) in tqdm(enumerate(test_loader), total=len(test_loader)):

                # compute output
                # print(args.deepsupervision)
                o =orgin[0,0,:,:]
                t=input[0,0,:,:]
                # print(t.shape)
                # tt=np.zeros((256,256,3))
                # tt[:,:,0]=t[0,:,:]
                # # tt[:,:,1]=t[0,:,:]
                # # tt[:,:,2]=t[1,:,:]
                # print(tt[0,100,100])
                # m=target[1,:,:,:]
                # mm=np.zeros((256,256))
                mm=target[0,:,:]
                ##print(np.maximum(mm, -1))
                # # mm[:,:,1]=target[1,:,:]
                # # mm[:,:,2]=target[2,:,:]
                # # t = t.transpose((1,2,0))
                # # print(tt.shape)


                if args.deepsupervision:
                    output = model(input)[-1]
                else:
                    output = model(input)

                input = input.cuda()
                target = target.cuda()
                start = time()
                casetime = time() - start
                output = output.squeeze(dim=1)
                # print(output.shape)
                output = output.cpu().numpy()
                # img_paths = test_img_paths[args.batch_size*i:args.batch_size*(i+1)]
                img_paths = test_img_paths[1*i:1*(i+1)]
                print(img_paths)
                # for j in range(output.shape[0]):
                #     # print(j)
                #     imsave('./output/%s/'%args.name+os.path.basename(img_paths[j]), (output[j,:,:]))
                imsave('./new_output/%s/'%args.name+os.path.basename(img_paths[0]), (output[0,:,:]))

                target = target.cpu().numpy()
                target = target.astype(np.float32)/255
                # print(target.shape,output.shape)

                # print(target[0,200,200],output[0,200,200])
                area = np.sum(target>0.5)
                pre,re,f1 = precision_and_recall_and_F1(target, output,False)
                distance = Hausdorff_Distance(target, output)

                if pre>0 and re >0 and f1>0:
                    precision.append(pre)
                    recall.append(re)
                    F1score.append(f1)
                # re = recall_score(mask, pb)

                iou = iou_score(output, target)
                ious.append(iou)
                dice = dice_coef(output, target)
                Dice.append(dice)
                Dist.append(distance)
                if i>0:
                    time_list.append(casetime)
                worksheet.write(i + 1, 1, dice)
                worksheet.write(i + 1, 2, iou)
                worksheet.write(i + 1, 3, pre)
                worksheet.write(i + 1, 5, math.sqrt(area/math.pi))
                worksheet.write(i + 1, 6, casetime)
                # worksheet.write(i + 1, 7, fpr)
                worksheet.write(i + 1, 8, distance)
                print(' dice: {:.3f}, iou: {:.3f}, precision: {:.3f}, recall: {:.3f},  F1-score: {:.3f}'.format( dice,iou,pre,re,f1))
                print('this case use {:.3f} s'.format(casetime))
                print('-----------------------')
                if dice>=0.9:
                    plt.subplot(221)
                    plt.imshow(o)
                    # plt.show()
                    plt.subplot(222)
                    plt.imshow(t)
                    plt.subplot(223)
                    plt.imshow(mm)#, cmap='Greys_r') # 显示图片
                    # plt.axis('off') # 不显示坐标轴
                    plt.subplot(224)
                    plt.imshow(output[0])
                    plt.show()
            workbook.close()
            # 输出整个测试集的平均dice系数和平均处理时间
            print('dice per case: {}'.format(sum(Dice) / len(Dice)))
            print('IoU per case: {}'.format(sum(ious) / len(ious)))
            print('precision per case: {}'.format(sum(precision) / len(precision)))
            print('recall per case: {}'.format(sum(recall) / len(recall)))
            print('F1-Score per case: {}'.format(sum(F1score) / len(F1score)))
            print('Dist per case: {}'.format(sum(Dist) / len(Dist)))
            print('time per case: {}'.format(sum(time_list) ))
        torch.cuda.empty_cache()

    # IoU

    # for i in tqdm(range(len(test_mask_paths))):
    #     mask = imread(test_mask_paths[i])
    #     pb = imread('output/LITS/%s/'%args.name+os.path.basename(test_mask_paths[i]))
    #     # print(mask[100,100])
    #     # mask = mask.astype('float32') / 255
    #     pb = pb.astype('float32') / 255

        '''
        plt.figure()
        plt.subplot(121)
        plt.imshow(mask)
        plt.subplot(122)
        plt.imshow(pb)
        plt.show()
        '''

        # plt.subplot(121)
        # plt.imshow(pb, cmap='Greys_r')
        # plt.subplot(122)
        # plt.imshow(mask, cmap='Greys_r')
        # plt.show()



        # acc = accuracy(pb, mask)
        # Acc.append(acc)


    # print('accuracy: %.4f' %np.mean(Acc))


if __name__ == '__main__':
    main()
