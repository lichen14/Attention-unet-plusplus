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
from dataset.dataset_2d import test_fix_ds
import xlsxwriter as xw
import new_archs
from metrics import dice_coef, batch_iou, mean_iou, iou_score,precision_and_recall
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
    args = joblib.load('/home/lc/学习/From Github/MICCAI-LITS2017/models/LITS/%s/args.pkl' %test_args.name)
    # args = joblib.load('models/None_NestedUNet_woDS/args.pkl')

    if not os.path.exists('./output/%s' %args.name):
        os.makedirs('./output/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    joblib.dump(args,'/home/lc/学习/From Github/MICCAI-LITS2017/models/LITS/%s/args.pkl' %args.name)
    # create model
    print("=> creating model %s" %args.arch)
    model = new_archs.__dict__[args.arch](args)

    model = model.cuda()

    # Data loading code
    # test_img_paths = glob('/home/lc/学习/DataBase/' + args.dataset + '/y_images/*')
    # test_mask_paths = glob('/home/lc/学习/DataBase/' + args.dataset + '/y_masks/*')

    # test_img_paths = glob('/home/lc/学习/DataBase/LITS(1)/new_train/Whole Test PNG_256-512/CT/*')
    # test_mask_paths = glob('/home/lc/学习/DataBase/LITS(1)/new_train/Whole Test PNG_256-512/seg/*')
    test_img_paths = glob('/home/lc/学习/DataBase/LITS(1)/new_train/Whole Test PNG_256/CT256/*')
    # test_mask_paths = glob('/home/lc/学习/DataBase/Unet-master/deform/y_mask/*')
    # train_img_paths, test_img_paths, train_mask_paths, test_mask_paths = \
    #     train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)

    model=torch.nn.DataParallel(model,device_ids=[0,1,2,3])

    # state_dict = torch.load('models/%s/model.pth' %args.name)
    # # create new OrderedDict that does not contain `module.`
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:] # remove `module.`
    #     new_state_dict[name] = v
    # # load params
    # model.load_state_dict(new_state_dict)

    model.load_state_dict(torch.load('/home/lc/学习/From Github/MICCAI-LITS2017/models/LITS/%s/model.pth' %args.name))
    model.eval()

    # test_dataset = Dataset(args, test_img_paths, test_mask_paths)
    # test_dataset = Dataset(test_img_paths, test_mask_paths)
    test_loader = torch.utils.data.DataLoader(
        test_fix_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ious = []
        Dice = []
        precision = []
        recall = []
        time_list = []
        workbook = xw.Workbook('./output/%s/result.xlsx' %args.name)
        if not os.path.exists('./output/%s' %args.name):
            os.mkdir('./output/%s' %args.name)
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
        worksheet.write(0, 1, 'spleen:dice')
        worksheet.write(0, 2, 'spleen:iou')
        worksheet.write(0, 3, 'spleen:precision')
        worksheet.write(0, 4, 'spleen:recall')
        worksheet.write(0, 5, 'speed')
        worksheet.write(0, 6, 'shape')
        casetime = 0
        with torch.no_grad():
            for i, (input, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
                input = input.cuda()
                target = target.cuda()
                start = time()
                # compute output
                # print(args.deepsupervision)
                if args.deepsupervision:
                    output = model(input)[-1]
                else:
                    output = model(input)#[-1]
                casetime = time() - start+ casetime
                output = output.squeeze(dim=1)
                # print(output.shape)
                output = output.cpu().numpy()
                img_paths = test_img_paths[args.batch_size*i:args.batch_size*(i+1)]

                for j in range(output.shape[0]):
                    # print(j)
                    imsave('./output/%s/'%args.name+os.path.basename(img_paths[j]), (output[j,:,:]))

                target = target.cpu().numpy()
                target = target.astype(np.float32)/255
                # print(target.shape,output.shape)

                # print(target[0,200,200],output[0,200,200])

                pre,re = precision_and_recall(target, output,False)
                if pre>0 and re >0:
                    precision.append(pre)
                    recall.append(re)
                # re = recall_score(mask, pb)

                iou = iou_score(output, target)
                ious.append(iou)
                dice = dice_coef(output, target)
                Dice.append(dice)
                time_list.append(casetime)
                worksheet.write(i + 1, 1, dice)
                worksheet.write(i + 1, 2, iou)
                worksheet.write(i + 1, 3, pre)
                worksheet.write(i + 1, 4, re)
                worksheet.write(i + 1, 5, casetime)
                print(' dice: {:.3f}, iou: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format( dice,iou,pre,re))
                print('this case use {:.3f} s'.format(casetime))
                print('-----------------------')

            # 输出整个测试集的平均dice系数和平均处理时间
            print('dice per case: {}'.format(sum(Dice) / len(Dice)))
            print('IoU per case: {}'.format(sum(ious) / len(ious)))
            print('precision per case: {}'.format(sum(precision) / len(precision)))
            print('recall per case: {}'.format(sum(recall) / len(recall)))
            print('time per case: {}'.format(sum(time_list) / len(time_list)))
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
