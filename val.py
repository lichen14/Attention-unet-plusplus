"""

肝脏分割在自己的测试集下的脚本
"""

import os
from time import time

import torch
import torch.nn.functional as F
from sklearn.externals import joblib
import numpy as np
import xlsxwriter as xw
import SimpleITK as sitk
import scipy.ndimage as ndimage
import skimage.morphology as sm
import skimage.measure as measure
import argparse
import new_archs
#from new_archs import VNet
import losses
from utils import str2bool, count_params
from metrics import dice_coef, batch_iou, mean_iou, iou_score,precision_and_recall

loss_names = list(losses.__dict__.keys())

on_server = False

os.environ['CUDA_VISIBLE_DEVICES'] = '2,1,0'
# USE_CUDA = torch.cuda.is_available()
# device = torch.device("cuda:3,1,2,0" if USE_CUDA else "cpu")
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args

arch_names = list(new_archs.__dict__.keys())
test_args = parse_args()

val_ct_dir = '/mnt/zcy/val_data/CT/' if on_server is True else \
    '/home/lc/学习/DataBase/LITS(1)/new_train/Whole Test Nii_256-512/CT'

val_seg_dir = '/mnt/zcy/val_data/GT/' if on_server is True else \
    '/home/lc/学习/DataBase/LITS(1)/new_train/Whole Test Nii_256-512/seg'

liver_pred_dir = '/mnt/zcy/val_data/liver_pred/' if on_server is True else \
    '/home/lc/学习/From Github/MICCAI-LITS2017/output/%s' %test_args.name
if not os.path.exists(liver_pred_dir):
    os.mkdir(liver_pred_dir)

module_dir = './module/net1060-0.032-0.015.pth' if on_server is True else '/home/lc/学习/From Github/MICCAI-LITS2017/models/LITS/%s/model.pth' %test_args.name
args = joblib.load('/home/lc/学习/From Github/MICCAI-LITS2017/models/LITS/%s/args.pkl' %test_args.name)
print('Config -----')
for arg in vars(args):
    print('%s: %s' %(arg, getattr(args, arg)))
print('------------')
upper = 200
lower = -200
down_scale = 0.5
size = 32
slice_thickness = 3
threshold = 0.7

dice_list = []
iou_list =[]
precision_list =[]
recall_list=[]
time_list = []
def main():
    # 创建一个表格对象，并添加一个sheet，后期配合window的excel来出图
    workbook = xw.Workbook('/home/lc/学习/From Github/MICCAI-LITS2017/output/%s/result.xlsx' %args.name)
    if not os.path.exists('/home/lc/学习/From Github/MICCAI-LITS2017/output/%s' %args.name):
        os.mkdir('/home/lc/学习/From Github/MICCAI-LITS2017/output/%s' %args.name)
    worksheet = workbook.add_worksheet('result')

    # 设置单元格格式
    bold = workbook.add_format()
    bold.set_bold()

    center = workbook.add_format()
    center.set_align('center')

    center_bold = workbook.add_format()
    center_bold.set_bold()
    center_bold.set_align('center')

    worksheet.set_column(1, len(os.listdir(val_ct_dir)), width=15)
    worksheet.set_column(0, 0, width=30, cell_format=center_bold)
    worksheet.set_row(0, 20, center_bold)

    # 写入文件名称
    worksheet.write(0, 0, 'file name')
    for index, file_name in enumerate(os.listdir(val_ct_dir), start=1):
        worksheet.write( index, 0, file_name)

    # 写入各项评价指标名称
    worksheet.write(0, 1, 'liver:dice')
    worksheet.write(0, 2, 'liver:iou')
    worksheet.write(0, 3, 'liver:precision')
    worksheet.write(0, 4, 'liver:recall')
    worksheet.write(0, 5, 'speed')
    worksheet.write(0, 6, 'shape')

    # 定义网络并加载参数
    net = new_archs.__dict__[args.arch](args).cuda()
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load(module_dir))
    net.eval()


    for file_index, file in enumerate(os.listdir(val_ct_dir)):

        start = time()
        # print(file)
        # 将CT读入内存
        ct = sitk.ReadImage(os.path.join(val_ct_dir, file), sitk.sitkInt16)

        ct_array_list = sitk.GetArrayFromImage(ct)
        # print(ct_array_list.shape)
        origin_shape = ct_array_list.shape
        worksheet.write( file_index + 1, 6, str(origin_shape))

        outputs_list = []

        ct_tensor = torch.FloatTensor(ct_array_list).cuda()
        ct_tensor = ct_tensor.unsqueeze(dim=0)
        ct_tensor = ct_tensor.unsqueeze(dim=0)    #3D
        print(ct_tensor.shape)
        pred_seg = net(ct_tensor)
        casetime = time() - start
        # 将金标准读入内存来计算dice系数
        seg = sitk.ReadImage(os.path.join(val_seg_dir, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
        seg_array = sitk.GetArrayFromImage(seg)
        seg_array[seg_array > 0] = 1
        # print(seg_array.shape)


        pred_seg = pred_seg.squeeze(dim=0).squeeze(dim=0).cpu().detach().numpy()
        # print(pred_seg.shape)
        # pred_seg = F.upsample(pred_seg_tensor, seg_array.shape, mode='trilinear').squeeze().detach().numpy()
        pred_seg = (pred_seg > 0.5).astype(np.int16)

        # # 先进行腐蚀
        # pred_seg = sm.binary_erosion(pred_seg, sm.ball(5))

        # 取三维最大连通域，移除小区域
        pred_seg = measure.label(pred_seg, 4)
        props = measure.regionprops(pred_seg)
        # print(seg_array.shape)
        max_area = 0
        max_index = 0
        for index, prop in enumerate(props, start=1):
            if prop.area > max_area:
                max_area = prop.area
                max_index = index

        pred_seg[pred_seg != max_index] = 0
        pred_seg[pred_seg == max_index] = 1

        pred_seg = pred_seg.astype(np.uint8)

        # # 进行膨胀恢复之前的大小
        # pred_seg = sm.binary_dilation(pred_seg, sm.ball(5))
        # pred_seg = pred_seg.astype(np.uint8)

        # print('size of pred: ', pred_seg.shape)
        # print('size of GT: ', seg_array.shape)

        dice = dice_coef(pred_seg ,seg_array)
        dice_list.append(dice)
        iou = iou_score(pred_seg ,seg_array)
        iou_list.append(iou)
        pre,re = precision_and_recall(seg_array, pred_seg,False)
        if pre>0 and re >0:
            precision_list.append(pre)
            recall_list.append(re)

        worksheet.write(file_index + 1, 1, dice)
        worksheet.write(file_index + 1, 2, iou)
        worksheet.write(file_index + 1, 3, pre)
        worksheet.write(file_index + 1, 4, re)
        print('file: {}, dice: {:.3f}, iou: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(file, dice,iou,pre,re))

        # 将预测的结果保存为nii数据
        pred_seg = sitk.GetImageFromArray(pred_seg)

        pred_seg.SetDirection(ct.GetDirection())
        pred_seg.SetOrigin(ct.GetOrigin())
        pred_seg.SetSpacing(ct.GetSpacing())

        sitk.WriteImage(pred_seg, os.path.join(liver_pred_dir, file.replace('volume', 'pred')))
        del pred_seg


        time_list.append(casetime)

        worksheet.write(file_index + 1, 5,  casetime)

        print('this case use {:.3f} s'.format(casetime))
        print('-----------------------')


    # 输出整个测试集的平均dice系数和平均处理时间
    print('dice per case: {}'.format(sum(dice_list) / len(dice_list)))
    print('IoU per case: {}'.format(sum(iou_list) / len(iou_list)))
    print('precision per case: {}'.format(sum(precision_list) / len(precision_list)))
    print('recall per case: {}'.format(sum(recall_list) / len(recall_list)))
    print('time per case: {}'.format(sum(time_list) / len(time_list)))

    # 最后安全关闭表格
    workbook.close()

if __name__ == '__main__':
    main()
