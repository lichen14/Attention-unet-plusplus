"""

获取固定取样方式下的训练数据
首先将灰度值超过upper和低于lower的灰度进行截断
然后调整slice thickness，然后将slice的分辨率调整为256*256
只有包含肝脏以及肝脏上下 expand_slice 张slice作为训练样本
最后将输入数据分块，以轴向 stride 张slice为步长进行取样

网络输入为256*256*size
当前脚本依然对金标准进行了缩小，如果要改变，直接修改第70行就行
"""

import os
import shutil
from time import time
from glob import glob
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage


upper = zz200
lower = -200
expand_slice = 20  # 轴向上向外扩张的slice数量
size = 48  # 取样的slice数量
stride = 3  # 取样的步长
down_scale = 0.25
slice_thickness = 2

ct_dir = '/home/lc/Study/DataBase/kits/Train/image/'#'/home/lc/Study/DataBase/LITS(1)/train/Whole Train/CT/'
seg_dir = '/home/lc/Study/DataBase/kits/Train/mask/'#'/home/lc/Study/DataBase/LITS(1)/train/Whole Train/seg/'

new_ct_dir = '/home/lc/Study/DataBase/kits/tumor_experiment/train/CT-kidney3/'
new_seg_dir = '/home/lc/Study/DataBase/kits/tumor_experiment/train/seg-kidney3/'

# if os.path.exists('/home/lc/学习/DataBase/LITS(1)/new_train/Training Batch 1/CT/'):
#     shutil.rmtree('/home/lc/学习/DataBase/LITS(1)/new_train/Training Batch 1/seg/')

# os.mkdir('/home/lc/学习/DataBase/LITS(1)/new_train/Training Batch 1/')
os.mkdir(new_ct_dir)
os.mkdir(new_seg_dir)
file_index = 0

# 用来统计最终剩下的slice数量
left_slice_list = []

start_time = time()
for ct_file in os.listdir(ct_dir):

    # 将CT和金标准入读内存
    ct = sitk.ReadImage(os.path.join(ct_dir, ct_file), sitk.sitkInt16)
    # ct = ct.resize((512,512))
    ct_array = sitk.GetArrayFromImage(ct)
    ct_array = ct_array.transpose((2,0,1))
    # print(ct_array.shape)     #(751, 512, 512)

    seg = sitk.ReadImage(os.path.join(seg_dir, ct_file.replace('image', 'mask')), sitk.sitkInt8)

    seg_array = sitk.GetArrayFromImage(seg)

    label_array = sitk.GetArrayFromImage(seg)


    label_array[label_array <1] = 0
    # seg_array[seg_array >2] = 0
    label_array[label_array >=1] = 1
    label_array = label_array.transpose((2,0,1))
    seg_array = seg_array.transpose((2,0,1))
    # print(seg_array.shape)
    # 将金标准中肝脏和肝肿瘤的标签融合为一个
    # seg_array[seg_array > 0] = 1
    seg_array[seg_array <2] = 0
    # seg_array[seg_array >2] = 0
    seg_array[seg_array >=2] = 1
    # 将灰度值在阈值之外的截断掉
    ct_array[ct_array > upper] = upper
    ct_array[ct_array < lower] = lower

    ct_array2=ct_array*label_array
    # 对CT和金标准进行插值，插值之后的array依然是int类型
    ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / slice_thickness, down_scale, down_scale), order=3)
    ct_array2 = ndimage.zoom(ct_array2, (ct.GetSpacing()[-1] / slice_thickness, down_scale, down_scale), order=3)
    seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / slice_thickness, down_scale, down_scale), order=0)
    # for i in range(size):
    #     t=ct_array[i,:,:]
    #     # print(t.shape)
    #     # tt=np.zeros((256,256,3))
    #     # tt[:,:,0]=t[0,:,:]
    #     # # tt[:,:,1]=t[0,:,:]
    #     # # tt[:,:,2]=t[1,:,:]
    #     # print(tt[0,100,100])
    #     # # m=target[1,:,:,:]
    #     # mm=np.zeros((512,512))
    #     mm=ct_array2[i,:,:]
    #     # # mm[:,:,1]=target[1,:,:]
    #     # # mm[:,:,2]=target[2,:,:]
    #     # # t = t.transpose((1,2,0))
    #     # # print(tt.shape)
    #     plt.subplot(121)
    #     plt.imshow(t) # 显示图片
    #     plt.subplot(122)
    #     plt.imshow(mm)#, cmap='Greys_r') # 显示图片
    #     # plt.axis('off') # 不显示坐标轴
    #     plt.show()

    print(ct_file)
    # print(seg_array.shape)
    # 找到肝脏区域开始和结束的slice，并各向外扩张
    z = np.any(seg_array, axis=(1, 2))
    start_slice, end_slice = np.where(z)[0][[0, -1]]

    # 两个方向上各扩张个slice
    if start_slice - expand_slice < 0:
        start_slice = 0
    else:
        start_slice -= expand_slice

    if end_slice + expand_slice >= seg_array.shape[0]:
        end_slice = seg_array.shape[0] - 1
    else:
        end_slice += expand_slice

    # 如果这时候剩下的slice数量不足size，直接放弃，这样的数据很少
    if end_slice - start_slice + 1 < size:
        print('!!!!!!!!!!!!!!!!')
        print(ct_file, 'too little slice')
        print('!!!!!!!!!!!!!!!!')
        continue

    ct_array = ct_array[start_slice:end_slice + 1, :, :]
    ct_array2 = ct_array2[start_slice:end_slice + 1, :, :]
    seg_array = seg_array[start_slice:end_slice + 1, :, :]
    # print(ct_array.shape)
    # print(seg_array.shape)
    print('{} have {} slice left'.format(ct_file, ct_array2.shape[0]))
    left_slice_list.append(ct_array2.shape[0])

    # 在轴向上按照一定的步长进行切块取样，并将结果保存为nii数据
    start_slice = 0
    end_slice = start_slice + size - 1

    while end_slice <= ct_array2.shape[0] - 1:

        old_ct_array = ct_array[start_slice:end_slice + 1, :, :]
        new_ct_array = ct_array2[start_slice:end_slice + 1, :, :]
        new_seg_array = seg_array[start_slice:end_slice + 1, :, :]

        old_ct = sitk.GetImageFromArray(old_ct_array)
        old_ct.SetDirection(ct.GetDirection())
        old_ct.SetOrigin(ct.GetOrigin())
        old_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / down_scale), ct.GetSpacing()[1] * int(1 / down_scale), slice_thickness))

        new_ct = sitk.GetImageFromArray(new_ct_array)
        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / down_scale), ct.GetSpacing()[1] * int(1 / down_scale), slice_thickness))


        new_seg = sitk.GetImageFromArray(new_seg_array)
        new_seg.SetDirection(ct.GetDirection())
        new_seg.SetOrigin(ct.GetOrigin())
        new_seg.SetSpacing((ct.GetSpacing()[0] * int(1 / down_scale), ct.GetSpacing()[1] * int(1 / down_scale), slice_thickness))

        old_ct_name = 'old_image' + str(file_index) + '.nii'
        new_ct_name = 'image' + str(file_index) + '.nii'
        new_seg_name = 'mask' + str(file_index) + '.nii'

        sitk.WriteImage(old_ct, os.path.join(new_ct_dir, old_ct_name))
        sitk.WriteImage(new_ct, os.path.join(new_ct_dir, new_ct_name))
        sitk.WriteImage(new_seg, os.path.join(new_seg_dir, new_seg_name))

        file_index += 1

        start_slice += stride
        end_slice = start_slice + size - 1

    # 当无法整除的时候反向取最后一个block
    if end_slice is not ct_array2.shape[0] - 1:
        old_ct_array = ct_array[-size:, :, :]
        new_ct_array = ct_array2[-size:, :, :]
        new_seg_array = seg_array[-size:, :, :]
        print(new_ct_array.shape,new_seg_array.shape)

        old_ct = sitk.GetImageFromArray(old_ct_array)
        old_ct.SetDirection(ct.GetDirection())
        old_ct.SetOrigin(ct.GetOrigin())
        old_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / down_scale), ct.GetSpacing()[1] * int(1 / down_scale), slice_thickness))


        new_ct = sitk.GetImageFromArray(new_ct_array)

        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / down_scale), ct.GetSpacing()[1] * int(1 / down_scale), slice_thickness))

        new_seg = sitk.GetImageFromArray(new_seg_array)

        new_seg.SetDirection(ct.GetDirection())
        new_seg.SetOrigin(ct.GetOrigin())
        new_seg.SetSpacing((ct.GetSpacing()[0]* int(1 / down_scale), ct.GetSpacing()[1]* int(1 / down_scale), slice_thickness))

        old_ct_name = 'old_image' + str(file_index) + '.nii'
        new_ct_name = 'image' + str(file_index) + '.nii'
        new_seg_name = 'mask' + str(file_index) + '.nii'

        sitk.WriteImage(old_ct, os.path.join(new_ct_dir, old_ct_name))
        sitk.WriteImage(new_ct, os.path.join(new_ct_dir, new_ct_name))
        sitk.WriteImage(new_seg, os.path.join(new_seg_dir, new_seg_name))

        file_index += 1

    # 每处理完一个数据，打印一次已经使用的时间
    print('already use {:.3f} min'.format((time() - start_time) / 60))
    print('-----------')


left_slice_list = np.array(left_slice_list)

plt.hist(left_slice_list, 200, rwidth=1)
plt.show()
