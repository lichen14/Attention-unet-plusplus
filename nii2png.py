import os
import shutil
from time import time
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from skimage.io import imread, imsave
from skimage import transform
# ct_dir = '/home/lc/学习/DataBase/LITS(1)/new_train/Whole Train_Nii/CT/'
# seg_dir = '/home/lc/学习/DataBase/LITS(1)/new_train/Whole Train_Nii/seg/'
# new_ct_dir = '/home/lc/学习/DataBase/LITS(1)/new_train/Whole Train_PNG/CT/'
# new_seg_dir = '/home/lc/学习/DataBase/LITS(1)/new_train/Whole Train_PNG/seg/'
ct_dir = '/home/lc/Study/Project/data/LITS/CT/valid_nii/'
seg_dir = '/home/lc/Study/Project/data/LITS/seg/valid_nii/'
new_ct_dir = '/home/lc/Study/Project/data/LITS/CT/valid_png_512_3/'
new_seg_dir = '/home/lc/Study/Project/data/LITS/seg/valid_png_512_3/'

upper = 1000
lower = -1000
down_scale = 1#0.5
slice_thickness =1
# test_ct_dir = '/home/lc/学习/DataBase/LITS(1)/png/test_image/'
# test_seg_dir = '/home/lc/学习/DataBase/LITS(1)/png/test_mask/'
# os.mkdir(new_ct_dir)
# os.mkdir(new_seg_dir)
# os.mkdir(test_ct_dir)
os.mkdir(new_ct_dir)
os.mkdir(new_seg_dir)
file_index = 0
test_index = 0
for ct_file in os.listdir(ct_dir):

    # 将CT和金标准入读内存
    
    ct = sitk.ReadImage(os.path.join(ct_dir, ct_file), sitk.sitkInt16)
    seg = sitk.ReadImage(os.path.join(seg_dir, ct_file.replace('volume', 'segmentation').replace('CT', 'seg')), sitk.sitkInt8)
    ct_name=ct_file.replace('.nii', '-')
    seg_name = ct_name.replace('volume', 'segmentation')#.replace('CT', 'seg')
    print(ct_name)
    # seg = ct.resize(256,256)

    ct_array = sitk.GetArrayFromImage(ct)
    ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / slice_thickness, down_scale, down_scale), order=3)
    
    # print(ct_array.shape)
    seg_array = sitk.GetArrayFromImage(seg)
    seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / slice_thickness, down_scale, down_scale), order=0)
    seg_array[seg_array !=1] = 0
    # seg_array[seg_array >2] = 0
    # seg_array[seg_array >0] = 1
    # print(seg_array.shape)
    for i in range(2,len(ct_array)):
        new_ct_array = ct_array[i, :, :]
        new_seg_array = seg_array[i, :, :]

        
        # label_array = label_array.transpose((2,0,1))
        # seg_array = seg_array.transpose((2,0,1))
        # print(seg_array.shape)
        # 将金标准中肝脏和肝肿瘤的标签融合为一个
        # 将灰度值在阈值之外的截断掉
        new_ct_array[new_ct_array > upper] = upper
        new_ct_array[new_ct_array < lower] = lower
        # new_ct = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / slice_thickness, down_scale, down_scale), order=3)#np.array(new_ct_array,dtype='uint8')
        # new_ct.SetOrigin(ct_array.GetOrigin())
        # new_ct.SetSpacing((ct_array.GetSpacing()[0] * int(1 / down_scale), ct_array.GetSpacing()[1] * int(1 / down_scale), slice_thickness))

        # if ct_name.find('98-')>0 or ct_name.find('53-')>0 or ct_name.find('54-')>0 or ct_name.find('55-')>0 or ct_name.find('56-')>0 or ct_name.find('57-')>0 or ct_name.find('58-')>0 or ct_name.find('59-')>0 or ct_name.find('60-')>0 or ct_name.find('61-')>0 or ct_name.find('62-')>0 or ct_name.find('63-')>0 or ct_name.find('64-')>0 or ct_name.find('65-')>0 or ct_name.find('66-')>0 or ct_name.find('67-')>0 or ct_name.find('83-')>0 or ct_name.find('84-')>0 or ct_name.find('85-')>0 or ct_name.find('86-')>0 or ct_name.find('87-')>0 or ct_name.find('88-')>0 or ct_name.find('89-')>0 or ct_name.find('90-')>0 or ct_name.find('91-')>0 or ct_name.find('92-')>0 or ct_name.find('93-')>0 or ct_name.find('94-')>0 or ct_name.find('95-')>0 or ct_name.find('96-')>0 or ct_name.find('97-')>0 or ct_name.find('99-')>0:
        #     print('come in')
        #     new_ct_array = np.flip(new_ct_array,axis=0)
        #     new_seg_array = np.flip(new_seg_array,axis=0)
        # new_ct_array = np.flip(new_ct_array,axis=0)
        # new_seg_array = np.flip(new_seg_array,axis=0)
        
        # new_seg = np.array(new_seg_array,dtype='uint8')
        # print(new_seg_array.shape)
        # new_seg.SetDirection(ct_array.GetDirection())
        # new_seg.SetOrigin(ct_array.GetOrigin())
        # new_seg.SetSpacing((ct_array.GetSpacing()[0]* int(1 / down_scale), ct_array.GetSpacing()[1] * int(1 / down_scale), slice_thickness))

        count0=sum(new_seg_array==1)    #剔除空白分割图

        count_s = sum(new_seg_array & seg_array[i-1, :, :]) #剔除相近的图
        # print(sum(count_s)) #1 4
        # if sum(count0)/(new_seg_array.shape[0]*new_seg_array.shape[1])>0.01 and sum(count_s)/(new_seg_array.shape[0]*new_seg_array.shape[1])>0.01:
        if sum(count_s)/(new_seg_array.shape[0]*new_seg_array.shape[1])>0.0001:

            # print(count0)
            file_index += 1
            new_ct_name =  ct_name+str(file_index) + '.png'
            new_seg_name = seg_name+str(file_index) + '.png'
            # new_seg_array = transform.resize(new_seg_array,(256,256))
            # print(new_ct_name)
            # print(new_seg_name)
            # test_index += 1
            # if test_index ==100:
            #     imsave(test_seg_dir+new_seg_name, new_seg_array)
            #     imsave(test_ct_dir+new_ct_name, new_ct_array)
            #     test_index=0
            # else:
            imsave(new_seg_dir+new_seg_name, new_seg_array)
            imsave(new_ct_dir+new_ct_name, new_ct_array)
    file_index = 0
        # cv2.imwrite(os.path.join(new_seg_dir, new_seg_name),new_seg)
        # plt.savefig(new_seg, os.path.join(new_seg_dir, new_seg_name))
        # sitk.WriteImage(new_ct, os.path.join(new_ct_dir, new_ct_name))
        # sitk.WriteImage(new_seg, os.path.join(new_seg_dir, new_seg_name))
    # print(ct_array.shape)



    # print(seg_array.shape)
