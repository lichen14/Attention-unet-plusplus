"""

固定取样方式下的数据集
"""

import shutil, os
from glob import glob
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset as dataset
from sklearn.model_selection import train_test_split
on_server = True


class Dataset(dataset):
    def __init__(self, ct_dir, seg_dir):

        self.ct_list = ct_dir
        # self.seg_list = list(map(lambda x: x.replace('image', 'mask'), self.ct_list)) #kits
        self.seg_list = list(map(lambda x: x.replace('image', 'mask'), self.ct_list))  #lits tumor
        # self.seg_list = list(map(lambda x: x.replace('CT', 'seg'), self.seg_list))
        # print(self.ct_list)
        # print(sdelf.seg_list)
        # self.ct_list = list(map(lambda x: os.path.join(ct_dir, x), self.ct_list))
        # self.seg_list = list(map(lambda x: os.path.join(seg_dir, x), self.seg_list))

    def __getitem__(self, index):

        ct_path = self.ct_list[index]
        seg_path = self.seg_list[index]
        print(ct_path,seg_path)
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)
        # print(ct_array.shape)
        # ct_array = ct_array.transpose((2,0,1))
        # print(seg_array.shape)
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array)

        return ct_array, seg_array

    def __len__(self):

        return len(self.ct_list)

# # kits
# ct_dir = glob('/home/lc/Study/DataBase/kits/Train_Nii256/image/*')
# seg_dir = glob('/home/lc/Study/DataBase/kits/Train_Nii256/mask/*' )
# segTHOR
ct_dir = glob('/home/lc/Study/DataBase/kits/tumor_experiment/train/image2/*')
seg_dir = glob('/home/lc/Study/DataBase/kits/tumor_experiment/train/mask2/*' )
#lits
# ct_dir = glob('/home/lc/Study/DataBase/LITS(1)/new_train/Whole Train Nii_256/CT/*')
#     # if on_server is False else './train/fix/ct/'
# seg_dir = glob('/home/lc/Study/DataBase/LITS(1)/new_train/Whole Train Nii_256/seg/*' )
    # if on_server is False else './train/fix/seg/'
# other_img_paths, test_img_paths, other_mask_paths, test_mask_paths = \
#         train_test_split(ct_dir, seg_dir, test_size=100, random_state=1)
train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        train_test_split(ct_dir, seg_dir, test_size=0.2, random_state=42)
# test_img_paths = glob('/home/lc/Study/DataBase/LITS(1)/new_train/Whole Test Nii_256/CT/*')
test_img_paths = glob('/home/lc/Study/DataBase/kits/tumor_experiment/test/image2/*')
# test_mask_paths = glob('/home/lc/Study/DataBase/LITS(1)/new_train/Whole Test Nii_256/seg/*')
test_mask_paths = glob('/home/lc/Study/DataBase/kits/tumor_experiment/test/mask2/*')
# os.mkdir('/home/lc/学习/DataBase/LITS(1)/new_train/Whole Test Nii_256-512/')
# os.mkdir('/home/lc/学习/DataBase/LITS(1)/new_train/Whole Test Nii_256-512/ct/')
# os.mkdir('/home/lc/学习/DataBase/LITS(1)/new_train/Whole Test Nii_256-512/seg/')
# for i in range(len(test_mask_paths)):
#     test_img_paths = test_mask_paths[i].replace('segmentation', 'volume')
#     test_img_paths = test_img_paths.replace('seg', 'CT')
#     shutil.move(test_mask_paths[i], '/home/lc/学习/DataBase/LITS(1)/new_train/Whole Test Nii_256-512/seg/')
#
#     shutil.move(test_img_paths, '/home/lc/学习/DataBase/LITS(1)/new_train/Whole Test Nii_256-512/ct/')



# shutil.move(test_mask_paths, '/home/lc/学习/DataBase/LITS(1)/new_train/Whole Test Nii_256-512/seg/')
train_fix_ds = Dataset(train_img_paths, train_mask_paths)
valid_fix_ds = Dataset(val_img_paths, val_mask_paths)
test_fix_ds = Dataset(test_img_paths, test_mask_paths)
# # 测试代码
# from torch.utils.data import DataLoader
#
# train_dl = DataLoader(train_fix_ds, 12, True, num_workers=2, pin_memory=True)
# for index, (ct, seg) in enumerate(train_dl):
#     # print(type(ct), type(seg))
#     print(index, ct.size(), seg.size())
print('----------------done')
