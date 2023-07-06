import numpy as np

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # plt 用于显示图片
from hausdorff import hausdorff_distance

def Hausdorff_Distance(output, target):
    # two random 2D arrays (second dimension must match)
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    dist = []
    print(target.shape,output.shape)
    # Test computation of Hausdorff distance with different base distances
    for p in range(output.shape[0]):
        target_ = target[p,:,:]
        output_ = output[p,:,:]
        # print("Hausdorff distance test: {0}".format( hausdorff_distance(target_, output_, distance="manhattan") ))
        # print("Hausdorff distance test: {0}".format( hausdorff_distance(target_, output_, distance="euclidean") ))
        # print("Hausdorff distance test: {0}".format( hausdorff_distance(target_, output_, distance="chebyshev") ))
        # print("Hausdorff distance test: {0}".format( hausdorff_distance(target_, output_, distance="cosine") ))
        dist.append(hausdorff_distance(target_, output_, distance="manhattan") )
    return sum(dist) / len(dist)
    # For haversine, use 2D lat, lng coordinates
    # def rand_lat_lng(N):
    #     lats = np.random.uniform(-90, 90, N)
    #     lngs = np.random.uniform(-180, 180, N)
    #     return np.stack([lats, lngs], axis=-1)
    #
    # X = rand_lat_lng(100)
    # Y = rand_lat_lng(250)
    # print("Hausdorff haversine test: {0}".format( hausdorff_distance(X, Y, distance="haversine") ))

def mean_iou(y_true_in, y_pred_in, print_table=False):
    if True: #not np.sum(y_true_in.flatten()) == 0:
        labels = y_true_in
        y_pred = y_pred_in

        true_objects = 2
        pred_objects = 2

        intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

        # Compute areas (needed for finding the union between all objects)
        area_true = np.histogram(labels, bins = true_objects)[0]
        area_pred = np.histogram(y_pred, bins = pred_objects)[0]
        area_true = np.expand_dims(area_true, -1)
        area_pred = np.expand_dims(area_pred, 0)

        # Compute union
        union = area_true + area_pred - intersection

        # Exclude background from the analysis
        intersection = intersection[1:,1:]
        union = union[1:,1:]
        union[union == 0] = 1e-9

        # Compute the intersection over union
        iou = intersection / union

        # Precision helper function
        def precision_at(threshold, iou):
            matches = iou > threshold
            true_positives = np.sum(matches, axis=1) == 1   # Correct objects
            false_positives = np.sum(matches, axis=0) == 0  # Missed objects
            false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
            tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
            return tp, fp, fn

        # Loop over IoU thresholds
        prec = []
        if print_table:
            print("Thresh\tTP\tFP\tFN\tPrec.")
        for t in np.arange(0.5, 1.0, 0.05):
            tp, fp, fn = precision_at(t, iou)
            if (tp + fp + fn) > 0:
                p = tp / (tp + fp + fn)
            else:
                p = 0
            if print_table:
                print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
            prec.append(p)

        if print_table:
            print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
        return np.mean(prec)

    else:
        if np.sum(y_pred_in.flatten()) == 0:
            return 1
        else:
            return 0

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

def batch_iou(output, target):
    output = torch.sigmoid(output).data.cpu().numpy() > 0.5
    target = (target.data.cpu().numpy() > 0.5).astype('int')
    output = output[:,0,:,:]
    target = target[:,0,:,:]

    ious = []
    for i in range(output.shape[0]):
        ious.append(mean_iou(output[i], target[i]))

    return np.mean(ious)


def mean_iou(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).data.cpu().numpy()
    target = target.data.cpu().numpy()
    ious = []
    for t in np.arange(0.5, 1.0, 0.05):
        output_ = output > t
        target_ = target > t
        intersection = (output_ & target_).sum()
        union = (output_ | target_).sum()
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)

    return np.mean(ious)

def compute_iou(img1, img2):

    img1 = np.array(img1.data.cpu().numpy())
    img2 = np.array(img2.data.cpu().numpy())

    if img1.shape[0] != img2.shape[0]:
        raise ValueError("Shape mismatch: the number of images mismatch.")
    IoU = np.zeros( (img1.shape[0],), dtype=np.float32)
    for i in range(img1.shape[0]):
        im1 = np.squeeze(img1[i]>0.5)
        im2 = np.squeeze(img2[i]>0.5)

        if im1.shape != im2.shape:
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

        # Compute Dice coefficient
        intersection = np.logical_and(im1, im2)

        if im1.sum() + im2.sum() == 0:
            IoU[i] = 100
        else:
            IoU[i] = 2. * intersection.sum() * 100.0 / (im1.sum() + im2.sum())
        #database.display_image_mask_pairs(im1, im2)

    return IoU

def compute_dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    # im1=im1.data.cpu().numpy()
    # im2= im2.data.cpu().numpy()
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    # print(intersection.sum())
    return 1- 2. * intersection.sum() / im_sum

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    # print(output.shape)
    # target = target /255
    # output = output* 255
    iou_s = []
    # for index_b in range(target.shape[0]): #not np.sum(y_true_in.flatten()) == 0:
        # for index_c in range(target.shape[1]):
        #     true_positives = 0   # Correct objects
        #     false_positives = 0  # Missed objects
        #     false_negatives = 0
        #     y_true_in = target[index_b,index_c,:]
        #     y_pred_in = output[index_b,index_c,:]
        #
        #     # t=output_[0,10,:,:]
        #     # print(y_true_in.shape)
        #
        #
        #     # print(y_true_in.shape)
        #     # print(y_pred_in.shape)
        #     # Compute the intersection over union
        #     # iou = intersection / union
        #     for i in range(target.shape[2]):
        #         for j in range(target.shape[3]):
        #             # print(y_true_in[i,j],y_pred_in[i,j])
        #             if y_true_in[i,j]>0 and y_pred_in[i,j]>=0.5:
        #                 true_positives +=1
        #             if y_true_in[i,j]==0 and y_pred_in[i,j]>=0.5:
        #                 false_positives +=1
        #             if y_true_in[i,j]>0 and y_pred_in[i,j]<0.5:
        #                 false_negatives +=1
        #
        #
        #     # print("Thresh\tTP\tFP\tFN\tPrec.")
        #     # for t in np.arange(0.5, 1.0, 0.05):
        #         # true_positives, false_positives, false_negatives = precision_at(t, iou)
        #     if (true_positives + false_positives+ false_negatives) > 0:
        #         ious = true_positives / (true_positives + false_positives+ false_negatives )
        #     else:
        #         ious = 0
        #     print("ious: ",ious)
        #     iou_s.append(ious)
    # print(t.shape)
    # tt=np.zeros((512,512))
    # tt[:,:]=target[10,:,:]
    # print("target_:",tt[200,200])
    # # # m=target[1,:,:,:]
    # mm=np.zeros((512,512))
    # mm[:,:]=output[10,:,:]
    # # # mm[:,:,1]=target[1,:,:]
    # # # mm[:,:,2]=target[2,:,:]
    # # # t = t.transpose((1,2,0))
    # print("output_:",mm[200,200])
    # plt.subplot(121)
    # plt.imshow(tt) # 显示图片
    # plt.subplot(122)
    # plt.imshow(mm) # 显示图片
    # plt.axis('off') # 不显示坐标轴
    # plt.show()
    #
    # print("target:",target[10,200,200])
    # # # print(output_.shape)
    # print("output:",output[10,200,200])
    # print(target_.shape)

    target_ = target > 0
    output_ = output > 0.5

    # print(target.shape)
    #print(target[0,150,150])
    # print(output_.shape)
    #print(output[0,150,150])
    # print(target_.shape)

    # t=output_[10,:,:]
    # # print(t.shape)
    # tt=np.zeros((512,512))
    # tt[:,:]=target_[10,:,:]
    # print("target_:",tt[200,200])
    # # # m=target[1,:,:,:]
    # mm=np.zeros((512,512))
    # mm[:,:]=output_[10,:,:]
    # # # mm[:,:,1]=target[1,:,:]
    # # # mm[:,:,2]=target[2,:,:]
    # # # t = t.transpose((1,2,0))
    # print("output_:",mm[200,200])
    # plt.subplot(121)
    # plt.imshow(tt) # 显示图片
    # plt.subplot(122)
    # plt.imshow(mm) # 显示图片
    # plt.axis('off') # 不显示坐标轴
    # plt.show()

    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    # print(intersection,union)
    return (intersection + smooth) / (union + smooth) #np.mean(iou_s)#

def dice_coef(output, target):
    smooth = 1e-5
    # print(output_[10,200,200])
    # # print(target.shape)
    # print(target_[10,200,200])
    # output= torch.from_numpy(output)
    # print(target)
    # output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    # target = target.view(-1).data.cpu().numpy()
    # target = target.astype('float32') * 255
    # output = output.astype('float32') * 255
    # if torch.is_tensor(output):
    #     output = torch.sigmoid(output).data.cpu().numpy()
    # if torch.is_tensor(target):
    #     target = target.data.cpu().numpy()

    #
    output_ = output > 0.5
    target_ = target > 0
    #
    # t=output[0,10,:,:].data.cpu().numpy()
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
    # plt.imshow(mm, cmap='Greys_r') # 显示图片
    # # plt.axis('off') # 不显示坐标轴
    # plt.show()

    intersection = (output_ * target_).sum()
    # intersection = torch.sum((output_ + target_)==2)#(output_ * target_).sum()
    # print(intersection)
    # print(output.sum())
    # print(target.sum())
    return (2. * intersection + smooth) / \
        (output_.sum() + target_.sum() + smooth)
    #return 1-((2. * intersection + smooth) / \
    #    (torch.sum(output_) + torch.sum(target_) + smooth))

def accuracy(output, target):
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    output = (np.round(output)).astype('int')
    target = target.view(-1).data.cpu().numpy()
    target = (np.round(target)).astype('int')
    (output == target).sum()

    return (output == target).sum() / len(output)

def precision_and_recall_and_F1(y_true_in1, y_pred_in1, print_table=False):
    # y_true_in = y_true_in.astype('float32') * 255
    # y_pred_in = y_pred_in.astype('float32') * 255
    if True: #not np.sum(y_true_in.flatten()) == 0:

        # print(y_true_in.shape)
        # print(y_pred_in.shape)
        # Compute the intersection over union
        # iou = intersection / union
        prec = []
        reca = []
        y_pred_in = y_pred_in1 > 0.5
        y_true_in = y_true_in1 > 0
        true_positives = np.sum((y_true_in * y_pred_in)==1)

        y_true_in1 = ((y_true_in+1)  * y_pred_in)
        false_positives = np.sum(y_true_in1 ==1)

        y_pred_in1 = ((y_pred_in+1)  * y_true_in)
        false_negatives = np.sum(y_pred_in1 ==1)

        y_pred_in2 = ((y_pred_in+1)  * (y_true_in+1))
        true_negatives = np.sum(y_pred_in2 ==1)
        # for k in range(y_true_in.shape[0]):
        #     # true_positives = 0   # Correct objects
        #     # false_positives = 0  # Missed objects
        #     # false_negatives = 0
        #     for i in range(512):
        #         for j in range(512):
        #             # print(y_true_in[k,i,j],y_pred_in[k,i,j])
        #             if y_true_in[k,i,j]>0 and y_pred_in[k,i,j]>=50:
        #                 print(y_true_in[k,i,j],y_pred_in[k,i,j])
        #                 true_positives +=1
        #             if y_true_in[k,i,j]==0 and y_pred_in[k,i,j]>=50:
        #                 false_positives +=1
        #             if y_true_in[k,i,j]>0 and y_pred_in[k,i,j]<50:
        #                 false_negatives +=1
            # Precision helper function
            # def precision_at(threshold, iou):
            #     matches = iou > threshold
            #     true_positives = np.sum(matches, axis=1) == 1   # Correct objects
            #     false_positives = np.sum(matches, axis=0) == 0  # Missed objects
            #     false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
            #     true_positives, false_positives, false_negatives = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
            #     return true_positives, false_positives, false_negatives

            # Loop over IoU thresholds

            # if print_table:
            #     print("Thresh\tTP\tFP\tFN\tPrec.")
            # for t in np.arange(0.5, 1.0, 0.05):
                # true_positives, false_positives, false_negatives = precision_at(t, iou)
        if (true_positives + false_positives) > 0:
            p1 = true_positives / (true_positives + false_positives )
        else:
            p1 = 0
        if (true_positives + false_negatives) > 0:
            p2 = true_positives / (true_positives + false_negatives )
        else:
            p2 = 0
        if p1+p2>0:
            p3 = 2*(p1*p2)/(p1+p2)
        else:
            p3 = 0

        if (false_positives + true_negatives) > 0:
            p4 = false_positives/(false_positives + true_negatives)
        else:
            p4 = 0
        # print("\t{}\t{}\t{}\t{:1.3f}\t{:1.3f}".format( true_positives, false_positives, false_negatives, p1 ,p2))
        # prec.append(p1)
        # reca.append(p2)
        # prec.append(p1)
        # recall.append(p2)
        # if print_table:
        #     print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
        return p1,p2,p4

    # else:
    #     if np.sum(y_pred_in.flatten()) == 0:
    #         return 1
    #     else:
    #         return 0
