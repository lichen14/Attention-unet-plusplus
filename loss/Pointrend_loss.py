import torch
import torch.nn as nn
import torch.nn.functional as F
# from ..new_archs.PointHead import point_sample
from loss.Dice_loss import DiceLoss
from metrics import dice_coef, batch_iou, mean_iou, iou_score,compute_iou,compute_dice,point_sample
from skimage.io import imread, imsave

class PointRendLoss(nn.CrossEntropyLoss):
    '''
    由于pointrend有整体预测和细分点预测两部分，所以loss也由两部分相加组成
    #result['res2']: [2, 256, 192, 192], 即xception的c1层提取到的特征
    #result['coarse']: [2, 19, 48, 48]
    #result['rend']: [2, 19, 48]
    #result['points']:[2, 48, 2]
    #gt:[2, 768, 768], 即图片对应的label
    pred:[B,1,256,256]
    
    '''
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(PointRendLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index
        self.loss = DiceLoss()
        # self.iou_func = iou_score()

    def forward(self, result, gt,point_feature,points):
        #result, gt = tuple(inputs)
        #result['res2']: [B, 256, 192, 192], 即xception的c1层提取到的特征
        #result['coarse']: [B, 19, 16, 16], 即xception的c4层提取到的特征
        #result['rend']: [B, 1, 16],更精准的预测结果
        #result['points']:[B, 16, 2],不确定点的位置
        #gt:[B, 768, 768], 即图片对应的label
        
        #pred:[2, 19, 768, 768]，将粗糙预测的插值到label大小
        # if train_or_inference:  #train
        # print('gt size is ',gt.shape[-2:])    [256,256]
        # print('coarse size is ',result["coarse"]) 
        pred = result#F.interpolate(result, gt.shape[-2:], mode="bilinear", align_corners=True)
        print('pred size is ',pred.shape) 
        print('gt size is ',gt.shape) 
        # else:   #inference
            # pred = F.interpolate(result, gt.shape[-2:], mode="bilinear", align_corners=True)
        # pred = pred.squeeze(dim=1)
        
        # gt_ = int(gt)
        #整体像素点的交叉熵loss
        # seg_loss = F.cross_entropy(pred, gt, ignore_index=self.ignore_index)
        # print(pred.shape)
        # print(gt.shape)

        seg_loss = self.loss(pred,gt)
        # seg_iou = iou_score(pred,gt)
        # if train_or_inference==False:   #inference
            # return seg_loss
        #根据不确定点坐标获得不确定点对应的gt
        gt_points = point_sample(
            gt.float().unsqueeze(1),
            points,
            mode="nearest",
            align_corners=False
        ).squeeze_(1).long()
        #不确定点的交叉熵loss
        # print('gt_points shape',gt_points.shape) #[B,16]
        # print("points_loss!result[rend] shape is",result["rend"][0,0,:],gt_points[0,:])
        points_loss = F.cross_entropy(point_feature, gt_points, ignore_index=self.ignore_index)
        # points_loss = self.loss(result["rend"], gt_points)
        #整体+不确定点loss
        loss = seg_loss + points_loss
        # output = pred.cpu().detach().numpy()
        # for j in range(output.shape[0]):
        #     imsave('/var/www/nextcloud/data/dbc2017/files/output/'+str(j)+'.png', (output[j,:,:]))
        
        print("seg loss is {:.3f}, point loss is {:.3f}, loss is {:.3f}".format(seg_loss,points_loss,loss))
        return seg_loss
        # return dict(loss=loss)
    