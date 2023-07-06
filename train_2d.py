"""

深度监督下的训练脚本
"""

from time import time
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from collections import OrderedDict
import new_archs
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt # plt 用于显示图片
from skimage.io import imread, imsave
from loss.Dice_loss import DiceLoss
from loss.Pointrend_loss import PointRendLoss
import torch.optim as optim
from torch.optim import lr_scheduler
#from net.DialResUNet import net
import pandas as pd
import joblib
from dataset.dataset_2d import train_fix_ds,valid_fix_ds,test_fix_ds
from metrics import dice_coef, batch_iou, mean_iou, iou_score,compute_iou,compute_dice
from utils import str2bool, count_params
import losses
# 定义超参数
on_server = True

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0,1,2" if USE_CUDA else "cpu")
cudnn.benchmark = True
Epoch = 20
leaing_rate_base = 1e-2
alpha = 0.33
num_workers = 1 if on_server is False else 3
pin_memory = False if on_server is False else True
arch_names = list(new_archs.__dict__.keys())
loss_names = list(losses.__dict__.keys())
start = time()
# torch.cuda.eset_device(0)
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: NestedUNet)')
    parser.add_argument('--deepsupervision', default=False, type=str2bool)
    parser.add_argument('--dataset', default=None,
                        help='dataset name')
    parser.add_argument('--input-channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='png',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='png',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                            ' | '.join(loss_names) +
                            ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=200, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=12, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('-o','--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()

    return args

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, train_loader, model, criterion, optimizer, scheduler,epoch):
    losses = AverageMeter()
    ious = AverageMeter()

    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # optimizer.zero_grad()
        # input=Variable(input,requires_grad=True)
        # print(input.shape)
        # print(target.shape)
        #
        # t=input[0,0,:,:]
        # # print(t.shape)
        # # tt=np.zeros((256,256,3))
        # # tt[:,:,0]=t[0,:,:]
        # # # tt[:,:,1]=t[0,:,:]
        # # # tt[:,:,2]=t[1,:,:]
        # # print(tt[0,100,100])
        # # m=target[1,:,:,:]
        # # mm=np.zeros((256,256))
        # mm=target[0,:,:]
        # ##print(np.maximum(mm, -1))
        # # # mm[:,:,1]=target[1,:,:]
        # # # mm[:,:,2]=target[2,:,:]
        # # # t = t.transpose((1,2,0))
        # # # print(tt.shape)
        # plt.subplot(121)
        # plt.imshow(t)
        # # plt.show()
        # plt.subplot(122)
        # plt.imshow(mm)#, cmap='Greys_r') # 显示图片
        # plt.axis('off') # 不显示坐标轴
        # plt.show()

        # print(input,'\t',target)
        input = input.to(device)


        # input = input.cuda()


        # target=Variable(target,requires_grad=True)
        # target = target.to(device)



        target = target.to(device)
        target = target/255




        # compute output

        if args.deepsupervision:
            outputs = model(input)
            loss = 0
            # output = model(input)[-1]
            for output in outputs:
                output = output.squeeze(dim=1)

                loss += criterion(output, target)
                # print(loss)
            loss /= len(outputs)
            iou = iou_score(outputs[-1].squeeze(dim=1), target)
        else:
            # with torch.no_grad():

            output = model(input)#[-1]
            # print(output.shape)
            # output =output/255
            #print(output[0,0,150,150])
            output = output.squeeze(dim=1)
            #print(target[0,150,150])

            # print(output.shape)
            #print(target[0,150,150])
            loss = criterion(output, target)
            # print(type(loss))
            # iou = compute_iou(output, target)
            iou= iou_score(output, target)

        losses.update(loss.item(), input.size(0))
        # print(loss.item())
        # print(losses)
        ious.update(iou, input.size(0))

        # loss = loss / 10
        # loss.backward()
        # if((i+1)%10)==0:
        #     optimizer.step()
        #     optimizer.zero_grad()

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        if i % 100 is 0:
            print('epoch:{}, step:{}, loss:{:.3f},iou:{:.3f}, time:{:.3f} min'
              .format(epoch, i, loss.item(),iou.item(), (time()-start) / 60))
        # torch.cuda.empty_cache()

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])


    return log

def validate(args, val_loader, model, criterion,scheduler,epoch):
    losses = AverageMeter()
    ious = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            # print(input.shape)
            # print(target.shape)
            # t=input[0,0,:,:]
            # # print(t.shape)
            # # tt=np.zeros((256,256,3))
            # # tt[:,:,0]=t[0,:,:]
            # # # tt[:,:,1]=t[0,:,:]
            # # # tt[:,:,2]=t[1,:,:]
            # # print(tt[0,100,100])
            # # m=target[1,:,:,:]
            # # mm=np.zeros((256,256))
            # mm=target[0,:,:]
            # ##print(np.maximum(mm, -1))
            # # # mm[:,:,1]=target[1,:,:]
            # # # mm[:,:,2]=target[2,:,:]
            # # # t = t.transpose((1,2,0))
            # # # print(tt.shape)
            # plt.subplot(121)
            # plt.imshow(t)
            # # plt.show()
            # plt.subplot(122)
            # plt.imshow(mm)#, cmap='Greys_r') # 显示图片
            # plt.axis('off') # 不显示坐标轴
            # plt.show()

            input = input.to(device)
            # print(input.shape)
            # input=Variable(input,requires_grad=True)


            target = target.to(device)
            # target=Variable(target,requires_grad=True)
            target = target/255
            # compute output
            if args.deepsupervision:
                output = model(input)[-1]
                output = output.squeeze(dim=1)
                # imsave(('/var/www/nextcloud/data/dbc2017/files/test1/Photos/'+'%d.png'%i), (output[0,:,:].data.cpu().numpy()))

                # print(output.shape)
                # print(output.shape)
                # print(target.shape)
                loss = criterion(output, target)
                # imsave('output/target.png', (target[0,:,:].cpu().detach().numpy()))
                iou = iou_score(output, target)
            else:
                # with torch.no_grad():
                output = model(input)#[-1]
                output = output.squeeze(dim=1)
                # imsave(('/var/www/nextcloud/data/dbc2017/files/test1/Photos/'+'%d.png'%i), (output[0,:,:].data.cpu().numpy()))

                # print(output.shape)
                # print(target.shape)
                loss = criterion(output, target)
                # iou = compute_iou(output, target)
                iou = iou_score(output, target)

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            if i % 100 is 0:
                print('epoch:{}, step:{}, loss:{:.3f},iou:{:.3f}, time:{:.3f} min'
                      .format(epoch, i, loss.item(),iou.item(), (time()-start) / 60))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])
    # scheduler.step(loss.item())
    return log

def main():
    args = parse_args()
    # 定义数据加载
    if not os.path.exists('models/tumor/%s' %args.name):
        os.makedirs('models/tumor/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/tumor/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/tumor/%s/args.pkl' %args.name)

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = losses.__dict__[args.loss]().to(device)

    train_loader = DataLoader(train_fix_ds, args.batch_size, True, num_workers=num_workers, pin_memory=pin_memory)
    valid_dl = DataLoader(valid_fix_ds, args.batch_size, False, num_workers=num_workers, pin_memory=pin_memory)
    # 定义损失函数
    loss_func = DiceLoss()#dice_coef#compute_dice#

    net = new_archs.__dict__[args.arch](args).to(device)
    model = torch.nn.DataParallel(net, device_ids = [0,1,2]).to(device)
    print("count_params:",count_params(model))
    # 定义优化器
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    # 学习率衰减
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3600,
             verbose=True, threshold=1e-2, threshold_mode='rel',
             cooldown=0, min_lr=1e-7, eps=1e-8)

    log = pd.DataFrame(index=[], columns=[
    'epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'
    ])
    # 训练网络
    best_iou = 0
    trigger = 0

    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch, args.epochs))
        train_log = train(args, train_loader, model, loss_func, optimizer, scheduler,epoch)
        val_log = validate(args, valid_dl, model, loss_func,scheduler,epoch)

        # lr_decay.step()
        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f -  time:%.3f min'
            %(train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou'], (time()-start) / 60))

        tmp = pd.Series([
            epoch,
            args.lr,
            train_log['loss'],
            train_log['iou'],
            val_log['loss'],
            val_log['iou'],
        ], index=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/tumor/%s/log.csv' %args.name, index=False)

        trigger += 1
        #torch.save(model.state_dict(), 'models/tumor/%s/%d-model.pth' %(args.name,epoch))
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/tumor/%s/model.pth' %args.name)
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0
        # else:
        #     model.load_state_dict(torch.load('models/tumor/%s/model.pth' %args.name))

        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break

        #torch.cuda.empty_cache()

        # mean_loss = []
        #
        # for step, (ct, seg) in enumerate(train_dl):
        #     # print(ct.shape,seg.shape)
        #     ct = ct.cuda()
        #     seg = seg.cuda()
        #
        #     outputs = net(ct)
        #
        #     loss1 = loss_func(outputs[0], seg)
        #     loss2 = loss_func(outputs[1], seg)
        #     loss3 = loss_func(outputs[2], seg)
        #     loss4 = loss_func(outputs[3], seg)
        #
        #     loss = (loss1 + loss2 + loss3) * alpha + loss4
        #
        #     mean_loss.append(loss4.item())
        #
        #     opt.zero_grad()
        #     loss.backward()
        #     opt.step()
        #
        #     if step % 20 is 0:
        #         print('epoch:{}, step:{}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4:{:.3f}, time:{:.3f} min'
        #               .format(epoch, step, loss1.item(), loss2.item(), loss3.item(), loss4.item(), (time() - start) / 60))
        #
        # mean_loss = sum(mean_loss) / len(mean_loss)
        #
        # if epoch % 10 is 0 and epoch is not 0:
        #
        #     # 网络模型的命名方式为：epoch轮数+当前minibatch的loss+本轮epoch的平均loss
        #     torch.save(net.state_dict(), './module/net{}-{:.3f}-{:.3f}.pth'.format(epoch, loss.item(), mean_loss))
        #
        # if epoch % 15 is 0 and epoch is not 0:
        #
        #     alpha *= 0.8
if __name__ == '__main__':
    main()
# 深度监督的系数变化
# 1.000
# 0.800
# 0.640
# 0.512
# 0.410
# 0.328
# 0.262
# 0.210
# 0.168
# 0.134
# 0.107
# 0.086
# 0.069
# 0.055
# 0.044
# 0.035
# 0.028
# 0.023
# 0.018
# 0.014
# 0.012
# 0.009
# 0.007
# 0.006
# 0.005
# 0.004
# 0.003
# 0.002
# 0.002
# 0.002
# 0.001
# 0.001
# 0.001
# 0.001
# 0.001
# 0.000
# 0.000
