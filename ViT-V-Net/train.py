from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch, models
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from models import CONFIGS as CONFIGS_ViT_seg
from natsort import natsorted
from Functions import generate_grid, Dataset_epoch_crop, Predict_dataset_crop
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

def MSE_torch(x, y):
    return torch.mean((x - y) ** 2)

def main():
    batch_size = 1
    # train_dir = 'D:/DATA/JHUBrain/Train/'
    # val_dir = 'D:/DATA/JHUBrain/Val/'
    datapath = './data/OASIS'
    train_dir = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/aligned_norm.nii.gz'))[0:255]
    val_dir = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/aligned_norm.nii.gz'))[255:261]
    save_dir = 'ViTVNet_reg0.02_mse_diff/'
    lr = 0.0001
    epoch_start = 0
    max_epoch = 500
    cont_training = False
    config_vit = CONFIGS_ViT_seg['ViT-V-Net']
    reg_model = utils.register_model((160, 160, 160), 'nearest')
    reg_model.cuda()
    model = models.ViTVNet(config_vit, img_size=(160, 160, 160))
    if cont_training:
        epoch_start = 335
        model_dir = 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[0])['state_dict']
        model.load_state_dict(best_model)
    else:
        updated_lr = lr
    model = nn.DataParallel(model,[0,1,2,3])
    model.cuda()
    train_composed = transforms.Compose([trans.RandomFlip(0),
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])

    val_composed = transforms.Compose([trans.Seg_norm(), #rearrange segmentation label to 1 to 46
                                       trans.NumpyType((np.float32, np.int16)),
                                        ])

    train_loader = DataLoader(Dataset_epoch_crop(train_dir, norm=True), batch_size=batch_size,
                                         shuffle=True, num_workers=4, pin_memory=True)
    # val_loader = DataLoader(Predict_dataset_crop(fixed_img, imgs, fixed_label, labels, norm=True),
                                                #   batch_size=1,
                                                #   shuffle=False, num_workers=2)
    # train_set = datasets.JHUBrainDataset(glob.glob(train_dir + '*.pkl'), transforms=train_composed)
    # val_set = datasets.JHUBrainInferDataset(glob.glob(val_dir + '*.pkl'), transforms=val_composed)
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(Dataset_epoch_crop(val_dir, norm=True), batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion = nn.MSELoss()
    criterions = [criterion]
    weights = [1]
    # prepare deformation loss
    criterions += [losses.Grad3d(penalty='l2')]
    weights += [0.02]
    best_mse = 0
    writer = SummaryWriter(log_dir='ViTVNet_log')
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            # print(data.shape,'ooooo')
            x = data[0]
            y = data[1]
            x_in = torch.cat((x,y), dim=1)
            output = model(x_in)
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], y) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del x_in
            del output
            # flip fixed and moving images
            loss = 0
            x_in = torch.cat((y, x), dim=1)
            output = model(x_in)
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], x) * weights[n]
                loss_vals[n] += curr_loss
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_vals[0].item()/2, loss_vals[1].item()/2))

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        '''
        Validation
        '''
        eval_dsc = AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
                # x = x.squeeze(0).permute(1, 0, 2, 3)
                # y = y.squeeze(0).permute(1, 0, 2, 3)
                x_in = torch.cat((x, y), dim=1)
                output = model(x_in)
                def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                dsc = utils.dice_val(def_out.long(), y_seg.long(), 46)
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)
        best_mse = max(eval_dsc.avg, best_mse)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mse': best_mse,
            'optimizer': optimizer.state_dict(),
        }, save_dir='experiments/'+save_dir, filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
        writer.add_scalar('MSE/validate', eval_dsc.avg, epoch)
        plt.switch_backend('agg')
        pred_fig = comput_fig(def_out)
        x_fig = comput_fig(x_seg)
        tar_fig = comput_fig(y_seg)
        writer.add_figure('input', x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure('ground truth', tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure('prediction', pred_fig, epoch)
        plt.close(pred_fig)
        loss_all.reset()
    writer.close()

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    # GPU_avai = torch.cuda.is_available()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()