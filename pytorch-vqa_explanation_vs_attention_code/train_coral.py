import sys
import os.path
import math
import json
import scipy.misc
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import misc.config as config
import misc.data as data
import misc.model as model
import misc.utils as utils
from skimage import transform, filters
import matplotlib.pyplot as plt
plt.switch_backend('agg')
# from mmd import mix_rbf_mmd2
import torch.nn.functional as F

def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5**(float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


total_iterations = 0


def get_p_gradcam(grads_val, target):
    cams = []
    for i in range(grads_val.shape[0]):
        # print("grads_val_i", grads_val[i].shape)#grads_val_i (512, 14, 14)
        weights = np.mean(grads_val[i], axis = (1, 2))
        # print("weights", weights.shape)#weights (512,)
        cam = np.zeros(target[i].shape[1 : ], dtype = np.float32)

        for k, w in enumerate(weights):
            cam += w * target[i, k, :, :]

        cams.append(cam)

    return cams


def get_blend_map_gradcam(img, gradcam_map):
    cam = np.maximum(gradcam_map, 0)
    cam = cv2.resize(cam, img.shape[:2])
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def get_blend_map_att(img, att_map, blur=True, overlap=True):
    att_map -= att_map.min()
    if att_map.max() > 0:
        att_map /= att_map.max()
    att_map = att_map.reshape((14, 14))
    att_map = transform.resize(att_map, (img.shape[:2]), order = 3)
    if blur:
        att_map = filters.gaussian_filter(att_map, 0.08*max(img.shape[:2]))
        att_map -= att_map.min()
        att_map /= att_map.max()
    cmap = plt.get_cmap('jet')
    att_map_v = cmap(att_map)
    att_map_v = np.delete(att_map_v, 3, 2)
    if overlap:
        att_map = (1-att_map**0.7).reshape(att_map.shape + (1,))*img + (att_map**0.7).reshape(att_map.shape+(1,)) * att_map_v
    return att_map


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

def CORAL(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 1, keepdim=True) - source
    xc = torch.matmul(torch.transpose(xm, 0, 1), xm)

    # target covariance
    xmt = torch.mean(target, 1, keepdim=True) - target
    xct = torch.matmul(torch.transpose(xmt, 0, 1), xmt)
    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss/(4*d*4)
    return loss

def rmse(y_hat,y):
    """Compute root mean squared error"""
    return torch.sqrt(torch.mean((y_hat - y).pow(2)))
def mse(y_hat,y):
    """Compute root mean squared error"""
    return torch.mean((y_hat - y).pow(2))

def run(net, loader, optimizer,optimizer_grad, tracker, train=False, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.train()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        answ = []
        idxs = []
        accs = []

    tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0, file=sys.stdout)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    loss_grad_tracker = tracker.track('{}_loss_grad'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

    log_softmax = nn.LogSoftmax().cuda()
    mmd_loss = nn.LogSoftmax().cuda()

    # sigma for MMD
    base = 1.0
    sigma_list = [1, 2, 4, 8, 16]
    sigma_list = [sigma / base for sigma in sigma_list]
    cnt = 0
    avg_loss_grad=0
    for v, q, a, idx, image_id, q_len in tq:
        var_params = {
            'volatile': False,
            'requires_grad': True,
        }
        v = Variable(v.cuda(async=True), **var_params)
        q = Variable(q.type(torch.FloatTensor).cuda(async=True), **var_params)
        a = Variable(a.type(torch.FloatTensor).cuda(async=True), **var_params)
        q_len = Variable(q_len.type(torch.FloatTensor).cuda(async=True), **var_params)
        # v.retain_grad = True

        out,p_att = net(v, q, q_len)
        nll = -log_softmax(out)
        loss = (nll * a / 10).sum(dim=1).mean()
        acc = utils.batch_accuracy(out.data, a.data).cpu()

        if train:
            global total_iterations
            update_learning_rate(optimizer, total_iterations)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            #New code
            gradients = get_p_gradcam(v.grad.cpu().data.numpy(), v.cpu().data.numpy())
            final_att=net_grad(p_att)
            grad_input=torch.Tensor(gradients).view(-1,196).cuda()
            loss_grad = CORAL(final_att,grad_input)

            avg_loss_grad +=loss_grad.item()
            if cnt%20 ==0:
                print('avg_loss_grad',avg_loss_grad)
            optimizer_grad.zero_grad()
            loss_grad.backward(retain_graph=True)
            optimizer_grad.step()
            #to update generator
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            total_iterations += 1
            
            loss_tracker.append(loss.item())
            loss_grad_tracker.append(loss_grad.item())
            acc_tracker.append(acc.mean())
            fmt = '{:.4f}'.format
            tq.set_postfix(loss=fmt(loss_tracker.mean.value),loss_grad=fmt(loss_grad_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

        else:
            # Uncomment the following lines when you need to get the attention and gradcam visualizations
            # ----------------------------------------
            
            loss.backward()
            gradients = get_p_gradcam(v.grad.cpu().data.numpy(), v.cpu().data.numpy())

            for i, imgIdx in enumerate(image_id):
                imgIdx = "COCO_" + prefix + "2014_000000" + "0" * (6-len(str(imgIdx.numpy()))) + str(imgIdx.numpy()) + ".jpg"
                rawImg = scipy.misc.imread(os.path.join('VQA/workspace_project/contex_attention_vqa/VQA/Images/mscoco', prefix + '2014/' + imgIdx), mode='RGB')
                rawImg = scipy.misc.imresize(rawImg, (448, 448), interp='bicubic')
                # plt.imsave("Results/RawImages/ep" + str(epoch) + "_" + "cnt" + str(cnt) + "_" + str(i) + "raw.png", rawImg)
                plt.imsave("Results/AttImages/ep" + str(epoch) + "_" + "cnt" + str(cnt) + "_" + str(i) + "att.png", get_blend_map_att(rawImg/255.0, p_att[i].cpu().data.numpy()))
                cv2.imwrite("Results/GradcamImages/ep" + str(epoch) + "_" + "cnt" + str(cnt) + "_" + str(i) + "gradcam.png", np.uint8(255 * get_blend_map_gradcam(rawImg/255.0, gradients[i])))
            
            # ----------------------------------------

            # store information about evaluation of this minibatch
            _, answer = out.data.cpu().max(dim=1)
            answ.append(answer.view(-1))
            accs.append(acc.view(-1))
            idxs.append(idx.view(-1).clone())
    
            loss_tracker.append(loss.item())
            acc_tracker.append(acc.mean())
            fmt = '{:.4f}'.format
            tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

        cnt = cnt + 1

    if not train:
        answ = list(torch.cat(answ, dim=0))
        accs = list(torch.cat(accs, dim=0))
        idxs = list(torch.cat(idxs, dim=0))
        return answ, accs, idxs


def main():
    if len(sys.argv) > 1:
        name = ' '.join(sys.argv[1:])
    else:
        from datetime import datetime
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # target_name = os.path.join('logs', '{}.pth'.format(name))
    target_name = os.path.join('logs', '{}'.format(name))
    print('will save to {}'.format(target_name))

    cudnn.benchmark = True

    train_loader = data.get_loader(train=True)
    val_loader = data.get_loader(val=True)

    net = nn.DataParallel(model.Net(train_loader.dataset.num_tokens)).cuda()

    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])
    tracker = utils.Tracker()
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

    # net_grad = nn.DataParallel(model.Net_grad(196)).cuda()
    optimizer_grad = optim.Adam([p for p in net.parameters() if p.requires_grad])
    

    # Uncomment the following lines while evaluating or want to start by loading checkpoint
    # ----------------------------------------

    if config.pretrain == 1 :
        ckp = torch.load('logs/2018-10-28_12:47:45_180.pth')
        name = ckp['name']
        # tracker = ckp['tracker']
        config_as_dict = ckp['config']
        net.load_state_dict(ckp['weights'])
        net_dis.load_state_dict(ckp['weights_grad'])
        train_loader.dataset.vocab = ckp['vocab']

    # ----------------------------------------

    for i in range(0, config.epochs):
        _ = run(net, train_loader, optimizer,optimizer_grad, tracker, train=True, prefix='train', epoch=i)
        r = run(net, val_loader, optimizer, optimizer_grad,tracker, train=False, prefix='val', epoch=i)

        results = {
            'name': name,
            'tracker': tracker.to_dict(),
            'config': config_as_dict,
            'weights': net.state_dict(),
            'weights_grad': net_grad.state_dict(),
            'eval': {
                'answers': r[0],
                'accuracies': r[1],
                'idx': r[2],
            },
            'vocab': train_loader.dataset.vocab,
        }
        torch.save(results, target_name+"_"+str(i)+".pth")


if __name__ == '__main__':
    main()
