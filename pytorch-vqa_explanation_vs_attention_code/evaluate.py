import sys
import os.path
import math
import json
import scipy.misc, scipy.stats
from pyemd import emd_samples
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import misc.config as config
import misc.data1 as data
import misc.model as model
import misc.utils as utils
from skimage import transform, filters
import matplotlib.pyplot as plt
plt.switch_backend('agg')


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


def rmse(y_hat,y):
    """Compute root mean squared error"""
    return torch.sqrt(torch.mean((y_hat - y).pow(2)))
def mse(y_hat,y):
    """Compute root mean squared error"""
    return torch.mean((y_hat - y).pow(2))

def run(net,net_dis, loader, optimizer,optimizer_dis, tracker, train=False, prefix='', epoch=0):
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
    d_loss = nn.BCELoss().cuda()
    cnt = 0
    hat_id = np.array(np.genfromtxt('../pytorch-vqa_att_new/hat_id.txt'), np.int)
    hat_id_wrong = np.array(np.genfromtxt('../pytorch-vqa_att_new/hat_id_wrong.txt'), np.int)
    rankCorr = 0
    pVal = 0
    # emd = 0
    avg_loss_dis=0
    for v, q, q_id, a, atxt, idx, image_id, q_len in tq:
        var_params = {
            'volatile': False,
            'requires_grad': True,
        }
        # var_params = {
        #     'volatile': not train,
        #     'requires_grad': False,
        # }
        v = Variable(v.cuda(async=True), **var_params)
        q = Variable(q.type(torch.FloatTensor).cuda(async=True), **var_params)
        a = Variable(a.type(torch.FloatTensor).cuda(async=True), **var_params)
        q_len = Variable(q_len.type(torch.FloatTensor).cuda(async=True), **var_params)

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
 
            gradients = get_p_gradcam(v.grad.cpu().data.numpy(), v.cpu().data.numpy())
            real_input=torch.Tensor(gradients).view(-1,196).cuda()# to convert numpy to tensor
            gt_real=torch.ones(real_input.size(0)).cuda()
            gt_fake=torch.zeros(real_input.size(0)).cuda()
            #for real
            output_real=net_dis(real_input)
            loss_r=d_loss(output_real,gt_real)
            optimizer_dis.zero_grad()
            loss_r.backward(retain_graph=True)
            optimizer_dis.step()

            #for fake
            output_fake=net_dis(p_att)
            loss_f=d_loss(output_fake,gt_fake)
            optimizer_dis.zero_grad()
            loss_f.backward(retain_graph=True)
            optimizer_dis.step()

            #for generator
            output_fake_grl=net_dis(p_att)
            loss_grl=d_loss(output_fake_grl,gt_real)
            optimizer.zero_grad()
            loss_grl.backward()
            optimizer.step()

            loss_dis=loss_r+loss_f
            avg_loss_dis +=loss_dis.item()
            if cnt%20 ==0:
                print('avg_loss_dis',avg_loss_dis/20)
                avg_loss_dis=0


            total_iterations += 1

            loss_tracker.append(loss.item())
            loss_grad_tracker.append(loss_dis.item())
            acc_tracker.append(acc.mean())
            fmt = '{:.4f}'.format
            tq.set_postfix(loss=fmt(loss_tracker.mean.value),loss_dis=fmt(loss_grad_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

        else:
            # Uncomment the following lines when you need to get the attention and gradcam visualizations
            # ----------------------------------------
            
            loss.backward()
            gradients = get_p_gradcam(v.grad.cpu().data.numpy(), v.cpu().data.numpy())

            for i, imgIdx in enumerate(image_id):
                imgIdx = "COCO_" + prefix + "2014_000000" + "0" * (6-len(str(imgIdx.numpy()))) + str(imgIdx.numpy()) + ".jpg"
                rawImg = scipy.misc.imread(os.path.join('/VQA/Images/mscoco', prefix + '2014/' + imgIdx), mode='RGB')
                rawImg = scipy.misc.imresize(rawImg, (448, 448), interp='bicubic')
                # plt.imsave("Results/RawImages/ep" + str(epoch) + "_" + "cnt" + str(cnt) + "_" + str(i) + "raw.png", rawImg)
                plt.imsave("Results/AttImages/ep" + str(epoch) + "_" + "cnt" + str(cnt) + "_" + str(i) + "att.png", get_blend_map_att(rawImg/255.0, p_att[i].cpu().data.numpy()))
                cv2.imwrite("Results/GradcamImages/ep" + str(epoch) + "_" + "cnt" + str(cnt) + "_" + str(i) + "gradcam.png", np.uint8(255 * get_blend_map_gradcam(rawImg/255.0, gradients[i])))
            
            # ----------------------------------------

            for i, quesId in enumerate(q_id):
                # quesIndices.append(quesId)
                if quesId.numpy() not in hat_id:
                    continue
                else:
                    cnt = cnt + 1
                    hatIdx = str(quesId.numpy())[:-1] + "0" + str(quesId.numpy())[-1]
                    if int(hatIdx) not in hat_id_wrong:
                        hatIdx = str(quesId.numpy())[:-1] + "1" + str(quesId.numpy())[-1]
                        if int(hatIdx) not in hat_id_wrong:
                            hatIdx = str(quesId.numpy())[:-1] + "2" + str(quesId.numpy())[-1]
                            
                    hatIdx = hatIdx[:-1] + "_" + str(int(hatIdx[-1])+1) + ".png"
                    hatImg = scipy.misc.imread('/HAT_dataset/vqahat_val/' + hatIdx)
                    hatAtt = transform.resize(hatImg, ((196, 1)), order=3)
                    hatAtt = np.reshape(hatAtt, (196))
                    hatAtt -= hatAtt.min()
                    if hatAtt.max() > 0:
                        hatAtt /= hatAtt.max()
                    # print(hatAtt)
                    # print(hatAtt.shape)
                    # print(p_att[i].cpu().data.numpy())
                    # print(p_att[i].cpu().data.numpy().shape)
                    nowRankCorr, nowPVal = scipy.stats.spearmanr(hatAtt, p_att[i].cpu().data.numpy())
                    rankCorr += nowRankCorr
                    pVal += nowPVal
                    # emd += emd_samples(hatAtt, p_att[i].cpu().data.numpy())
                    print("update cnt: ", cnt, " ", quesId.numpy())
                    print("update rankcorrelation : ", rankCorr/cnt)
                    print("update p Value : ", pVal/cnt)
                    # print("update emd : ", emd/cnt)

            # store information about evaluation of this minibatch
            _, answer = out.data.cpu().max(dim=1)
            answ.append(answer.view(-1))
            accs.append(acc.view(-1))
            idxs.append(idx.view(-1).clone())
    
            loss_tracker.append(loss.item())
            acc_tracker.append(acc.mean())
            fmt = '{:.4f}'.format
            tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

        # cnt = cnt + 1

    print("total match : ", cnt)
    print("rankcorrelation : ", rankCorr/cnt)
    print("p Value : ", pVal/cnt)
    # print("emd : ", emd/cnt)

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
    target_name = os.path.join('logs', '{}'.format(name))
    print('will save to {}'.format(target_name))

    cudnn.benchmark = True

    train_loader = data.get_loader(train=True)
    val_loader = data.get_loader(val=True)

    net = nn.DataParallel(model.Net(train_loader.dataset.num_tokens)).cuda()

    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])
    tracker = utils.Tracker()
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

    net_dis = nn.DataParallel(model.Net_Discriminitor(196)).cuda()
    optimizer_dis = optim.Adam([p for p in net_dis.parameters() if p.requires_grad],lr = 0.0002, betas = (0.5, 0.999))
    

    # Uncomment the following lines while evaluating or want to start by loading checkpoint
    # ----------------------------------------

    ckp = torch.load('logs/2018-11-11_19:22:46_127.pth')
    name = ckp['name']
    # tracker = ckp['tracker']
    config_as_dict = ckp['config']
    net.load_state_dict(ckp['weights'])
    net_dis.load_state_dict(ckp['weights_grad'])
    train_loader.dataset.vocab = ckp['vocab']

    # ----------------------------------------

    for i in range(127, 128):
        # _ = run(net,net_dis, train_loader, optimizer,optimizer_dis, tracker, train=True, prefix='train', epoch=i)
        r = run(net,net_dis, val_loader, optimizer, optimizer_dis,tracker, train=False, prefix='val', epoch=i)

        results = {
            'name': name,
            'tracker': tracker.to_dict(),
            'config': config_as_dict,
            'weights': net.state_dict(),
            'weights_grad': net_dis.state_dict(),
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

