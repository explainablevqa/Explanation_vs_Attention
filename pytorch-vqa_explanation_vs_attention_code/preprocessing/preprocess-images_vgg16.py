import h5py
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.models as models
from tqdm import tqdm

import misc.config as config
import misc.data as data
import misc.utils as utils
# from resnet import resnet as caffe_resnet
import torchvision.models as models


import torch
import torch.nn as nn
from torchvision import models

# original_model = models.alexnet(pretrained=True)

# class AlexNetConv4(nn.Module):
#             def __init__(self):
#                 super(AlexNetConv4, self).__init__()
#                 self.features = nn.Sequential(*list(original_model.features.children())[:-3])# stop at conv4
                
#             def forward(self, x):
#                 x = self.features(x)
#                 return x

# model = AlexNetConv4()
# to extract conv5 feature
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.model = caffe_resnet.resnet152(pretrained=True)
        self.model = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(self.model.features.children()))#[:-3])# stop at conv5

    def forward(self, x):
        x = self.features(x)
        return x

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # self.model = caffe_resnet.resnet152(pretrained=True)
#         self.model = models.vgg16(pretrained=True)

#         def save_output(module, input, output):
#             self.buffer = output
#         self.model.conv2[5].register_forward_hook(save_output)

#     def forward(self, x):
#         self.model(x)
#         return self.buffer


def create_coco_loader(*paths):
    transform = utils.get_transform(config.image_size, config.central_fraction)
    datasets = [data.CocoImages(path, transform=transform) for path in paths]
    dataset = data.Composite(*datasets)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.preprocess_batch_size,
        num_workers=config.data_workers,
        shuffle=False,
        pin_memory=True,
    )
    return data_loader


def main():
    cudnn.benchmark = True

    net = Net().cuda()
    net.eval()

    loader = create_coco_loader(config.train_path, config.val_path)
    features_shape = (
        len(loader.dataset),
        config.output_features,
        config.output_size,
        config.output_size
    )

    with h5py.File(config.preprocessed_path, libver='latest') as fd:
        features = fd.create_dataset('features', shape=features_shape, dtype='float16')
        coco_ids = fd.create_dataset('ids', shape=(len(loader.dataset),), dtype='int32')

        i = j = 0
        for ids, imgs in tqdm(loader):
            imgs = Variable(imgs.cuda(async=True), volatile=True)
            out = net(imgs)
            # print(out.shape)

            j = i + imgs.size(0)
            features[i:j, :, :] = out.data.cpu().numpy().astype('float16')
            coco_ids[i:j] = ids.numpy().astype('int32')
            i = j


if __name__ == '__main__':
    main()
