import copy
import functools

import torch
import imageio
import numpy as np
import pretrainedmodels
import albumentations as A
import pytorch_lightning as pl
from torch import nn
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch.encoders import get_preprocessing_params


def read_png(path):
    img = imageio.imread(str(path))
    return img

def get_preprocessing2(size, model_name, num_channels=3):
    if num_channels == 1:
        try:
            params = get_preprocessing_params(model_name, pretrained='imagenet')
        except:
            params = get_preprocessing_params('resnet18', pretrained='imagenet')
        params['input_space'] = 'G'
        params['mean'] = [params['mean'][0]] * num_channels
        params['std'] = [params['std'][0]] * num_channels

        preprocess_fn = functools.partial(preprocess_input, **params)

    return A.Compose([
        A.Resize(size, size),
        A.Lambda(image=preprocess_fn),
        ToTensorV2(),
    ])


def preprocess_input(
    x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs
):

    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std
    return x


class BaseEncoder(nn.Module):
    def __init__(self):
        super(BaseEncoder, self).__init__()

    def forward(self, x):
        x0 = self.init_block(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x0, x1, x2, x3, x4]

class ResNetEncoder(BaseEncoder):
    def __init__(self, model):
        super(ResNetEncoder, self).__init__()
        self.init_block = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4


def get_resnet(model_name):
    base_model = pretrainedmodels.__dict__[model_name]()
    model = ResNetEncoder(base_model)
    return model


def get_backbone(model_name, num_channels=3):
    model = get_resnet(model_name)

    if num_channels == 1:
        named_conv = next(model.named_parameters())
        name = named_conv[0].split('.')[:-1]
        w = named_conv[1]                           # (out_filters, 3, k, k)

        nn_module = model
        for n in name[:-1]:
            nn_module = getattr(nn_module, n)
        first_conv = next(nn_module.children())

        setattr(
            nn_module,
            name[-1],
            nn.Conv2d(
                1, w.shape[0], kernel_size=w.shape[2:],
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias,
                dilation=first_conv.dilation,
                groups=first_conv.groups,
            ),
        )
        w = torch.sum(w, dim=1, keepdim=True)
        getattr(nn_module, name[-1]).weight = nn.Parameter(w)
        
    return model

class MainModel(nn.Module):
    def __init__(self, model_name, n_classes=2, num_channels=3):
        super(MainModel, self).__init__()
        self.backbone = get_backbone(model_name, num_channels)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.out_dim = self.backbone(torch.zeros((1, num_channels, 128, 128)))[-1].shape[1]
        self.head = nn.Linear(self.out_dim, n_classes)
    
    def forward(self, x):
        [x0, x1, x2, x3, x4] = self.backbone(x)
        x = self.pooling(x4)
        x = x.reshape(-1, self.out_dim)
        x = self.head(x)
        
        return x
    
def get_model(cfg):
    return MainModel(cfg.model.name, n_classes=cfg.model.n_classes, num_channels=cfg.model.num_channels)

class ModelClassification(pl.LightningModule):
    def __init__(self, experiment):
        super(ModelClassification, self).__init__()
        self.orig_cfg, self.cfg = (copy.deepcopy(experiment['cfg']) for _ in range(2))
        self.args = experiment['args']
        self.hparams = dict(args=self.args, cfg=self.cfg)

        self.model = get_model(self.cfg)