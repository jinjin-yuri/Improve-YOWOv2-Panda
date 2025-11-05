import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.hub import load_state_dict_from_url
from functools import partial
from torch.nn.parameter import Parameter
import os
from collections import OrderedDict
# from models.backbone.backbone_3d.SimAM_3D import Simam_module3D
# from models.backbone.backbone_3d.ECA_3D import ECAAttention_3D
# from models.backbone.backbone_3d.SEAttention_3D import SEAttention_3D
# from models.backbone.backbone_3d.SA_3D import SA_3D
from models.backbone.backbone_3d.SGE_3D import SpatialGroupEnhance_3D

__all__ = ['resnext50', 'resnext101', 'resnet152']


model_urls = {
    "resnext50": "https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/resnext-50-kinetics.pth",
    "resnext101": "https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/resnext-101-kinetics.pth",
    "resnext152": "https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/resnext-152-kinetics.pth"
}



def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()

    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    zero_pads = zero_pads.to(out.data.device)
    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        # self.sge = SpatialGroupEnhance_3D()



    def forward(self, x):
        residual = x # x [1,64,8,56,56]

        out = self.conv1(x) #[1,128,8,56,56]
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) #[1,128,8,56,56]
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out) #[1,256,8,56,56]
        out = self.bn3(out) #[1,256,8,56,56]

        # out = self.sge(out)#liu

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B',
                 cardinality=32):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)
        
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)
        
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c2 = self.maxpool(c1)

        c2 = self.layer1(c2)

        c3 = self.layer2(c2)

        c4 = self.layer3(c3)

        c5 = self.layer4(c4)

        if c5.size(2) > 1:
            c5 = torch.mean(c5, dim=2, keepdim=True)
        
        return c5.squeeze(2)


def load_weight(model, arch):
    print('Loading pretrained weight ...')

    url = model_urls[arch]
    # check
    if url is None:
        print('No pretrained weight for 3D CNN: {}'.format(arch.upper()))
        return model
        
    print('Loading 3D backbone pretrained weight: {}'.format(arch.upper()))
    # checkpoint state dict
    checkpoint = load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
    checkpoint_state_dict = checkpoint.pop('state_dict')

    # model state dict
    model_state_dict = model.state_dict()
    # reformat checkpoint_state_dict:
    new_state_dict = {}
    for k in checkpoint_state_dict.keys():
        v = checkpoint_state_dict[k]
        new_state_dict[k[7:]] = v

    # check
    for k in list(new_state_dict.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(new_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                new_state_dict.pop(k)
                # print(k)
        else:
            new_state_dict.pop(k)
            # print(k)

    model.load_state_dict(new_state_dict, False) # 增加了False
    # model.load_state_dict(new_state_dict) # 原代码
        
    return model


def resnext50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        model = load_weight(model, 'resnext50')

    return model


def resnext101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)

    if pretrained:
        model = load_weight(model, 'resnext101')

    return model


def resnext152(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)

    if pretrained:
        model = load_weight(model, 'resnext152')

    return model

def sa_resnet101(pretrained=False):
    model = _sanet(SABottleneck, [3, 4, 23, 3], pretrained=pretrained)

    return model

def _sanet(block, layers, pretrained, **kwargs):
    model = ResNet(block, layers, **kwargs)

    if pretrained:
        state_dict = load_state_dict('/home/dyp/lx/YOWOv2-master/models/backbone/backbone_3d/cnn_3d/sa_resnet101.pth.tar',model)
        model.load_state_dict(state_dict)

    return model

def load_state_dict(checkpoint_path, model):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'

        if state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint

        # # Reshape all parameters
        # for key in state_dict:
            # if state_dict[key].numel() > 1:
                # Only add a singleton dimension if it's not a scalar
                # state_dict[key] = state_dict[key].unsqueeze(-1)
        
        print("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        print("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.

    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=64):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # Modified for 3D input
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, t, h, w = x.shape

        x = x.reshape(b, groups, -1, t, h, w)
        x = x.permute(0, 2, 1, 3, 4, 5)

        # flatten
        x = x.reshape(b, -1, t, h, w)

        return x

    def forward(self, x):
        b, c, t, h, w = x.shape

        x = x.reshape(b * self.groups, -1, t, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, t, h, w)

        out = self.channel_shuffle(out, 2)
        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SABottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(SABottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sa = sa_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.sa(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
            self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv3d(3, self.inplanes, kernel_size=7, stride=(1, 2, 2),
                               padding=(3, 3, 3), bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, SABottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c2 = self.maxpool(c1)

        c2 = self.layer1(c2)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        if c5.size(2) > 1:
            c5 = torch.mean(c5, dim=2, keepdim=True)

        return c5.squeeze(2)


# build 3D resnet
def build_resnext_3d(model_name='resnext101', pretrained=True):
    if model_name == 'resnext50':
        model = resnext50(pretrained=pretrained)
        feats = 2048

    elif model_name == 'resnext101':
        model = resnext101(pretrained=pretrained)
        feats = 2048

    elif model_name == 'resnext152':
        model = resnext152(pretrained=pretrained)
        feats = 2048
    
    elif model_name == 'sa_resnet101':
        model = sa_resnet101(pretrained=pretrained)
        feats = 2048

    return model, feats


if __name__ == '__main__':
    import time
    from thop import profile

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    model, feats = build_resnext_3d(model_name='resnext101', pretrained=False)
    # model, feats = build_resnext_3d(model_name='sa_resnet101', pretrained=True)
    model = model.to(device)

    x = torch.randn(1, 3, 32, 256, 256).to(device)
    # star time
    t0 = time.time()
    # inference
    outs = model(x)
    print('输出：', outs.size())
    for y in outs:
        print(y.shape)
    # end time
    print('Inference time: {}'.format(time.time() - t0))

    # FLOPs & Params
    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9))
    print('Params : {:.2f} M'.format(params / 1e6))
