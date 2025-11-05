import numpy as np
import torch
import torch.nn as nn

from models.backbone.backbone_2d.cnn_2d.yolo_free.yolo_free_fpn import build_fpn
from models.backbone.backbone_2d.cnn_2d.yolo_free.yolo_free_head import build_head
from models.backbone.backbone_2d.cnn_2d.yolo_free.yolo_free_neck import build_neck

# from models.backbone.backbone_2d.CPCA import CPCA
# from models.backbone.backbone_2d.Biformer import BiLevelRoutingAttention_nchw
from models.backbone.backbone_2d.CBAM import CBAM

yolov8_config = {
    'yolo_free_nano': {
        # model
        'backbone': 'shufflenetv2_1.0x',
        'pretrained': True,
        'stride': [8, 16, 32],  # P3, P4, P5
        'anchor_size': None,
        # neck
        'neck': 'sppf',
        'neck_dim': 232,
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'lrelu',
        'neck_norm': 'BN',
        'neck_depthwise': True,
        # fpn
        'fpn': 'pafpn_elan',
        'fpn_size': 'nano',
        'fpn_dim': [116, 232, 232],
        'fpn_norm': 'BN',
        'fpn_act': 'lrelu',
        'fpn_depthwise': True,
        # head
        'head': 'decoupled_head',
        'head_dim': 64,
        'head_norm': 'BN',
        'head_act': 'lrelu',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': True,
    },

    'yolo_free_tiny': {
        # model
        'backbone': 'elannet_tiny',
        'pretrained': True,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'spp_block_csp',
        'neck_dim': 256,
        'expand_ratio': 0.5,
        'pooling_size': [5, 9, 13],
        'neck_act': 'lrelu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'pafpn_elan',
        'fpn_size': 'tiny',  # 'tiny', 'large', 'huge
        'fpn_dim': [128, 256, 256],
        'fpn_norm': 'BN',
        'fpn_act': 'lrelu',
        'fpn_depthwise': False,
        # head
        'head': 'decoupled_head',
        'head_dim': 64,
        'head_norm': 'BN',
        'head_act': 'lrelu',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
    },

    'yolov8_large': {
        # model
        'backbone': 'yolov8',
        # 'backbone': 'elannet_large_c',
        'pretrained': True,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'spp_block_csp',
        'neck_dim': 512,
        'expand_ratio': 0.5,
        'pooling_size': [5, 9, 13],
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'pafpn_elan',
        'fpn_size': 'large',  # 'tiny', 'large', 'huge
        'fpn_dim': [512, 1024, 512],
        'fpn_norm': 'BN',
        'fpn_act': 'silu',
        'fpn_depthwise': False,
        # head
        'head': 'decoupled_head',
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'silu',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
    },

}



def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def autopad(k, p=None, d=1):
    # kernel, padding, dilation
    # 对输入的特征层进行自动padding，按照Same原则
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class SiLU(nn.Module):
    # SiLU激活函数
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Conv(nn.Module):
    # 标准卷积+标准化+激活函数
    default_act = SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # 标准瓶颈结构，残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    # CSPNet结构，大残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # 进行一个卷积，然后划分成两份，每个通道都为c
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # 每进行一次残差结构都保留，然后堆叠在一起，密集残差
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    # SPP结构，5、9、13最大池化核的最大池化。
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Backbone(nn.Module):
    def __init__(self, base_channels, base_depth, deep_mul, phi, pretrained=False):
        super().__init__()
        # -----------------------------------------------#
        #   输入图片是3, 640, 640
        # -----------------------------------------------#
        # 3, 640, 640 => 32, 640, 640 => 64, 320, 320
        self.stem = Conv(3, base_channels, 3, 2)

        # 64, 320, 320 => 128, 160, 160 => 128, 160, 160
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            C2f(base_channels * 2, base_channels * 2, base_depth, True),
            # CBAM(base_channels * 2)
        )
        # 128, 160, 160 => 256, 80, 80 => 256, 80, 80
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
            # CBAM(base_channels * 4)
        )
        # 256, 80, 80 => 512, 40, 40 => 512, 40, 40
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
            # CBAM(base_channels * 8)
        )
        # 512, 40, 40 => 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 20, 20
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
            C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5),
            # CBAM(int(base_channels * 16 * deep_mul))
        )

        if pretrained:
            url = {
                "n": 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
                "s": 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
                "m": 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
                "l": 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
                "x": 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from " + url.split('/')[-1])

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        # -----------------------------------------------#
        #   dark3的输出为256, 80, 80，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        # -----------------------------------------------#
        #   dark4的输出为512, 40, 40，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        # -----------------------------------------------#
        #   dark5的输出为1024 * deep_mul, 20, 20，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3


def fuse_conv_and_bn(conv, bn):
    # 混合Conv2d + BatchNorm2d 减少计算量
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # 准备kernel
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # 准备bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


class DFL(nn.Module):
    # DFL模块
    # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        # bs, self.reg_max * 4, 8400
        b, c, a = x.shape
        # bs, 4, self.reg_max, 8400 => bs, self.reg_max, 4, 8400 => b, 4, 8400
        # 以softmax的方式，对0~16的数字计算百分比，获得最终数字。
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


# ---------------------------------------------------#
#   yolo_body
# ---------------------------------------------------#
class YoloBody(nn.Module):
    # def __init__(self, input_shape, num_classes, phi, pretrained=False):
    def __init__(self, phi='l', pretrained=False):
        super(YoloBody, self).__init__()
        depth_dict = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.00, }
        width_dict = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        deep_width_dict = {'n': 1.00, 's': 1.00, 'm': 0.75, 'l': 0.50, 'x': 0.50, }
        dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3
        # -----------------------------------------------#
        #   输入图片是3, 640, 640
        # -----------------------------------------------#

        # ---------------------------------------------------#
        #   生成主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   256, 80, 80
        #   512, 40, 40
        #   1024 * deep_mul, 20, 20
        # ---------------------------------------------------#
        self.backbone = Backbone(base_channels, base_depth, deep_mul, phi, pretrained=pretrained)

        # ------------------------加强特征提取网络------------------------#
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # 1024 * deep_mul + 512, 40, 40 => 512, 40, 40
        self.conv3_for_upsample1 = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, base_channels * 8,
                                       base_depth, shortcut=False)
        # 768, 80, 80 => 256, 80, 80
        self.conv3_for_upsample2 = C2f(base_channels * 8 + base_channels * 4, base_channels * 4, base_depth,
                                       shortcut=False)

        # 256, 80, 80 => 256, 40, 40
        self.down_sample1 = Conv(base_channels * 4, base_channels * 4, 3, 2)
        # 512 + 256, 40, 40 => 512, 40, 40
        self.conv3_for_downsample1 = C2f(base_channels * 8 + base_channels * 4, base_channels * 8, base_depth,
                                         shortcut=False)

        # 512, 40, 40 => 512, 20, 20
        self.down_sample2 = Conv(base_channels * 8, base_channels * 8, 3, 2)
        # 1024 * deep_mul + 512, 20, 20 =>  1024 * deep_mul, 20, 20
        self.conv3_for_downsample2 = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8,
                                         int(base_channels * 16 * deep_mul), base_depth, shortcut=False)
        # ------------------------加强特征提取网络------------------------#

    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        return self

    def forward(self, x):
        #  backbone
        feat1, feat2, feat3 = self.backbone.forward(x)

        # ------------------------加强特征提取网络------------------------#
        # 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 40, 40
        P5_upsample = self.upsample(feat3)
        # 1024 * deep_mul, 40, 40 cat 512, 40, 40 => 1024 * deep_mul + 512, 40, 40
        P4 = torch.cat([P5_upsample, feat2], 1)
        # 1024 * deep_mul + 512, 40, 40 => 512, 40, 40
        P4 = self.conv3_for_upsample1(P4)

        # 512, 40, 40 => 512, 80, 80
        P4_upsample = self.upsample(P4)

        # x1 = P4_upsample # [1,512,28,28]

        # 512, 80, 80 cat 256, 80, 80 => 768, 80, 80
        P3 = torch.cat([P4_upsample, feat1], 1)
        # 768, 80, 80 => 256, 80, 80
        P3 = self.conv3_for_upsample2(P3)

        # 256, 80, 80 => 256, 40, 40
        P3_downsample = self.down_sample1(P3)
        # 512, 40, 40 cat 256, 40, 40 => 768, 40, 40
        P4 = torch.cat([P3_downsample, P4], 1)
        # 768, 40, 40 => 512, 40, 40
        P4 = self.conv3_for_downsample1(P4)

        # 512, 40, 40 => 512, 20, 20
        P4_downsample = self.down_sample2(P4)
        # 512, 20, 20 cat 1024 * deep_mul, 20, 20 => 1024 * deep_mul + 512, 20, 20
        P5 = torch.cat([P4_downsample, feat3], 1)
        # 1024 * deep_mul + 512, 20, 20 => 1024 * deep_mul, 20, 20
        P5 = self.conv3_for_downsample2(P5)

        x3 = torch.cat([P3, P3], dim=1)
        x4 = torch.cat([P4, P4], dim=1)
        x5 = torch.cat([P5, P5], dim=1)

        # ------------------------加强特征提取网络------------------------#
        outputs = {
            'layer2': x3,  # [1,512,28,28]
            'layer3': x4,  # [1,1024,14,14]
            'layer4': x5  # [1,1024,7,7]
        }
        return outputs


class YOLOv8(nn.Module):
    def __init__(self, cfg):
        super(YOLOv8, self).__init__()
        # --------- Basic Config -----------
        self.cfg = cfg

        # --------- Network Parameters ----------
        ## backbone
        self.backbone = YoloBody()
        bk_dim = [512, 1024, 1024]

        ## neck
        self.neck = build_neck(cfg=self.cfg, in_dim=bk_dim[-1], out_dim=self.cfg['neck_dim'])

        ## fpn
        self.fpn = build_fpn(cfg=self.cfg, in_dims=self.cfg['fpn_dim'], out_dim=self.cfg['head_dim'])

        ## non-shared heads
        self.non_shared_heads = nn.ModuleList(
            [build_head(cfg)
             for _ in range(len(cfg['stride']))
             ])

    def forward(self, x):
        # backbone
        feats = self.backbone(x)

        # neck
        feats['layer4'] = self.neck(feats['layer4'])

        # fpn
        pyramid_feats = [feats['layer2'], feats['layer3'], feats['layer4']]
        pyramid_feats = self.fpn(pyramid_feats)

        # non-shared heads
        all_cls_feats = []
        all_reg_feats = []
        for feat, head in zip(pyramid_feats, self.non_shared_heads):
            # [B, C, H, W]
            cls_feat, reg_feat = head(feat)

            all_cls_feats.append(cls_feat)
            all_reg_feats.append(reg_feat)

        return all_cls_feats, all_reg_feats


def build_yolov8(pretrained, model_name='yolov8_large'):
    # model config
    cfg = yolov8_config[model_name]

    model = YOLOv8(cfg)
    feat_dims = [model.cfg['head_dim']] * 3

    # Load COCO pretrained weight
    if pretrained:
        print('Loading pretrained weight ...')
        path = 'C:/jiang/YOWOv2/models/backbone/backbone_2d/yolov8_l.pth'

        import torch
        import numpy as np
        pretrained_dict = torch.load(path)

        model_dict = model.state_dict()

        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():

            k1 = 'backbone.' + k

            if k1 in model_dict.keys() and np.shape(model_dict[k1]) == np.shape(v):
                temp_dict[k1] = v
                load_key.append(k1)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict,strict=False)
        # ------------------------------------------------------#
        #   显示没有匹配上的Key
        # ------------------------------------------------------#

        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    return model, feat_dims


if __name__ == '__main__':
    model, fpn_dim = build_yolov8(pretrained=True)
    # model.eval()

    x = torch.randn(2, 3, 64, 64)
    feats = model(x)
    cls_feats, reg_feats = model(x)

    for cls_feat, reg_feat in zip(cls_feats, reg_feats):
        print(cls_feat.shape, reg_feat.shape)
