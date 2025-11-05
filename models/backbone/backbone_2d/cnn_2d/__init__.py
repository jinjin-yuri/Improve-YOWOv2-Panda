# import 2D backbone
from nets.nn import build_yolov11
from .yolo_free.yolo_free import build_yolo_free
from ..yolov8 import build_yolov8


def build_2d_cnn(cfg, pretrained=False):
    print('==============================')
    print('2D Backbone: {}'.format(cfg['backbone_2d'].upper()))
    print('--pretrained: {}'.format(pretrained))

    if cfg['backbone_2d'] in ['yolo_free_nano', 'yolo_free_tiny',
                              'yolo_free_large', 'yolo_free_huge']:
        model, feat_dims = build_yolo_free(cfg['backbone_2d'], pretrained)

    # 在这里新增backbone
    elif cfg['backbone_2d'] == 'yolov8':
        model, feat_dims = build_yolov8(pretrained)
    elif cfg['backbone_2d'] == 'yolov11':
        model, feat_dims = build_yolov11(pretrained)


    else:
        print('Unknown 2D Backbone ...')
        exit()

    return model, feat_dims
