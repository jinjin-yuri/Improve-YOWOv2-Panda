import torch
import torch.nn as nn
import typing as t
from einops import rearrange

from models.basic import Conv2d
from models.basic.DPSCSA import DPSCSA
from models.basic.FRFN import FRFN
from models.basic.pinyuself import FFTNetBlock, pinyu
from models.yowo.scsa import SCSA


# Channel Self Attetion Module
class CSAM(nn.Module):
    """ Channel attention module """

    def __init__(self):
        super(CSAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        print('CSAM CSAM')

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B x C x H x W )
            returns :
                out : attention value + input feature
                attention: B x C x C
        """
        B, C, H, W = x.size()
        # query / key / value
        query = x.view(B, C, -1)
        key = x.view(B, C, -1).permute(0, 2, 1)
        value = x.view(B, C, -1)

        # attention matrix
        energy = torch.bmm(query, key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        # attention
        out = torch.bmm(attention, value)
        out = out.view(B, C, H, W)

        # output
        out = self.gamma * out + x

        return out



class ChannelEncoder_orign(nn.Module):
    def __init__(self, in_dim, out_dim, act_type='', norm_type=''):
        super().__init__()
        self.fuse_convs = nn.Sequential(
            Conv2d(in_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            # CSAM(),
            SCSA(out_dim, out_dim/4),
            Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            nn.Dropout(0.1, inplace=False),
            nn.Conv2d(out_dim, out_dim, kernel_size=1)
        )

    def forward(self, x1, x2):
        """
            x: [B, C, H, W]
        """
        x = torch.cat([x1, x2], dim=1)
        # [B, CN, H, W] -> [B, C, H, W]
        x = self.fuse_convs(x)

        return x


# Channel Encoder
class ChannelEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, dim2, dim3, act_type='', norm_type=''):
        super().__init__()
        # 确保 dim3 能被 4 整除
        assert dim3 % 4 == 0, "dim3 must be divisible by 4"
        self.att2d = att_2d(dim2)
        self.att3d = att_3d(dim3, dim3 // 4)  # 使用整数除法
        # 确保 in_dim 等于 dim2 + dim3
        assert in_dim == dim2 + dim3, "in_dim must be equal to dim2 + dim3"
        self.block = nn.Sequential(
            Conv2d(in_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            # CSAM(),
            pinyu(dim=out_dim),
            Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            nn.Dropout(0.1, inplace=False),
            nn.Conv2d(out_dim, out_dim, kernel_size=1)
        )

    def forward(self, x1, x2):
        """
        参数:
            x1: [B, dim2, H, W]  # 2D 特征图
            x2: [B, dim3, H, W]  # 3D 特征图
        返回:
            x: [B, out_dim, H, W]  # 输出特征图
        """
        # # 应用注意力机制
        # x1 = self.att2d(x1)  # 确保 x1 的形状仍然是 [B, dim2, H, W]
        # x2 = self.att3d(x2)  # 确保 x2 的形状仍然是 [B, dim3, H, W]

        # 检查 x1 和 x2 的空间维度是否一致
        if x1.shape[2:] != x2.shape[2:]:
            raise ValueError("x1 and x2 must have the same spatial dimensions (H, W)")

        # 沿通道维度拼接
        x = torch.cat([x1, x2], dim=1)

        # 通过 CFAMBlock 处理
        x = self.block(x)

        return x


def build_channel_encoder(cfg, in_dim, out_dim, dim2, dim3):
    encoder = ChannelEncoder(
        in_dim=in_dim,
        out_dim=out_dim,
        dim2=dim2,
        dim3=dim3,
        act_type=cfg['head_act'],
        norm_type=cfg['head_norm']
    )

    return encoder

def build_orign_encoder(cfg, in_dim, out_dim):
    encoder = ChannelEncoder_orign(
            in_dim=in_dim,
            out_dim=out_dim,
            act_type=cfg['head_act'],
            norm_type=cfg['head_norm']
        )

    return encoder

class att_2d(nn.Module):
    def __init__(
            self,
            dim: int,
            window_size: int = 3,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            group_kernel_sizes_m: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            gate_layer: str = 'sigmoid',
    ):
        super(att_2d, self).__init__()  # 调用 nn.Module 的构造函数
        self.dim = dim  # 特征维度
        self.window_size = window_size  # 窗口大小
        self.qkv_bias = qkv_bias  # 是否使用偏置
        self.fuse_bn = fuse_bn  # 是否融合批归一化

        assert self.dim % 4 == 0, 'The dimension of input feature should be divisible by 4.'  # 确保维度可被4整除
        self.group_chans = group_chans = self.dim // 4  # 分组通道数

        # 定义局部和全局深度卷积层
        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)
        # 定义局部和全局深度卷积层
        self.local_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes_m[0],
                                   padding=group_kernel_sizes_m[0] // 2, groups=group_chans)
        self.global_dwc_s_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes_m[1],
                                      padding=group_kernel_sizes_m[1] // 2, groups=group_chans)
        self.global_dwc_m_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes_m[2],
                                      padding=group_kernel_sizes_m[2] // 2, groups=group_chans)
        self.global_dwc_l_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes_m[3],
                                      padding=group_kernel_sizes_m[3] // 2, groups=group_chans)

        # 注意力门控层
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)  # 水平方向的归一化
        self.norm_w = nn.GroupNorm(4, dim)  # 垂直方向的归一化
        # 动态权重生成器
        # 输入维度为 dim * 2，因为拼接了全局和局部特征
        self.dynamic_alpha = nn.Sequential(
            nn.Linear(dim * 2, 4),  # 输入特征维度 -> 4 个权重
            nn.Softmax(dim=1)  # 归一化
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入张量 x 的维度为 (B, C, H, W)
        """
        # 计算空间注意力优先级
        b, c, h_, w_ = x.size()  # 获取输入的形状
        # (B, C, H)
        x_h = x.mean(dim=3)  # 沿着宽度维度求平均
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)  # 拆分通道
        # (B, C, W)
        x_w = x.mean(dim=2)  # 沿着高度维度求平均
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)  # 拆分通道

        # 计算水平注意力
        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)  # 调整形状

        # 计算垂直注意力
        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)  # 调整形状

        x_h_m = x.max(dim=3).values  # 沿着宽度维度求最大池化
        l_x_h_m, g_x_h_s_m, g_x_h_m_m, g_x_h_l_m = torch.split(x_h_m, self.group_chans, dim=1)  # 拆分通道
        # (B, C, W)
        x_w_m = x.max(dim=2).values  # 沿着高度维度求最大池化
        l_x_w_m, g_x_w_s_m, g_x_w_m_m, g_x_w_l_m = torch.split(x_w_m, self.group_chans, dim=1)  # 拆分通道

        # 计算水平注意力
        x_h_attn_m = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc_m(l_x_h_m),
            self.global_dwc_s_m(g_x_h_s_m),
            self.global_dwc_m_m(g_x_h_m_m),
            self.global_dwc_l_m(g_x_h_l_m),
        ), dim=1)))
        x_h_attn_m = x_h_attn_m.view(b, c, h_, 1)  # 调整形状

        # 计算垂直注意力
        x_w_attn_m = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc_m(l_x_w_m),
            self.global_dwc_s_m(g_x_w_s_m),
            self.global_dwc_m_m(g_x_w_m_m),
            self.global_dwc_l_m(g_x_w_l_m)
        ), dim=1)))
        x_w_attn_m = x_w_attn_m.view(b, c, 1, w_)  # 调整形状

        # 提取多尺度特征
        global_feature = x.mean(dim=(2, 3))  # 全局特征 (B, C)
        local_feature = x.max(dim=3).values.mean(dim=2)  # 局部特征 (B, C)

        # 拼接多尺度特征
        combined_feature = torch.cat([global_feature, local_feature], dim=1)  # (B, C * 2)

        # 动态生成权重
        alpha_normalized = self.dynamic_alpha(combined_feature)  # (B, 4)

        # 加权求和
        fused_attention = (
                alpha_normalized[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x_h_attn +
                alpha_normalized[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x_w_attn +
                alpha_normalized[:, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x_h_attn_m +
                alpha_normalized[:, 3].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x_w_attn_m
        )

        # 应用融合后的注意力权重
        x = x * fused_attention
        return x

class att_3d(nn.Module):
    def __init__(
            self,
            dim: int,
            head_num: int,
            window_size: int = 3,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            down_sample_mode: str = 'avg_pool',
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'sigmoid',
    ):
        super(att_3d, self).__init__()  # 调用 nn.Module 的构造函数
        self.dim = dim  # 特征维度
        self.head_num = head_num  # 注意力头数
        self.head_dim = dim // head_num  # 每个头的维度
        self.scaler = self.head_dim ** -0.5  # 缩放因子
        self.group_kernel_sizes = group_kernel_sizes  # 分组卷积核大小
        self.window_size = window_size  # 窗口大小
        self.qkv_bias = qkv_bias  # 是否使用偏置
        self.fuse_bn = fuse_bn  # 是否融合批归一化
        self.down_sample_mode = down_sample_mode  # 下采样模式

        assert self.dim % 4 == 0, 'The dimension of input feature should be divisible by 4.'  # 确保维度可被4整除
        self.group_chans = group_chans = self.dim // 4  # 分组通道数

        # 定义局部和全局深度卷积层
        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)

        # 注意力门控层
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)  # 水平方向的归一化
        self.norm_w = nn.GroupNorm(4, dim)  # 垂直方向的归一化

        self.conv_d = nn.Identity()  # 直接连接
        self.norm = nn.GroupNorm(1, dim)  # 通道归一化
        # 定义查询、键和值的卷积层
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)  # 注意力丢弃层
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()  # 通道注意力门控

        # 根据窗口大小和下采样模式选择下采样函数
        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.space_to_chans  # 重组合下采样
                # 维度降低
                self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)  # 平均池化
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)  # 最大池化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 基于自注意力的通道注意力
        # 减少计算量
        y = self.down_func(x)
        y = self.conv_d(y)  # 维度转换
        _, _, h_, w_ = y.size()  # 获取形状
        # 先归一化，然后重塑 -> (B, H, W, C) -> (B, C, H * W)，并生成 q, k 和 v
        y = self.norm(y)  # 归一化
        q = self.q(y)  # 计算查询
        k = self.k(y)  # 计算键
        v = self.v(y)  # 计算值
        # (B, C, H, W) -> (B, head_num, head_dim, N)
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))

        # 计算注意力
        attn = q @ k.transpose(-2, -1) * self.scaler  # 点积注意力计算
        attn = self.attn_drop(attn.softmax(dim=-1))  # 应用注意力丢弃
        # (B, head_num, head_dim, N)
        attn = attn @ v  # 加权值
        # (B, C, H_, W_)
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
        # (B, C, 1, 1)
        attn = attn.mean((2, 3), keepdim=True)  # 求平均
        attn = self.ca_gate(attn)  # 应用通道注意力门控
        return attn * x  # 返回加权后的输入

class CFAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CFAMBlock, self).__init__()
        inter_channels = out_channels
        self.conv_bn_relu1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU()
        )
        self.conv_bn_relu2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU()
        )
        # self.c_dropout = ChannelDropout(drop_ratio=0.2, channels=inter_channels)
        self.conv_bn_relu3 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU()
        )
        self.conv_out = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channels, out_channels, 1)
        )

    def forward(self, x):
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        output = self.conv_out(x)
        return output


class MYAMFusion(nn.Module):
    def __init__(self, channels_2D, channels_3D, interchannels, mode='decoupled'):
        super(MYAMFusion, self).__init__()
        assert mode in ['coupled', 'decoupled'], "wrong mode in CFAMFusion"
        self.mode = mode

        # 2D 和 3D 特征的注意力模块
        if mode == 'coupled':
            # channels_2D 是一个整数列表，例如 [64, 128]
            self.attention_2D = nn.ModuleList([att_2d(dim=dim) for dim in channels_2D])
        elif mode == 'decoupled':
            # channels_2D 是一个包含元组的列表，例如 [(64, 32), (128, 64)]
            self.attention_2D = nn.ModuleList([att_2d(dim=dim[0]) for dim in channels_2D])

        self.attention_3D = att_3d(channels_3D, channels_3D / 4)

        if mode == 'coupled':
            layers = []
            for channels2D in channels_2D:
                layers.append(CFAMBlock(channels2D + channels_3D, interchannels))
            self.fusion = nn.ModuleList(layers)
        elif mode == 'decoupled':
            box = []
            cls = []
            for channels2D in channels_2D:
                box.append(CFAMBlock(channels2D[0] + channels_3D, interchannels))
                cls.append(CFAMBlock(channels2D[1] + channels_3D, interchannels))
            self.box = nn.ModuleList(box)
            self.cls = nn.ModuleList(cls)

    def forward(self, ft_2D, ft_3D):
        # 对 2D 和 3D 特征分别应用注意力模块
        if self.mode == 'coupled':
            ft_2D = [self.attention_2D[idx](ft) for idx, ft in enumerate(ft_2D)]
        elif self.mode == 'decoupled':
            ft_2D = [(self.attention_2D[idx](ft[0]), ft[1]) for idx, ft in enumerate(ft_2D)]

        ft_3D = self.attention_3D(ft_3D)

        _, C_3D, H_3D, W_3D = ft_3D.shape
        fts = []

        if self.mode == 'coupled':
            for idx, ft2D in enumerate(ft_2D):
                _, C_2D, H_2D, W_2D = ft2D.shape
                assert H_2D / H_3D == W_2D / W_3D, "can't upscale"

                upsampling = nn.Upsample(scale_factor=H_2D / H_3D)
                ft_3D_t = upsampling(ft_3D)
                ft = torch.cat((ft2D, ft_3D_t), dim=1)
                fts.append(self.fusion[idx](ft))

        elif self.mode == 'decoupled':
            for idx, ft2D in enumerate(ft_2D):
                _, C_2D, H_2D, W_2D = ft2D[0].shape
                assert H_2D / H_3D == W_2D / W_3D, "can't upscale"

                upsampling = nn.Upsample(scale_factor=H_2D / H_3D)
                ft_3D_t = upsampling(ft_3D)
                ft_box = torch.cat((ft2D[0], ft_3D_t), dim=1)
                ft_cls = torch.cat((ft2D[1], ft_3D_t), dim=1)
                fts.append([self.box[idx](ft_box), self.cls[idx](ft_cls)])

        return fts


if __name__ == '__main__':
    # attention = SHSA()
    # x = torch.randn(2, 3, 64, 64)
    # y = attention(x)
    # print("SAM输入:", x.size())
    #print("SAM输出:", y.size())

    block = ChannelEncoder(2304, 256,256,2048)
    input1 = torch.randn(2, 256, 64, 64)
    input2 = torch.randn(2, 2048, 64, 64)
    output = block(input1, input2)
    print("ChannelEncoder输入:", input1.size(), input2.size())
    print("ChannelEncoder输出:", output.size())
