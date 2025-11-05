import typing as t
import torch
import torch.nn as nn


class dongtai_2d(nn.Module):
    def __init__(
            self,
            dim: int,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            group_kernel_sizes_m: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            gate_layer: str = 'sigmoid',
    ):
        super(dongtai_2d, self).__init__()  # 调用 nn.Module 的构造函数
        self.dim = dim  # 特征维度
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


class jingtai_2d(nn.Module):
    def __init__(
            self,
            dim: int,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            group_kernel_sizes_m: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            gate_layer: str = 'sigmoid',
    ):
        super(jingtai_2d, self).__init__()  # 调用 nn.Module 的构造函数
        self.dim = dim  # 特征维度
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
        # 定义加权求和的权重参数
        self.alpha = nn.Parameter(torch.ones(4))  # 4 个注意力权重
        self.softmax = nn.Softmax(dim=0)  # 用于归一化权重

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

        # 加权求和
        alpha_normalized = self.softmax(self.alpha)  # 归一化权重
        fused_attention = (
                alpha_normalized[0] * x_h_attn +
                alpha_normalized[1] * x_w_attn +
                alpha_normalized[2] * x_h_attn_m +
                alpha_normalized[3] * x_w_attn_m
        )

        # 应用融合后的注意力权重
        x = x * fused_attention
        return x


class two_2d(nn.Module):
    def __init__(
            self,
            dim: int,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            group_kernel_sizes_m: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            gate_layer: str = 'sigmoid',
    ):
        super(two_2d, self).__init__()  # 调用 nn.Module 的构造函数
        self.dim = dim  # 特征维度
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

        x = x * x_h_attn * x_w_attn * x_h_attn_m * x_w_attn_m
        return x



class origin_2d(nn.Module):
    def __init__(
            self,
            dim: int,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            gate_layer: str = 'sigmoid',
    ):
        super(origin_2d, self).__init__()  # 调用 nn.Module 的构造函数
        self.dim = dim  # 特征维度
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
        # 注意力门控层
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)  # 水平方向的归一化
        self.norm_w = nn.GroupNorm(4, dim)  # 垂直方向的归一化
        print('o2d')

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

        x = x * x_h_attn * x_w_attn
        return x