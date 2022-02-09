import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from models.sync_batchnorm import SynchronizedBatchNorm3d
from utils import get_model_complexity_info

# adapt from https://github.com/MIC-DKFZ/BraTS2017


def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(8, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    elif norm == 'sync_bn':
        m = SynchronizedBatchNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


def one_hot(ori, classes):

    batch, h, w, d = ori.size()
    new_gd = torch.zeros((batch, classes, h, w, d), dtype=ori.dtype).cuda()
    for j in range(classes):
        index_list = torch.nonzero((ori == j))
        # print('index_list:', index_list.shape)
        for i in range(len(index_list)):
            batch, height, width, depth = index_list[i]
            new_gd[batch, j, height, width, depth] = 1

    return new_gd.float()


def both(ori, attention):

    batch, channel, h, w, d = ori.size()
    ori_out = torch.zeros(size=(ori.size()), dtype=ori.dtype).cuda()
    intar_class = torch.zeros(size=(batch, channel, 4), dtype=ori.dtype).cuda()
    attention = torch.argmax(attention, dim=1)  # (B, H, W, D)
    attention = one_hot(attention, classes=4)  # (B, Cl, H, W, D)

    # intra class
    for catagory in range(attention.shape[1]):
        # print('catagory:', catagory)
        catagory_map = attention[:, catagory:catagory + 1, ...]  # (B, 1, H, W, D)
        ori_catagory = torch.einsum('bchwd, bfhwd -> bchwd', ori, catagory_map)

        sum_catagory = torch.sum(ori_catagory, dim=(2, 3, 4))  # (B, C)
        number_cata = torch.sum(catagory_map, dim=(2, 3, 4)) + 1e-5  # (B, 1)
        # number_cata = catagory_map.sum() + 1
        avg_catagory = sum_catagory / number_cata  # (B, C)
        intar_class[:, :, catagory] = avg_catagory               # (B, C, Cl)

        avg_catagory2 = torch.einsum('bc, bfhwd -> bchwd', avg_catagory, catagory_map)
        # print('avg_catagory2:', avg_catagory2.shape)
        ori_out = ori_out + avg_catagory2

    # inter class
    A = B = intar_class  # (B, C, Cl)
    BT = B.permute(0, 2, 1)  # (B, Cl, C)
    vecProd = torch.matmul(BT, A)  # (B, Cl, Cl)
    SqA = A ** 2
    sumSqA = torch.sum(SqA, dim=1).unsqueeze(dim=2).repeat(1, 1, 4)  # (B, Cl, Cl)
    Euclidean_dis = sumSqA * 2 - vecProd * 2
    Euclidean_dis = torch.sum(Euclidean_dis, dim=(1, 2))
    Euclidean_dis = torch.pow(Euclidean_dis, 1/2).sum()

    return ori_out, Euclidean_dis


class InitConv(nn.Module):
    def __init__(self, in_channels=4, out_channels=16, dropout=0.2):
        super(InitConv, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = dropout

    def forward(self, x):
        y = self.conv(x)
        y = F.dropout3d(y, self.dropout)

        return y


class EnBlock(nn.Module):
    def __init__(self, in_channels, norm='gn'):
        super(EnBlock, self).__init__()

        self.bn1 = normalization(in_channels, norm=norm)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

        self.bn2 = normalization(in_channels, norm=norm)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        y = self.bn2(x1)
        y = self.relu2(y)
        y = self.conv2(y)
        y = y + x

        return y


class EnDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnDown, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        y = self.conv(x)

        return y


class DeUp_Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp_Cat, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(out_channels*2, out_channels, kernel_size=1)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        # y = y + prev
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y


class DeUp_Plus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp_Plus, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        y = y + prev
        return y


class DeUp_Tri(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp_Tri, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.interpolation = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.interpolation(x1)
        y = y + prev
        return y


class DeBlock(nn.Module):
    def __init__(self, in_channels, norm='gn'):
        super(DeBlock, self).__init__()

        self.bn1 = normalization(in_channels, norm=norm)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

        self.bn2 = normalization(in_channels, norm=norm)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        y = self.bn2(x1)
        y = self.relu2(y)
        y = self.conv2(y)
        y = y + x

        return y


class EndConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EndConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        y = self.conv(x)
        return y


class Unet_Both(nn.Module):
    def __init__(self, in_channels=4, base_channels=16, num_classes=4):
        super(Unet_Both, self).__init__()

        self.InitConv = InitConv(in_channels=in_channels, out_channels=base_channels, dropout=0.2)
        self.EnBlock1 = EnBlock(in_channels=base_channels)
        self.EnDown1 = EnDown(in_channels=base_channels, out_channels=base_channels*2)

        self.EnBlock2_1 = EnBlock(in_channels=base_channels*2)
        self.EnBlock2_2 = EnBlock(in_channels=base_channels*2)
        self.EnDown2 = EnDown(in_channels=base_channels*2, out_channels=base_channels*4)

        self.EnBlock3_1 = EnBlock(in_channels=base_channels * 4)
        self.EnBlock3_2 = EnBlock(in_channels=base_channels * 4)
        self.EnDown3 = EnDown(in_channels=base_channels*4, out_channels=base_channels*8)

        self.EnBlock4_1 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_2 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_3 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_4 = EnBlock(in_channels=base_channels * 8)

        self.AttentionBlock = nn.Sequential(
            nn.Conv3d(4, 16, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm3d(32),
            # nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm3d(32),
            # nn.ReLU(inplace=True),
            nn.Conv3d(16, 4, kernel_size=3, stride=2, padding=1),
        )

        self.DeUp3 = DeUp_Cat(in_channels=base_channels*8, out_channels=base_channels*4)
        self.DeBlock3 = DeBlock(in_channels=base_channels*4)

        self.DeUp2 = DeUp_Cat(in_channels=base_channels*4, out_channels=base_channels*2)
        self.DeBlock2 = DeBlock(in_channels=base_channels*2)

        self.DeUp1 = DeUp_Cat(in_channels=base_channels * 2, out_channels=base_channels)
        self.DeBlock1 = DeBlock(in_channels=base_channels*1)

        self.EndConv = EndConv(in_channels=base_channels, out_channels=num_classes)
        # self.Sigmoid = nn.Sigmoid()
        self.Softmax = nn.Softmax(dim=1)

        self.weight_attention = nn.Parameter(torch.ones(1))

    def forward(self, x0):
        x = self.InitConv(x0)  # (1, 16, 128, 128, 128)
        # print('x.shape:', x.shape)
        x1_1 = self.EnBlock1(x)
        x1_2 = self.EnDown1(x1_1)  # (1, 32, 64, 64, 64)

        x2_1 = self.EnBlock2_1(x1_2)
        x2_1 = self.EnBlock2_2(x2_1)
        x2_2 = self.EnDown2(x2_1)  # (1, 64, 32, 32, 32)

        x3_1 = self.EnBlock3_1(x2_2)
        x3_1 = self.EnBlock3_2(x3_1)
        x3_2 = self.EnDown3(x3_1)  # (1, 128, 16, 16, 16)

        x4_1 = self.EnBlock4_1(x3_2)
        x4_2 = self.EnBlock4_2(x4_1)
        x4_3 = self.EnBlock4_3(x4_2)
        x4 = self.EnBlock4_4(x4_3)   # (1, 128, 16, 16, 16)
        # x4 = torch.cat((x4_1, x4_2, x4_3, x4_4), dim=1)
        # print('x4:', x4.shape)
        x4_pre = x4

        attention = self.AttentionBlock(x0)
        x4_out, Euclidean_dis = both(x4, attention)
        weight = self.weight_attention
        # x4 = x4 + x4_out * weight
        x4 = x4 + x4_out
        x4_post = x4

        y3_1 = self.DeUp3(x4, x3_1)  # (1, 64, 32, 32, 32)
        y3_2 = self.DeBlock3(y3_1)

        y2_1 = self.DeUp2(y3_2, x2_1)  # (1, 32, 64, 64, 64)
        y2_2 = self.DeBlock2(y2_1)

        y1_1 = self.DeUp1(y2_2, x1_1)  # (1, 16, 128, 128, 128)
        y1_2 = self.DeBlock1(y1_1)

        y = self.EndConv(y1_2)
        # output = self.Sigmoid(y)
        output = self.Softmax(y)

        return output, attention, Euclidean_dis, weight, x4_pre, x4_post


if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((2, 4, 128, 128, 128), device=cuda0)
        # model = Unet1(in_channels=4, base_channels=16, num_classes=4)
        model = Unet_Both(in_channels=4, base_channels=16, num_classes=4)
        model.cuda()
        output, attention, Euclidean_dis, weight = model(x)
        print('output:', output.shape, 'attention:', attention.shape, 'Euclidean_dis:', Euclidean_dis)

        flops, params = get_model_complexity_info(model, (4, 128, 128, 128), as_strings=True, print_per_layer_stat=False)
        print('Flops:  ' + flops)
        print('Params: ' + params)
