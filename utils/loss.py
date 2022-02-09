import torch
import torch.nn as nn
import logging

# TODO attention the output of model to use loss

class DiceLoss(nn.Module):
    """2D Cross Entropy Loss with Multi-L1oss"""
    def __init__(self):
        super(DiceLoss, self).__init__()

        self.Softmax = nn.Softmax(dim=1)

    def forward(self, *inputs):
        # pred, att, target = tuple(inputs) # unet_cp
        pred, target = tuple(inputs)# other
        pred = self.Softmax(pred)
        # print(pred.shape, target.shape)
        loss = softmax_dice(pred, target)
        # print('loss_main: {:.5f}'.format(loss))
        return loss


def softmax_dice(output, target):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss
    '''
    loss1 = Dice(output[:, 1, ...], (target == 1).float())
    loss2 = Dice(output[:, 2, ...], (target == 2).float())
    loss3 = Dice(output[:, 3, ...], (target == 4).float())

    return loss1 + loss2 + loss3, 1-loss1.data, 1-loss2.data, 1-loss3.data


def Dice(output, target, eps=1e-5):
    target = target.float()
    num = 2 * (output * target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den


class AffinityLoss(nn.Module):

    def __init__(self):
        super(AffinityLoss, self).__init__()

        self.Softmax = nn.Softmax(dim=1)

    def forward(self, *inputs):
        pred, att_map, target, aff_label = tuple(inputs)
        # print(pred.shape, target.shape)
        loss_main, loss1, loss2, loss3 = softmax_dice(pred, target)
        # loss_cp = global_term(att_map, aff_label)
        loss_att = MSE_loss(att_map, aff_label) * 20
        # loss_att = softmax_dice(att_map, aff_label)
        loss = loss_main + loss_att

        # print('loss_main: {:.5f} || loss_cp: {:.5f}'.format(loss_main, loss_cp*100))

        return loss, loss_main, loss_att, loss1, loss2, loss3


class AttentionLoss(nn.Module):

    def __init__(self):
        super(AttentionLoss, self).__init__()

        self.Softmax = nn.Softmax(dim=1)

    def forward(self, *inputs):
        att_map, aff_label = tuple(inputs)
        # print(pred.shape, target.shape)
        # loss_cp = global_term(att_map, aff_label)
        loss_att = MSE_loss(att_map, aff_label) * 100

        # print('loss_main: {:.5f} || loss_cp: {:.5f}'.format(loss_main, loss_cp*100))

        return loss_att


class AffinityLoss_small(nn.Module):

    def __init__(self):
        super(AffinityLoss_small, self).__init__()

        self.Softmax = nn.Softmax(dim=1)

    def forward(self, *inputs):
        pred, att_map, target, aff_label = tuple(inputs)
        # print(pred.shape, target.shape)
        loss_main = softmax_dice(pred, target)
        # loss_cp = global_term(att_map, aff_label)
        loss_cp = MSE_loss(att_map, aff_label) * 0.001
        loss = loss_main + loss_cp

        # print('loss_main: {:.5f} || loss_cp: {:.5f}'.format(loss_main, loss_cp*100))

        return loss, loss_main, loss_cp


def BCE_loss(pred, label):
    bce_loss = nn.BCEWithLogitsLoss()
    # pred[pred < 0.0] = 0.0
    #     pred[pred > 1.0] = 1.0

    bce_out = bce_loss(pred, label)
    # print("bce_loss:", bce_out.data.cpu().numpy())
    return bce_out


def MSE_loss(pred, label):
    mse_loss = nn.MSELoss()
    # pred[pred < 0.0] = 0.0
    #     pred[pred > 1.0] = 1.0

    mse_out = mse_loss(pred, label)
    # print("bce_loss:", bce_out.data.cpu().numpy())
    return mse_out


def CrossEntropy_loss(pred, label):
    crossentropy_loss = nn.CrossEntropyLoss()
    out = crossentropy_loss(pred, label)
    return out


def global_term(pred, label, eps=1e-6):
    # label = label.float()
    loss1 = BCE_loss(pred, label)
    func = nn.Sigmoid()
    pred = func(pred)
    assert len(pred.shape) == 3 and len(label.shape) == 3 and pred.shape == label.shape
    b, h, w = label.shape
    weight = torch.mul(pred, label)
    weight_sum = torch.sum(weight, -1)+eps  # (b,h)
    pred_sum = torch.sum(pred, -1) + eps
    label_sum = torch.sum(label, -1) + eps

    other_label_sum = torch.sum(1.0-label, -1) + eps
    other_weight = torch.mul((1.0-pred), (1.0-label))
    other_weight_sum = torch.sum(other_weight, -1) + eps

    Tp = torch.div(weight_sum, pred_sum)
    Tr = torch.div(weight_sum, label_sum)
    Ts = torch.div(other_weight_sum, other_label_sum)

    Lp = torch.mean(torch.log(Tp))  # all mean
    Lr = torch.mean(torch.log(Tr))
    Ls = torch.mean(torch.log(Ts))
    # print(loss1, Lp, Lr, Ls)
    loss2 = Lp + Lr + Ls

    return loss1 + loss2*(-1)