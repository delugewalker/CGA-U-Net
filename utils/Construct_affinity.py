import numpy as np
import torch
from torch.nn.functional import interpolate

class_index = [0, 1, 2, 4]


def one_hot(ori, classes=4, rate=8):

    batch, h, w, d = ori.size()
    ori = interpolate(ori.unsqueeze(1).float(), size=(h // rate, w // rate, d // rate), mode='nearest')
    ori = ori.squeeze(1)
    # print('ori:', ori.shape)
    batch, h, w, d = ori.size()
    new_gd = torch.zeros((batch, classes, h, w, d), dtype=ori.dtype).cuda()
    for j in class_index:
        index_list = torch.nonzero((ori == j))
        # print('index_list:', index_list.shape)
        if j == 4:
            j = 3
        for i in range(len(index_list)):
            batch, height, width, depth = index_list[i]
            new_gd[batch, j, height, width, depth] = 1

    return new_gd


def reconstruct_one_hot(label, classes=4, rate=8):

    batch, h, w, d = label.shape
    # new_label = np.zeros((batch, h//rate, w//rate), dtype=label.dtype)
    # new_label = interpolate(label.unsqueeze(1).float(), size=(h//rate, w//rate, d//rate), mode='nearest')
    # new_label = new_label.squeeze(1)
    # print(new_label.unique())
    label_one_hot = one_hot(label, classes)
    # print(label_one_hot.unique())
    # label_one_hot = label_one_hot.float().cuda()  # cuda()
    B, C, H, W, D = label_one_hot.size()
    label_re = label_one_hot.view(B, C, H*W*D)
    label_re_t = label_re.permute(0, 2, 1)
    label_att = torch.bmm(label_re_t, label_re)

    return label_att


if __name__ == '__main__':

    ori = np.random.randint(0, 4, size=(2, 128, 128, 128), dtype='int16')
    ori[np.where(ori == 3)] = 4
    # print(ori)
    ori = torch.from_numpy(ori)
    new = one_hot(ori, 4)
    print(new.shape)

    output = reconstruct_one_hot(ori, 4, 8)
    print(output.shape)
