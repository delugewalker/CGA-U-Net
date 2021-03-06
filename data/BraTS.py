import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision.transforms import transforms
import pickle
from scipy import ndimage
from config import opts
import monai.transforms as mt

config = opts()


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


# def MaxMinNormalization(x):
#     Max = np.max(x)
#     Min = np.min(x)
#     x = (x - Min) / (Max - Min)
#     return x


class MaxMinNormalization(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        Max = np.max(image)
        Min = np.min(image)
        image = (image - Min) / (Max - Min)

        return {'image': image, 'label': label}


class Random_Flip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = np.flip(image, 0)
            label = np.flip(label, 0)
        if random.random() < 0.5:
            image = np.flip(image, 1)
            label = np.flip(label, 1)
        if random.random() < 0.5:
            image = np.flip(image, 2)
            label = np.flip(label, 2)

        return {'image': image, 'label': label}


class Random_Crop(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        H = random.randint(0, config.input_H - config.crop_H)
        W = random.randint(0, config.input_W - config.crop_W)
        D = random.randint(0, config.input_D - config.crop_D)

        image = image[H: H + config.crop_H, W: W + config.crop_W, D: D + config.crop_D, ...]
        label = label[..., H: H + config.crop_H, W: W + config.crop_W, D: D + config.crop_D]

        return {'image': image, 'label': label}


class Random_intencity_shift(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        label = sample['label']

        scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[1, image.shape[1], 1, image.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1, image.shape[-1]])

        image = image*scale_factor+shift_factor

        return {'image': image, 'label': label}


class Random_rotate(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
        label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False)

        return {'image': image, 'label': label}


class Pad(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
        label = np.pad(label, ((0, 0), (0, 0), (0, 5)), mode='constant')

        return {'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
        label = sample['label']
        label = np.ascontiguousarray(label)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}


def monai_trans_train():
    keys = ("image", "label")
    dtype = (np.float32, np.uint8)
    mt_train = [
        # mt.LoadNiftiD(keys),
        mt.AddChannelD(keys[1]),
        mt.RandSpatialCropD(keys, roi_size=(128, 128, 128), random_size=False),
        mt.RandAffineD(
            keys,
            prob=0.15,
            rotate_range=(-0.05, 0.05),
            scale_range=(-0.1, 0.1),
            mode=("bilinear", "nearest"),
            as_tensor_output=False,
        ),
        mt.RandGaussianNoiseD(keys[0], prob=0.15, std=0.01),
        mt.RandRotateD(keys, range_x=10, range_y=10, range_z=10, prob=0.3, mode=("bilinear", "nearest"),),
        mt.RandFlipD(keys, spatial_axis=0, prob=0.5),
        mt.RandFlipD(keys, spatial_axis=1, prob=0.5),
        mt.RandFlipD(keys, spatial_axis=2, prob=0.5),
        mt.CastToTypeD(keys, dtype=dtype),
        mt.ToTensorD(keys)
    ]
    return mt.Compose(mt_train)


def transform(sample):
    trans = transforms.Compose([
        Pad(),
        # Random_rotate(),  # ???????????????
        Random_Crop(),
        Random_Flip(),
        Random_intencity_shift(),
        ToTensor()
    ])

    return trans(sample)


def transform_valid(sample):
    trans = transforms.Compose([
        Pad(),
        # MaxMinNormalization(),
        ToTensor()
    ])

    return trans(sample)


class BraTS(Dataset):
    def __init__(self, list_file, root='', mode='train'):
        self.lines = []
        paths, names = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, line, name + '_')
                paths.append(path)
                self.lines.append(line)
        self.mode = mode
        self.names = names
        self.paths = paths
        # print("Found %d samples" % (len(self.names)))

    def __getitem__(self, item):
        path = self.paths[item]
        if self.mode == 'train':
            image, label = pkload(path + 'data_f32b0.pkl')
            # image = image.transpose(3, 0, 1, 2)             # (4, 240, 240, 155)
            sample = {'image': image, 'label': label}
            # trans = monai_trans_train()
            # sample = trans(sample)
            sample = transform(sample)
            # print('image:', image.shape)
            # print('label:', label.shape)
            return sample['image'], sample['label']
        elif self.mode == 'valid':
            image, label = pkload(path + 'data_f32b0.pkl')
            sample = {'image': image, 'label': label}
            sample = transform_valid(sample)
            return sample['image'], sample['label']
        else:
            image = pkload(path + 'data_f32b0.pkl')
            image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
            image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
            image = torch.from_numpy(image).float()
            return image

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]


# ????????????
if __name__ == '__main__':

    train_list = '/data1/yuhong/Datasets/BraTS2019/Train/train.txt'
    train_root = '/data1/yuhong/Datasets/BraTS2019/Train/'

    valid_list = '/data1/yuhong/Datasets/BraTS2019/Valid/valid.txt'
    valid_root = '/data1/yuhong/Datasets/BraTS2019/Valid/'

    # mode = 'train'
    mode = 'test'

    train_set = BraTS(train_list, train_root, mode)
    valid_set = BraTS(valid_list, valid_root, mode)

    from torch.utils.data import DataLoader
    train_dl = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)
    valid_dl = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    if mode == 'train':
        for index, (image, label) in enumerate(train_dl):

            print(index, train_set.names[index], image.size(), label.size())
            # print('image_class:', np.unique(image))
            print('label_class:', np.unique(label))
            print('label:', label.shape)
            print('-------------------------------------------------------')
    else:
        for index, image in enumerate(valid_dl):
            print('name:', train_set.names[index])
            print(index+1, train_set.names[index], image.size())
            # print('image_class:', np.unique(image))
            # print('label_class:', np.unique(label))
            print('-------------------------------------------------------')

