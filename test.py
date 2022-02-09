import argparse
import os
import time
import random
import numpy as np
import setproctitle

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
cudnn.benchmark = True

from data.BraTS import BraTS
from predict import validate_softmax
import models

parser = argparse.ArgumentParser()

parser.add_argument('--user', default='yuhong', type=str)

parser.add_argument('--root', default='/data1/yuhong/Datasets/BraTS2019/', type=str)

parser.add_argument('--valid_dir', default='Valid', type=str)

parser.add_argument('--valid_file', default='valid.txt', type=str)

parser.add_argument('--output_dir', default='output', type=str)

parser.add_argument('--submission', default='submission', type=str)

parser.add_argument('--visual', default='visualization', type=str)

parser.add_argument('--heatmap_dir', default='heatmap', type=str)

parser.add_argument('--experiment', default='Unet_', type=str)

parser.add_argument('--test_date', default='2020-12-23', type=str)

parser.add_argument('--test_file', default='model_epoch_last.pth', type=str)

parser.add_argument('--use_TTA', default=False, type=bool)

parser.add_argument('--post_process', default=True, type=bool)

parser.add_argument('--save_format', default='nii', choices=['npy', 'nii'], type=str)

parser.add_argument('--crop_H', default=128, type=int)

parser.add_argument('--crop_W', default=128, type=int)

parser.add_argument('--crop_D', default=128, type=int)

parser.add_argument('--seed', default=1223, type=int)

parser.add_argument('--model_name', default='Unet', type=str)

parser.add_argument('--model', default='resnet', type=str)

parser.add_argument('--model_depth', default=34, type=int)

parser.add_argument('--resnet_shortcut', default='B', type=str)

parser.add_argument('--num_class', default=4, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='0', type=str)

parser.add_argument('--num_workers', default=4, type=int)

args = parser.parse_args()


def main():

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model = getattr(models, args.model_name)(in_channels=4, base_channels=16, num_classes=4)
    model = torch.nn.DataParallel(model).cuda()

    load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                             'checkpoint', args.experiment+args.test_date, args.test_file)

    if os.path.exists(load_file):
        checkpoint = torch.load(load_file)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        print('Successfully load checkpoint {}'.format(os.path.join(args.experiment+args.test_date, args.test_file)))
    else:
        print('There is no resume file to load!')

    valid_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    valid_root = os.path.join(args.root, args.valid_dir)
    valid_set = BraTS(valid_list, valid_root, mode='test')
    print('Samples for valid = {}'.format(len(valid_set)))

    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    submission = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.output_dir,
                              args.submission, args.experiment+args.test_date)
    visual = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.output_dir,
                          args.visual, args.experiment+args.test_date)
    heatmap = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.output_dir,
                          args.heatmap_dir, args.experiment + args.test_date)
    if not os.path.exists(submission):
        os.makedirs(submission)
    if not os.path.exists(visual):
        os.makedirs(visual)
    if not os.path.exists(heatmap):
        os.makedirs(heatmap)

    start_time = time.time()

    with torch.no_grad():
        validate_softmax(valid_loader=valid_loader,
                         model=model,
                         load_file=load_file,
                         multimodel=False,
                         heatmap_use=False,
                         savepath=submission,
                         visual=visual,
                         heatmap_dir=heatmap,
                         names=valid_set.names,
                         scoring=False,
                         use_TTA=args.use_TTA,
                         save_format=args.save_format,
                         snapshot=True,
                         postprocess=True
                         )

    end_time = time.time()
    full_test_time = (end_time-start_time)/60
    average_time = full_test_time/len(valid_set)
    print('{:.2f} minutes!'.format(average_time))


if __name__ == '__main__':
    setproctitle.setproctitle('{}: Testing!'.format(args.user))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    main()


