from __future__ import print_function

import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders
from dataset.tiny_imagenet import get_tiny_imagenet_dataloader
from dataset.mini_imagenet import get_mini_imagenet_dataloader

from helper.util import adjust_learning_rate, accuracy, AverageMeter, save_dict_to_json
from helper.loops import train_vanilla as train, validate


def parse_option():
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,60,90', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='resnet110')
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset')

    parser.add_argument('--gpu', type=str, default='4', help='id(s) for CUDA_VISIBLE_DEVICES')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = './save/mini/models'
        opt.tb_path = './save/mini/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'local_{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                                  opt.weight_decay, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


total_time = time.time()


def main():
    best_acc = 0
    global total_time

    opt = parse_option()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 100
    elif opt.dataset == 'tiny-imagenet':
        train_loader, val_loader = get_tiny_imagenet_dataloader(batch_size=opt.batch_size,
                                                                num_workers=opt.num_workers)
        n_cls = 200
    elif opt.dataset == 'mini_imagenet':
        train_loader, val_loader = get_mini_imagenet_dataloader(batch_size=opt.batch_size,
                                                                num_workers=opt.num_workers)
        n_cls = 100
    elif opt.dataset == 'stl_10':
        train_loader, val_loader, n_data = get_stl_10_dataloaders(batch_size=opt.batch_size,
                                                                  num_workers=opt.num_workers)
        n_cls = 10
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model = model_dict[opt.model](num_classes=n_cls)

    # num_ftrs = model.fc
    # model.fc = nn.Linear(num_ftrs, n_cls)
    #
    #
    # # /home/tjut_wangshuo/RepDistiller/save/student_models/kd/S:wrn_16_2_T:wrn_40_2_cifar100_kd_r:0.1_a:0.9_b:0.0_tg:0_tl:0_5/wrn_16_2_best.pth
    # model_path = '/home/tjut_wangshuo/RepDistiller/save/student_models/kd/S:wrn_16_2_T:wrn_40_2_cifar100_kd_r:0.1_a:0.9_b:0.0_tg:0_tl:0_5/wrn_16_2_best.pth'
    # model.load_state_dict(torch.load(model_path)['model'])

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))

            test_merics = {'test_loss': test_loss,
                           'test_acc': test_acc,
                           'test_acc_top5': test_acc_top5,
                           'epoch': epoch}

            save_dict_to_json(test_merics, os.path.join(opt.save_folder, "test_best_metrics.json"))

            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)

    # save parameters
    save_state = {k: v for k, v in opt._get_kwargs()}
    # No. parameters(M)
    num_params = (sum(p.numel() for p in model.parameters()) / 1000000.0)
    save_state['Total params'] = num_params
    save_state['Total time'] = (time.time() - total_time) / 3600.0
    params_json_path = os.path.join(opt.save_folder, "parameters.json")
    save_dict_to_json(save_state, params_json_path)

    # save model
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
