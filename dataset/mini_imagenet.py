from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader
import torch
from torchvision import datasets
from torchvision import transforms
from PIL import Image
# from .my_dataset import MyDataSet

from .my_dataset import MyDataSet


def get_data_folder():
    """
    return server-dependent path to store the data
    """
    data_folder = '/data/tjut_jjl/dataset/mini_imagenet'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


class ImageFolderInstance(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img, target = super().__getitem__(index)
        return img, target, index


def get_test_loader(dataset='mini_imagenet', batch_size=64, num_workers=8):
    """get the test data loader"""

    if dataset == 'mini_imagenet':
        data_folder = get_data_folder()
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_folder = os.path.join(data_folder, 'val')
    test_set = datasets.ImageFolder(test_folder, transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)

    return test_loader


def get_mini_imagenet_dataloader(dataset='mini_imagenet', batch_size=64, num_workers=8):
    """
    Data Loader for imagenet
    """
    if dataset == 'mini_imagenet':
        data_folder = get_data_folder()
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(72),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        normalize,
    ])

    data_root = '/data/tjut_jjl/dataset/mini_imagenet'
    json_path = './classes_name.json'

    train_dataset = MyDataSet(root_dir=data_root,
                              csv_name="new_train.csv",
                              json_path=json_path,
                              transform=train_transform)

    val_dataset = MyDataSet(root_dir=data_root,
                            csv_name="new_val.csv",
                            json_path=json_path,
                            transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=num_workers,
                                               collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=num_workers,
                                             collate_fn=val_dataset.collate_fn)
    return train_loader, val_loader


class MultiCropAugmentation(object):
    def __init__(self, local_crops_scale, local_crops_number):
        common_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        # global crop
        self.global_transform = transforms.Compose([
            transforms.RandomCrop(224, padding=4),
            common_transform,
            normalize,
        ])

        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=local_crops_scale, interpolation=Image.BICUBIC),
            common_transform,
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transform(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops
