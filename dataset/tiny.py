import glob
import os
from shutil import move
from os import rmdir

import torch
from torchvision import transforms
from torchvision import datasets


target_folder = '/data/tjut_jinjiali/dataset/tiny-imagenet-200/val/'

val_dict = {}
with open('/data/tjut_jinjiali/dataset/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
    for line in f.readlines():
        split_line = line.split('\t')
        val_dict[split_line[0]] = split_line[1]

paths = glob.glob('/data/tjut_jinjiali/dataset/tiny-imagenet-200/val/images/*')
for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    if not os.path.exists(target_folder + str(folder)):
        os.mkdir(target_folder + str(folder))
        os.mkdir(target_folder + str(folder) + '/images')

for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    dest = target_folder + str(folder) + '/images/' + str(file)
    move(path, dest)

rmdir('/data/tjut_jinjiali/dataset/tiny-imagenet-200/val/images')




# def tiny_loader(batch_size, data_dir):
#     num_label = 200
#     normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
#     transform_train = transforms.Compose(
#         [transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
#          normalize, ])
#     transform_test = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize, ])
#     trainset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
#     testset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
#     train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
#     test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)
#     return train_loader, test_loader, num_label
