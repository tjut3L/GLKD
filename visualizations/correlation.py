import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from models import model_dict
from tsne import resnet8x4, resnet32x4
from vgg import vgg13, vgg8
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


# visualize the difference between the teacher's output logits and the student's
def get_output_metric(model, val_loader, num_classes=100):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, (data, labels) in tqdm(enumerate(val_loader)):
            outputs, _ = model(data)
            preds = outputs
            all_preds.append(preds.data.cpu().numpy())
            all_labels.append(labels.data.cpu().numpy())

    all_preds = np.concatenate(all_preds, 0)
    all_labels = np.concatenate(all_labels, 0)
    matrix = np.zeros((num_classes, num_classes))
    cnt = np.zeros((num_classes, 1))
    for p, l in zip(all_preds, all_labels):
        cnt[l, 0] += 1
        matrix[l] += p
    matrix /= cnt
    return matrix


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = resnet32x4(num_classes=100)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model


def get_tea_stu_diff(tea, stu, mpath, max_diff):
    # cfg.defrost()
    # cfg.DISTILLER.STUDENT = stu
    # cfg.DISTILLER.TEACHER = tea
    # cfg.DATASET.TYPE = 'cifar100'
    # cfg.freeze()
    train_loader, val_loader, num_data = get_cifar100_dataloaders(batch_size=64,
                                                                  num_workers=8,
                                                                  is_instance=True)

    # model = cifar_model_dict[cfg.DISTILLER.STUDENT][0](num_classes=100)
    # model.load_state_dict(load_checkpoint(mpath)["model"])
    model = resnet8x4(num_classes=100)
    model.load_state_dict(torch.load(mpath)['model'])
    tea_model = load_teacher("/home/tjut_jinjiali/models/resnet32x4_vanilla/ckpt_epoch_240.pth")
    print("load model successfully!")

    ms = get_output_metric(model, val_loader)
    mt = get_output_metric(tea_model, val_loader)
    diff = np.abs((ms - mt)) / max_diff
    for i in range(100):
        diff[i, i] = 0
    print('max(diff):', diff.max())
    print('mean(diff):', diff.mean())
    seaborn.heatmap(diff, vmin=0, vmax=1.0, cmap="PuBuGn")
    plt.show()


MAX_DIFF = 3.0

# mpath = "/home/tjut_jinjiali/RepDistiller/save/student_models/kd/S:resnet8x4_T:resnet32x4_cifar100_kd_r:0.1_a:0.9_b:0.0_tg:0_tl:0_1/resnet8x4_best.pth"
mpath = "/home/tjut_jinjiali/RepDistiller/save/student_models/dkd/S:resnet8x4_T:resnet32x4_cifar100_dkd_r:1.0_a:0.0_b:1.0_tg:0_tl:0_1/resnet8x4_best.pth"
get_tea_stu_diff("resnet32x4", "resnet8x4", mpath, MAX_DIFF)
