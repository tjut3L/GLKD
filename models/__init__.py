from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, \
    resnet56, resnet110, resnet8x4, resnet32x4, resnet110x2, resnet116

from .resnetv2 import ResNet50, ResNet34, ResNet18
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .mobilenetv2 import mobile_half, mobile_half_double
from .mobilenetv1 import MobileNetV1
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2, ShuffleV2_1_5

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet110x2': resnet110x2,
    'resnet116': resnet116,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'ResNet50': ResNet50,
    'ResNet34': ResNet34,
    'ResNet18': ResNet18,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'MobileNetV2': mobile_half,
    'MobileNetV1': MobileNetV1,
    'MobileNetV2_1_0': mobile_half_double,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2_1_5': ShuffleV2_1_5,
    'ShuffleV2': ShuffleV2,
}
