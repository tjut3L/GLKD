

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s resnet8x4 --model_st resnet32x4 -r 5 -a 1 -b 0.1 --trial 3 --gpu 3
python train_student.py --path_t ./save/models/resnet110x2_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s resnet110 --model_st resnet110x2 -r 5 -a 1 -b 0.1 --trial 3 --gpu 3
python train_student.py --path_t ./save/models/resnet110x2_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s resnet116 --model_st resnet110x2 -r 5 -a 1 -b 0.1 --trial 3 --gpu 3
python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s vgg8 --model_st vgg13 -r 5 -a 1 -b 0.01 --trial 3 --gpu 3

#python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s vgg8 --model_st resnet32x4 -r 5 -a 1 -b 0.01 --trial 1 --gpu 3
#python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s MobileNetV2 --model_st ResNet50 -r 5 -a 1 -b 0.01 --trial 1 --gpu 3
#python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s ShuffleV1 --model_st resnet32x4 -r 5 -a 1 -b 0.01 --trial 1 --gpu 3
#python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s ShuffleV2 --model_st resnet32x4 -r 5 -a 1 -b 0.01 --trial 1 --gpu 3
#
#python train_student.py --path_t ./save/models/resnet110x2_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s ShuffleV2 --model_st resnet110x2 -r 5 -a 1 -b 0.01 --trial 1 --gpu 3
#python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s MobileNetV2_1_0 --model_st resnet32x4 -r 5 -a 1 -b 0.01 --trial 1 --gpu 3
#python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s MobileNetV2 --model_st wrn_40_2 -r 5 -a 1 -b 0.01 --trial 1 --gpu 3
#python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s wrn_16_2 --model_st resnet32x4 -r 5 -a 1 -b 0.01 --trial 1 --gpu 3
