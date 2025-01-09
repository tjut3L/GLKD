


python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s vgg8 --model_st resnet32x4 -r 5 -a 1 -b 0.1 --trial 3 --gpu 7
python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s MobileNetV2 --model_st ResNet50 -r 5 -a 1 -b 0.1 --trial 3 --gpu 7
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s ShuffleV1 --model_st resnet32x4 -r 5 -a 1 -b 0.1 --tria 3 --gpu 7
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s ShuffleV2 --model_st resnet32x4 -r 5 -a 1 -b 0.1 --trial 3 --gpu 7
