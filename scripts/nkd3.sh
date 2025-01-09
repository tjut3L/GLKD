

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill nkd --model_s MobileNetV2_1_0 -r 1 -a 1 -b 1 --trial 1 --gpu 1
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill nkd --model_s MobileNetV2 -r 1 -a 1 -b 1 --trial 1 --gpu 1
