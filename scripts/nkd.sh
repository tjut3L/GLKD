

python train_student.py --path_t ./save/models/resnet110x2_vanilla/ckpt_epoch_240.pth --distill nkd --model_s resnet110 -r 1 -a 1 -b 1 --trial 1 --gpu 7
python train_student.py --path_t ./save/models/resnet110x2_vanilla/ckpt_epoch_240.pth --distill nkd --model_s resnet116 -r 1 -a 1 -b 1 --trial 1 --gpu 7
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill nkd --model_s vgg8 -r 1 -a 1 -b 1 --trial 1 --gpu 7