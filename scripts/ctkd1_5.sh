

#python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s resnet8x4 r 5 -a 1 -b 0.1 --trial 1 --gpu 1
python train_student.py --path_t ./save/models/resnet110x2_vanilla/ckpt_epoch_240.pth --distill ctkd1 --model_s resnet110 -r 10 -a 1 -b 1 --trial 1 --gpu 3
python train_student.py --path_t ./save/models/resnet110x2_vanilla/ckpt_epoch_240.pth --distill ctkd1 --model_s resnet116 -r 10 -a 1 -b 1 --trial 1 --gpu 3