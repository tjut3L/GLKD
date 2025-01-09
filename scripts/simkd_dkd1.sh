




python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill simkd --model_s ShuffleV1 -r 0 -a 0.01 -b 1 -tg 1 -tl 0.1 --trial 1 --gpu 0
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill simkd --model_s ShuffleV2 -r 0 -a 0.01 -b 1 -tg 1 -tl 0.1 --trial 1 --gpu 0




python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill simkd_dkd --model_s vgg8 -r 0 -a 0.01 -b 1 -tg 0 -tl 1 --trial 1 --gpu 0
python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill simkd_dkd --model_s vgg8 -r 0 -a 0.1 -b 1 -tg 0 -tl 1 --trial 1 --gpu 1











