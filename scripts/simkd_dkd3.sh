




python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill simkd --model_s vgg8 -r 0 -a 0.01 -b 1 -tg 1 -tl 0.01 --trial 1 --gpu 1
python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill simkd --model_s vgg8 -r 0 -a 0.1 -b 1 -tg 1 -tl 0.01 --trial 1 --gpu 1
python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill simkd --model_s resnet20 -r 0 -a 0.01 -b 1 -tg 1 -tl 0.1 --trial 1 --gpu 1











