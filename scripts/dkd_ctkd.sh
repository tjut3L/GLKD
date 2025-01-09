



                                                                                                                                                       # -r L2   -a div(dkd_loss) -b attention


python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s vgg8 --model_st vgg13 -r 0.1 -a 1 -b 0 --trial 1 --gpu 0
python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s vgg8 --model_st vgg13 -r 0.1 -a 1 -b 500 --trial 1 --gpu 0
python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s vgg8 --model_st vgg13 -r 0.1 -a 1 -b 1000 --trial 1 --gpu 0


python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s vgg8 --model_st vgg13 -r 0.01 -a 1 -b 0 --trial 1 --gpu 0
python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s vgg8 --model_st vgg13 -r 0.01 -a 1 -b 500 --trial 1 --gpu 0

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s resnet8x4 --model_st resnet32x4 -r 0.1 -a 1 -b 0 --trial 1 --gpu 0
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s resnet8x4 --model_st resnet32x4 -r 0.01 -a 1 -b 0 --trial 1 --gpu 0



python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s ShuffleV1 --model_st resnet32x4 -r 0.1 -a 1 -b 1000 --trial 1 --gpu 0
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s ShuffleV1 --model_st resnet32x4 -r 0.1 -a 1 -b 500 --trial 1 --gpu 0
