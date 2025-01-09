



                                                                                                                                                       # -r L2   -a div(dkd_loss) -b attention


python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s resnet8x4 --model_st resnet32x4 -r 0.1 -a 0.5 -b 0 --trial 0 --gpu 4
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s ShuffleV1 --model_st resnet32x4 -r 0.1 -a 0.5 -b 0 --trial 0 --gpu 4
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s ShuffleV2 --model_st resnet32x4 -r 0.1 -a 0.5 -b 0 --trial 0 --gpu 4