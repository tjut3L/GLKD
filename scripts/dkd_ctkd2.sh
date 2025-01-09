



                                                                                                                                                       # -r L2   -a div(dkd_loss) -b attention


python train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s resnet32 --model_st resnet110 -r 0.1 -a 0.5 -b 0 --trial 0 --gpu 6
python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s resnet20 --model_st resnet56 -r 0.1 -a 0.5 -b 0 --trial 0 --gpu 6
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s wrn_40_1 --model_st wrn_40_2 -r 0.1 -a 0.5 -b 0 --trial 0 --gpu 6