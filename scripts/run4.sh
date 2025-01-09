


python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s ShuffleV2 --model_st resnet32x4 -r 0.01 -a 1 -b 1000 --trial 1 --gpu 5         #75.2
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s ShuffleV2 --model_st resnet32x4 -r 0.01 -a 1 -b 1000 -s 0.5 --trial 1 --gpu 5  #74.27


