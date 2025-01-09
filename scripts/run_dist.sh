
#

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s resnet8x4 --model_st resnet32x4 -r 1 -a 0.1 -b 1 --trial 1 --gpu 5
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s resnet8x4 --model_st resnet32x4 -r 1 -a 0.01 -b 1 --trial 1 --gpu 5
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s resnet8x4 --model_st resnet32x4 -r 10 -a 0.1 -b 1 --trial 1 --gpu 5

