

python train_student1.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s resnet8x4 --model_st resnet32x4 -r 5 -a 1 -b 1 --scratch 1 --trial 7 --method l1 --gpu 1
python train_student1.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s resnet8x4 --model_st resnet32x4 -r 5 -a 1 -b 1 --scratch 1 --trial 7 --method kl --gpu 1
python train_student1.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s resnet8x4 --model_st resnet32x4 -r 100 -a 1 -b 1 --scratch 1 --trial 7 --method cos --gpu 1



