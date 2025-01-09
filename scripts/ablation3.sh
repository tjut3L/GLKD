

python train_student1.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s vgg8 --model_st vgg13 -r 5 -a 1 -b 1 --scratch 1 --trial 7 --method l1 --gpu 0
python train_student1.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s vgg8 --model_st vgg13 -r 5 -a 1 -b 1 --scratch 1 --trial 7 --method kl --gpu 0
python train_student1.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s vgg8 --model_st vgg13 -r 100 -a 1 -b 1 --scratch 1 --trial 7 --method cos --gpu 0

