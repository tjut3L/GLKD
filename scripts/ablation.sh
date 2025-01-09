

python train_student1.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s MobileNetV2 --model_st ResNet50 -r 5 -a 1 -b 1 --scratch 1 --trial 7 --method l1 --gpu 3
python train_student1.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s MobileNetV2 --model_st ResNet50 -r 5 -a 1 -b 1 --scratch 1 --trial 7 --method kl --gpu 3
python train_student1.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s MobileNetV2 --model_st ResNet50 -r 100 -a 1 -b 1 --scratch 1 --trial 7 --method cos --gpu 3

