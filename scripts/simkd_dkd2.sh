




python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill simkd --model_s MobileNetV2 -r 0 -a 0.01 -b 1 -tg 1 -tl 0.1 --trial 1 --gpu 0
python train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill simkd --model_s resnet32 -r 0 -a 0.01 -b 1 -tg 1 -tl 0.1 --trial 1 --gpu 0











