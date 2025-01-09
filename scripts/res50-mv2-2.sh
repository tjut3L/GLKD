



python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill dist --model_s MobileNetV2 --model_st ResNet50 -r 1 -a 0 -b 1 --trial 1 --gpu 6
python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s MobileNetV2 --model_st ResNet50 -r 0.1 -a 0.9 -b 0 --trial 1 --gpu 6

