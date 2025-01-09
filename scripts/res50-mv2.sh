


python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill crd --model_s MobileNetV2 --model_st ResNet50 -r 1 -a 0 -b 0.8 --trial 1 --gpu 5
python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill dkd --model_s MobileNetV2 --model_st ResNet50 -r 1 -a 0 -b 1 --trial 1 --gpu 5
