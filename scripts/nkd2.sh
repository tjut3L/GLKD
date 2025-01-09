

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill nkd --model_s ShuffleV1 -r 1 -a 1 -b 1 --trial 1 --gpu 3
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill nkd --model_s ShuffleV2 -r 1 -a 1 -b 1 --trial 1 --gpu 3
python train_student.py --path_t ./save/models/resnet110x2_vanilla/ckpt_epoch_240.pth --distill nkd --model_s ShuffleV2 -r 1 -a 1 -b 1 --trial 1 --gpu 3