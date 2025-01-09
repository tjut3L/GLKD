
#
#python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill tts --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --trial 1 --tg 1 --tl 1
#python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill tts --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --trial 1 --tg 0.6 --tl 0.4  #73.05
#python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill tts --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --trial 1 --tg 0.7 --tl 0.3  #73.43
#python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill tts --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --trial 1 --tg 0.5 --tl 0.5  #71.29
#


python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s resnet8x4 --model_st resnet32x4 -r 1 -a 1 -b 0.1 --trial 1 --gpu 1
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s resnet8x4 --model_st resnet32x4 -r 1 -a 1 -b 0.01 --trial 1 --gpu 1
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ctkd --model_s resnet8x4 --model_st resnet32x4 -r 10 -a 1 -b 0.1 --trial 1 --gpu 1
