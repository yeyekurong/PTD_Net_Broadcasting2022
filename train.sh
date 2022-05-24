cd
source .bashrc
source activate
conda activate py1.1
cd /data2/GQ/cwgan
cp /data2/GQ/cwgan/Enhancement_models/cycle_gan_model_EBP.py /data2/GQ/cwgan/Enhancement_models/cycle_gan_model.py
cp /data2/GQ/cwgan/Enhancement_models/cgan_model_EBP.py /data2/GQ/cwgan/Enhancement_models/cgan_model.py
cp /data2/GQ/cwgan/Enhancement_models/networks-fasterrcnn.py /data2/GQ/cwgan/Enhancement_models/networks.py
mv /data2/GQ/cwgan/datasets/enhance/test.txt /data2/GQ/cwgan/datasets/enhance/test1.txt
cp /data2/GQ/faster-rcnn.pytorch-pytorch-1.0/data/VOCdevkit2007/VOC2007/ImageSets/Main/thinfog.txt /data2/GQ/cwgan/datasets/enhance/train_B.txt
#CUDA_VISIBLE_DEVICES=3,4,5,6 python Enhance_train.py --name PTD_EBP_2000_3 --init_epoch 280 --gpu_ids "0,1,2,3" --test_dir enhance/test --train_fogdir enhance/EBP_small --crop_size 272 #--continue_train --load_iter 22
#CUDA_VISIBLE_DEVICES=2 python Enhance_train.py --name PTD_EBP_4000_2 --continue_train --load_iter 200 --init_epoch 201 --refine_epoch 270 --feature_culsum --model cgan --gpu_ids 0 --test_dir enhance/test --train_fogdir_l enhance/EBP

cp /data2/GQ/cwgan/Enhancement_models/cycle_gan_model_network.py /data2/GQ/cwgan/Enhancement_models/cycle_gan_model.py
cp /data2/GQ/cwgan/Enhancement_models/cgan_model_network.py /data2/GQ/cwgan/Enhancement_models/cgan_model.py
CUDA_VISIBLE_DEVICES=4,5,6 python Enhance_train.py --name PTD_RFB_network_NGAN --init_epoch 300 --gpu_ids "0,1,2" --test_dir enhance/test --crop_size 272 --train_fogdir enhance/real_defog_small #--continue_train --load_iter 38
#CUDA_VISIBLE_DEVICES=2 python Enhance_train.py --name PTD_RFB_network4 --continue_train --load_iter 202 --init_epoch 203 --refine_epoch 267 --feature_culsum --model cgan --gpu_ids 0 --test_dir enhance/test  