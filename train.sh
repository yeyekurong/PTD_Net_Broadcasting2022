cd /data2/GQ/cwgan

CUDA_VISIBLE_DEVICES=4,5,6 python Enhance_train.py --name PTD_RFB --init_epoch 300 --gpu_ids "0,1,2" --test_dir enhance/test --crop_size 272 --train_fogdir enhance/real_defog_small
CUDA_VISIBLE_DEVICES=2 python Enhance_train.py --name PTD_RFB --continue_train --load_iter 200 --init_epoch 201 --refine_epoch 267 --feature_culsum --model cgan --gpu_ids 0 --test_dir enhance/test  