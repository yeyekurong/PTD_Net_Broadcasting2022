####
Paper:
PTD-Net: Progressive Domain Translation Defogging Network for Real-world Image

####
Input images in 'datasets/enhance/real_hazy/' and output images in results/SGP_unsuper/test_latest/images/

####
Requirment environment:
pytorch 1.1
python3.6

####
Inference:
python test.py --name SGP_unsuper --gpu_ids 0 --load_iter 240
