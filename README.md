####
Paper:
"Qiang Guo, Mingliang Zhou. Progressive Domain Translation Defogging Network for Real-world Fog Images [J]. IEEE transactions on broadcasting", 
Using the defogging network in this paper instead of the traditional method in yeyekurong/Compatable_defogging for defogging will make the enhance effect more stable and low-noise.

####
Requirment environment:
pytorch 1.1
python3.6

####
Train:
bash train.sh

####
Inference:
python test.py --name SGP_unsuper --gpu_ids 0 --load_iter 230
Input images in 'datasets/enhance/real_hazy/' and output images in results/SGP_unsuper/test_latest/images/

####
Future:
Now, we have upgraded the PTD_Net to improve its enhance results on observation and object detection. The newest PTD_Net2 will be released this year.
