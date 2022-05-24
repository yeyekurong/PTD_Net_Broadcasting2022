"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import sys,os
import time
import torch
#torch.set_printoptions(profile="full")
from Enhancement_models.options.train_options import TrainOptions
from Enhancement_models.options.test_options import TestOptions
from data import create_dataset
from Enhancement_models import create_model
#import torch.backends.cudnn as cudnn
from util.visualizer import Visualizer
from util.visualizer import save_images
from util import html
import numpy as np 
import warnings
warnings.filterwarnings("ignore")
torch.set_printoptions(threshold=np.inf)

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True
    opt.max_dataset_size = 3
    optaspect_ratio = opt.aspect_ratio
    optdisplay_winsize = opt.display_winsize
    opt.preprocess = 'none'
    verify_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(verify_dataset)    # get the number of images in the dataset.
    print('===> The number of verifying images = %d' % dataset_size)    
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    
    opt = TrainOptions().parse()   # get training options
    #if torch.cuda.is_available():
    #    cudnn.benchmark = True
    opt.max_dataset_size = float("inf")
    opt.batch_size = len(opt.gpu_ids)*3
    opt.num_threads = opt.batch_size
    opt.preprocess = 'crop'
    print('===> The number of batch images = %d' % opt.batch_size)
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('===> The number of training images = %d' % dataset_size)
    
    #opt.lr = 0.00002
    if int(opt.load_iter)+int(opt.epoch_count) >= int(opt.init_epoch):
        if opt.target_type == 'yolo':
            opt.lr = 0.0005
        print('===> lr is set to %f'% (opt.lr))
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    detection_initialize = True
    if isinstance(opt.load_iter, str):
        opt.load_iter = max([int(load_iter) for load_iter in opt.load_iter.split(',')])
    for epoch in range(opt.load_iter+opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        t_data_sum = 0
        print('===> Start of epoch %d / %d'% (epoch, opt.niter + opt.niter_decay))
        if detection_initialize and epoch>=opt.init_epoch:
            if opt.load_iter+opt.epoch_count<opt.init_epoch:
                print('===> End of cyclegan epoch %d / %d'% (epoch, opt.niter + opt.niter_decay))
                break
            detection_initialize = False
            opt.batch_size = len(opt.gpu_ids)*1
            opt.num_threads = len(opt.gpu_ids)*1
            opt.no_flip = True
            opt.preprocess = 'scale_min'
            opt.load_size = 320 #336
            opt.dataset_mode = 'detection'
            dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
            dataset_size = len(dataset)    # get the number of images in the dataset.
            print('===> The number of training images for detection adjust = %d' % dataset_size)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
                t_data_sum += t_data
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters(epoch, opt.init_epoch, opt.refine_epoch, opt.load_path, opt.target_type)   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            iter_data_time = time.time()
                   
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('===> Saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

            for i, data in enumerate(verify_dataset):  #verifying dataset 
                model.set_input(data)  # unpack data from data loader
                model.test(opt.state)           # run inference
                visuals = model.get_current_visuals()  # get image results
                img_path = model.get_image_paths()    # get image paths
                #if i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))
                visualizer.display_current_results(model.get_current_visuals(), epoch, 1, img_path[0], 'test')
             

        print('===> End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        print('===> End of epoch %d / %d \t Data Load Time Taken: %f sec' % (epoch, opt.niter + opt.niter_decay, t_data_sum))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
    
        