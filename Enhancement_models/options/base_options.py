import argparse
import os
from util import util
import torch
import Enhancement_models
import data
import glob

class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', type=str, default='datasets', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default='enhance_model', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='resnet_6blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        parser.add_argument('--load_size', type=int, default=440, help='scale images to this size') #440
        parser.add_argument('--crop_size', type=int, default=224, help='then crop to this size')  #224
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_min | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, flip the images for data augmentation')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        parser.add_argument('--defog_model', type=str, default='', help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--train_cleardir', type=str, default='enhance/gt_train_small', help='directory of train images to run')
        parser.add_argument('--train_vocdir', type=str, default='enhance/VOCimages', help='how many test images to run')
        parser.add_argument('--train_fogdir', type=str, default='enhance/real_defog_small', help='directory of train images to run')
        parser.add_argument('--train_fogdir_l', type=str, default='enhance/real_defog', help='directory of train images to run')
        parser.add_argument('--test_dir', type=str, default='enhance/real_hazy', help='how many test images to run')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=str, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--state', default='test', type=str, help='chooses if use the defog model before the gan network, [train|test|Etest]')
        # target loss parameters
        parser.add_argument('--feature_culsum', action='store_true', help='trainging the generators by the culsum difference of object classfication feature')
        parser.add_argument('--init_epoch', default=155, type=int, help='start epoch to init the target loss')
        parser.add_argument('--refine_epoch', default=231, type=int, help='start epoch to init the target loss')
        parser.add_argument('--detection_type', default='vgg16', type=str, help='vgg16, res101')
        parser.add_argument('--target_type', default='fasterrcnn', type=str, help='fasterrcnn, yolo')
        parser.add_argument('--detection_model', default='checkpoints/detection_model/faster_rcnn_1_16_10021.pth',help='detection model to load model', type=str)
        parser.add_argument('--detection_annotation', default='/data2/GQ/cwgan/datasets/enhance/VOCdevkit2007-RTTS/VOC2007/Annotations/',help='detection model to load model', type=str)
        parser.add_argument('--detection_annotation_source', default='/data2/GQ/cwgan/datasets/enhance/VOCdevkit2007-voc/VOC2007/Annotations/',help='detection model to load model', type=str)
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = Enhancement_models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)
        
        # define the defog model
        max_epoch = 0
        for modelPath in glob.glob(os.path.join('checkpoints/defog_model/', '*.pth')):
            po = modelPath.find('epoch')
            modelPath_last = modelPath[po+6:]
            po = modelPath_last.find('_')
            epoch = int(modelPath_last[:po])
            if epoch > max_epoch:
                max_epoch = epoch
                model = modelPath
            #print('max defog model',max_epoch)
        parser.set_defaults(defog_model=model)     
        
        load_filename = '%s_net_G_B.pth' % (opt.init_epoch-1)
        save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        G_B_load_path = os.path.join(save_dir, load_filename)
        parser.set_defaults(load_path=G_B_load_path)    
        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt