import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from Defogging_models.networks import Defogging_Net
import torchvision.transforms as transforms
import torch.nn.functional as F 
from .target_loss.model.faster_rcnn.vgg16 import vgg16
#from .target_loss.model.faster_rcnn.resnet import resnet
from .target_loss.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from utils.loss import ComputeLoss
import pdb
import numpy as np
import math
from models.experimental import attempt_load
from utils.general import check_img_size

def calc_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolates = real_data
        elif type == 'fake':
            interpolates = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = alpha.to(device)
            interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
            interpolates = interpolates.to(device)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolates.requires_grad_(True)
        disc_interpolates = netD(interpolates)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty
    else:
        return 0.0, None
###############################################################################
# Helper Functions
###############################################################################
def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'layer':
        norm_layer = functools.partial(nn.LayerNorm)        
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    #print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, type = 'G', init_type='normal', init_gain=0.02, gpu_ids=[], defog_model=None):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if torch.cuda.is_available():
        net.to(gpu_ids[0])
        init_weights(net, init_type, init_gain=init_gain)
        #print("===> Load network from GPU: ", defog_model)
        if type == 'G' and defog_model:
            weights = torch.load(defog_model)
            if isinstance(weights['defog_model'], torch.nn.DataParallel):
                net.defog_model.load_state_dict(weights['defog_model'].module.state_dict())
            else:
                net.defog_model.load_state_dict(weights['defog_model'].state_dict())
            print("===> Load dehaze network: ", defog_model)
        if len(gpu_ids)>1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
            print("===> %s is set to Multiple GPU"%type)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], defog_model=None):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, 'G', init_type, init_gain, gpu_ids, defog_model)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, 'D', init_type, init_gain, gpu_ids)

##############################################################################
# Classes
##############################################################################
class TargetLoss_yolo(nn.Module):
    """Define target objectives.
    """
    def __init__(self, load_size=640, device=''):
        super(TargetLoss_yolo, self).__init__()
        self.device = device
        weights = 'checkpoints/detection_model/yolov5s.pt'
        w = str(weights[0][0] if isinstance(weights, list) else weights)
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        self.model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        stride = int(self.model.stride.max())  # model stride
        load_size = [load_size, load_size]
        load_size = check_img_size(load_size, s=stride)  # check image size
        self.model(torch.zeros(1, 3, *load_size).to(device).type_as(next(self.model.parameters())))  # run once
        for param in self.model.parameters():
            param.requires_grad = False
        self.compute_loss = ComputeLoss(self.model)  # init loss class
        self.frame_id = 1
        self.mask_scale = 4
        self.yolo_classes = ['person', 'bicycle', 'car', 'motorbike', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.faster_rcnn_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        
    def get_color(self, idx):
        idx = idx * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return color
        
    def plot_tracking(self, image, tlwhs, scores=None, frame_id=0, fps=0., ids2=None):
        im = np.ascontiguousarray(np.copy(image))
        im_h, im_w = im.shape[:2]
    
        top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255
    
        text_scale = max(1, image.shape[1] / 1600.)
        text_thickness = 2
        line_thickness = max(1, int(image.shape[1] / 500.))
    
        radius = max(5, int(im_w/140.))
        cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                    (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)
    
        for i, tlwh in enumerate(tlwhs):
            #print(tlwh)
            x1, y1, x2, y2, cls, a = tlwh
            intbox = tuple(map(int, (x1, y1, x2, y2)))
            obj_id = int(i)
            id_text = '{}'.format(int(obj_id))
            if ids2 is not None:
                id_text = id_text + ', {}'.format(int(ids2[i]))
            _line_thickness = 1 if obj_id <= 0 else line_thickness
            color = self.get_color(abs(obj_id))
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
            cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                        thickness=text_thickness)
        return im
    
    def xywhn2xyxy(self, size, x, box_num, padw=0, padh=0):
      # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        w = size[3]
        h = size[2]
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = w * (x[:, 2] - x[:, 4] / 2) + padw  # top left x
        y[:, 1] = h * (x[:, 3] - x[:, 5] / 2) + padh  # top left y
        y[:, 2] = w * (x[:, 2] + x[:, 4] / 2) + padw  # bottom right x
        y[:, 3] = h * (x[:, 3] + x[:, 5] / 2) + padh  # bottom right y
        y[:, 4] = x[:, 1]
        return y
    
    def convert(self, size, x, box_num):
        
        w = 1. / size[3]
        h = 1. / size[2]
        #print(x.size())
        y = torch.zeros(x.size()[1], x.size()[2]+1)
        x = x[0]
        cls_list = []
        for i in x[:, 4]:
            cls = self.faster_rcnn_classes[int(i.item())-1]
            cls_list.append(self.yolo_classes.index(cls))
        y[:, 1] = torch.Tensor(cls_list)#class
        y[:, 2] = ((x[:, 0] + x[:, 2]) / 2) * w  # x center
        y[:, 3] = ((x[:, 1] + x[:, 3]) / 2) * h  # y center
        y[:, 4] = (x[:, 2] - x[:, 0]) * w  # width
        y[:, 5] = (x[:, 3] - x[:, 1]) * h  # height
        return y
        
    def forward(self, gt_data, target_data, gt_boxes, gt_num_boxes, target_boxes, num_boxes):
        
        size = gt_data.size()
        gt_h, gt_w = size[2], size[3]
        gt_mask_low = torch.ones(size=(1, 1, math.floor(gt_h/self.mask_scale), math.floor(gt_w/self.mask_scale))).type_as(gt_data) * 0.3
        size = target_data.size()
        ta_h, ta_w = size[2], size[3]
        target_mask_low = torch.ones(size=(1, 1, math.floor(ta_h/self.mask_scale), math.floor(ta_w/self.mask_scale))).type_as(gt_data) * 0.3

        if num_boxes:
            target_boxes = target_boxes[0][:num_boxes.item()].unsqueeze(0)
            self.model.train()
        if gt_num_boxes:
            gt_boxes = gt_boxes[0][:gt_num_boxes.item()].unsqueeze(0)
        
        #print('gt_data',gt_data.size())
        _, gt_base_feature_low = self.model(gt_data, augment=False, visualize=False)
        pred_target, target_base_feature_low = self.model(target_data, augment=False, visualize=False)
        #print('gt_base_feature_low',gt_base_feature_low.size())
        
        targets = self.convert(size, target_boxes, num_boxes).to(self.device)
        #target_boxes = self.xywhn2xyxy(size, targets, num_boxes).to(self.device)
        #online_im =  self.plot_tracking(gt_data[0].cpu().detach().numpy().transpose(1, 2, 0), target_boxes, frame_id=self.frame_id)
        #cv2.imwrite( '{:05d}.jpg'.format(self.frame_id), online_im)
        #print(online_im.shape)
        #print('{:05d}.jpg'.format(self.frame_id))
        #print(targets)
        #print(len(pred_target))
        #print(pred_target[0].size())
        loss, loss_items = self.compute_loss(pred_target, targets)
        
        for i in range(gt_num_boxes):
            min_h = max(math.floor(gt_boxes[0][i][1]/self.mask_scale)-3,0)
            max_h = min(math.ceil(gt_boxes[0][i][3]/self.mask_scale)+3,math.floor(gt_h/self.mask_scale))
            min_w = max(math.floor(gt_boxes[0][i][0]/self.mask_scale)-3,0)
            max_w = min(math.ceil(gt_boxes[0][i][2]/self.mask_scale)+3,math.floor(gt_w/self.mask_scale))
            gt_mask_low[:, :, min_h:max_h, min_w:max_w] = 1 #x->w, y->h
        for i in range(num_boxes):
            min_h = max(math.floor(target_boxes[0][i][1]/self.mask_scale)-3,0)
            max_h = min(math.ceil(target_boxes[0][i][3]/self.mask_scale)+3,math.floor(ta_h/self.mask_scale))
            min_w = max(math.floor(target_boxes[0][i][0]/self.mask_scale)-3,0)
            max_w = min(math.ceil(target_boxes[0][i][2]/self.mask_scale)+3,math.floor(ta_w/self.mask_scale))
            #print(min_h,max_h,min_w,max_w)
            target_mask_low[:, :, min_h:max_h, min_w:max_w] = 1 #x->w, y->h
        
        gt_mask_low.detach()
        target_mask_low.detach()
        self.frame_id += 1
        return loss, gt_base_feature_low, gt_mask_low, target_base_feature_low, target_mask_low
##############################################################################
# Classes
##############################################################################
class TargetLoss(nn.Module):
    """Define target objectives.
    """
    def __init__(self, detection_type, detection_model, load_size, batch_size, gpu_ids=[]):
        super(TargetLoss, self).__init__()
        
        set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        cfg_file = "./Enhancement_models/target_loss/cfgs/{}.yml".format(detection_type)
        if cfg_file is not None:
            cfg_from_file(cfg_file)
        if set_cfgs is not None:
            cfg_from_list(set_cfgs)
        #print('=> Using config:')
        classes = ('__background__',  # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        # initilize the network here.
        if detection_type == 'vgg16':
            self.fasterRCNN = vgg16(classes, detection_type, pretrained=True, class_agnostic=False)
            self.mask_scale = 2
        elif detection_type == 'res101':
            self.fasterRCNN = resnet(classes, detection_type, 101, pretrained=True, class_agnostic=False)
            self.mask_scale = 4
        elif detection_type == 'res50':
            self.fasterRCNN = resnet(classes, detection_type, 50, pretrained=True, class_agnostic=False)
            print("===> Detection Network is not defined")
            pdb.set_trace()
        self.fasterRCNN.create_architecture()
        for param in self.fasterRCNN.parameters():
            param.requires_grad = False

        load_name = detection_model
        print("===> Loading faster-rcnn checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        self.fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
          cfg.POOLING_MODE = checkpoint['pooling_mode']
          print("===> Faster-rcnn POOLING_MODE %s" % (checkpoint['pooling_mode']))
        #print("===> Loaded faster-rcnn checkpoint %s" % (load_name))
        
        # initilize the variable.
        self.load_size = load_size
        batch_list = gpu_ids
        self.pooled_person_target = {str(i):0 for i in batch_list}
        self.pooled_car_target = {str(i):0 for i in batch_list}
        self.pooled_person_source = {str(i):0 for i in batch_list} 
        self.pooled_car_source = {str(i):0 for i in batch_list}
        #print('===> Loading device source in detection module',self.pooled_car_source)
        self.loss = nn.MSELoss()
        self.iteration = 0
        
    def forward(self, gt_data, target_data, im_info, gt_boxes, gt_num_boxes, target_boxes, num_boxes, culsum=False):
        
        #for i in range(len(unalign_boxes[0])):
        #    if unalign_boxes[0][i][4]:
        #        unalign_mask[int(unalign_boxes[0][i][1]):int(unalign_boxes[0][i][3]),int(unalign_boxes[0][i][0]):int(unalign_boxes[0][i][2])] = 1
        size = gt_data.size()
        gt_h, gt_w = size[2], size[3]
        #print(im_info)
        gt_im_info = torch.from_numpy(np.array([[gt_h, gt_w, im_info[0]]], dtype=np.float32)).type_as(gt_data) #[[h, w, im_info[0][0]],[h, w, im_info[1][0]]]
        gt_mask_low = torch.ones(size=(1, 1, math.floor(gt_h/self.mask_scale), math.floor(gt_w/self.mask_scale))).type_as(gt_data) * 0.3
        
        size = target_data.size()
        ta_h, ta_w = size[2], size[3]
        target_im_info = torch.from_numpy(np.array([[ta_h, ta_w, im_info[1]]], dtype=np.float32)).type_as(gt_data)
        target_mask_low = torch.ones(size=(1, 1, math.floor(ta_h/self.mask_scale), math.floor(ta_w/self.mask_scale))).type_as(gt_data) * 0.3
        
        gpu_id = torch.cuda.current_device()
        if num_boxes:
            target_boxes = target_boxes[0][:num_boxes.item()].unsqueeze(0)
            # target_boxes = torch.cat((target_boxes[0][:num_boxes[0].item()].unsqueeze(0), target_boxes[1][:num_boxes[1].item()].unsqueeze(0)), 0)  
        if gt_num_boxes:
            gt_boxes = gt_boxes[0][:gt_num_boxes.item()].unsqueeze(0)
            #gt_boxes = torch.cat((gt_boxes[0][:gt_num_boxes[0].item()].unsqueeze(0), gt_boxes[1][:gt_num_boxes[1].item()].unsqueeze(0)), 0)
        
        #print('===> num_boxes: ',num_boxes)
        #print('===> target_boxes: ',target_boxes)
        #culsum = False
        source_detect = culsum
        superdetection = True
        if num_boxes:
            self.fasterRCNN.train()
        else:
            self.fasterRCNN.train(False)
            culsum = True
            source_detect = True
            superdetection = False
        target_base_feature_low, target_detection_loss, self.pooled_person_target[str(gpu_id)], self.pooled_car_target[str(gpu_id)] = self.fasterRCNN(target_data, target_im_info, target_boxes, num_boxes, self.pooled_person_target[str(gpu_id)], self.pooled_car_target[str(gpu_id)], is_detect = True, detection_loss=superdetection, culsum_loss=culsum)    # 
        
        if gt_num_boxes:
            self.fasterRCNN.train()
        else:
            self.fasterRCNN.train(False)
        gt_base_feature_low, gt_detection_loss, self.pooled_person_source[str(gpu_id)], self.pooled_car_source[str(gpu_id)] = self.fasterRCNN(gt_data, gt_im_info, gt_boxes, gt_num_boxes, self.pooled_person_source[str(gpu_id)], self .pooled_car_source[str(gpu_id)], is_detect = source_detect, detection_loss=False, culsum_loss=culsum)
        for i in range(gt_num_boxes):
            min_h = max(math.floor(gt_boxes[0][i][1]/self.mask_scale)-3,0)
            max_h = min(math.ceil(gt_boxes[0][i][3]/self.mask_scale)+3,math.floor(gt_h/self.mask_scale))
            min_w = max(math.floor(gt_boxes[0][i][0]/self.mask_scale)-3,0)
            max_w = min(math.ceil(gt_boxes[0][i][2]/self.mask_scale)+3,math.floor(gt_w/self.mask_scale))
            gt_mask_low[:, :, min_h:max_h, min_w:max_w] = 1 #x->w, y->h
        for i in range(num_boxes):
            min_h = max(math.floor(target_boxes[0][i][1]/self.mask_scale)-3,0)
            max_h = min(math.ceil(target_boxes[0][i][3]/self.mask_scale)+3,math.floor(ta_h/self.mask_scale))
            min_w = max(math.floor(target_boxes[0][i][0]/self.mask_scale)-3,0)
            max_w = min(math.ceil(target_boxes[0][i][2]/self.mask_scale)+3,math.floor(ta_w/self.mask_scale))
            #print(min_h,max_h,min_w,max_w)
            target_mask_low[:, :, min_h:max_h, min_w:max_w] = 1 #x->w, y->h
        
        gt_mask_low.detach()
        target_mask_low.detach()
        
        self.iteration += 1
        if self.iteration>5000 and culsum:
            feature_culsum_loss = self.loss(self.pooled_person_source[str(gpu_id)], self.pooled_person_target[str(gpu_id)]) + self.loss(self.pooled_car_target[str(gpu_id)], self.pooled_car_source[str(gpu_id)])
            target_detection_loss += 10 * feature_culsum_loss
            if self.iteration % 200 == 4:
                print('===> feature_culsum_loss: ',1 * feature_culsum_loss)
        
        return target_detection_loss, gt_base_feature_low, gt_mask_low, target_base_feature_low, target_mask_low
##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real, mask = None, label = 1):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            #target_tensor = target_tensor*2 + label
            if mask is not None:
                #print('object',prediction.size())
                #print('mask',mask)
                (batch, channel, Ori_h, Ori_w) = prediction.size()
                mask = torch.nn.functional.interpolate(mask,size=(Ori_h, Ori_w), mode = 'bilinear')
                #print(mask)
                prediction = prediction * mask
                target_tensor = target_tensor * mask
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, norm, stride=1, padding=0, dilation=1, groups=1, use_bias=True, relu=True):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=use_bias)
        if padding:
            if isinstance(padding,tuple):
                self.padding = nn.ReflectionPad2d(padding=(padding[1],padding[1],padding[0],padding[0])) 
            else:
                self.padding = nn.ReflectionPad2d(padding=padding) 
        else:
            self.padding = None
        self.bn = norm(out_planes)
        self.relu = nn.LeakyReLU(0.2, True) if relu else None

    def forward(self, x):
        if self.padding:
            x = self.padding(x)
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, norm_layer, stride=1, scale = 1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        out_planes = in_planes
        inter_planes = in_planes //8

        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, norm = norm_layer, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, norm = norm_layer, stride=1, padding=1, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, norm = norm_layer, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(3,1), norm = norm_layer, stride=1, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, norm = norm_layer, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, norm = norm_layer, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1,3), norm = norm_layer, stride=stride, padding=(0,1)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, norm = norm_layer, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, norm = norm_layer, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(3,1), norm = norm_layer, stride=1, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, norm = norm_layer, stride=1, padding=5, dilation=5, relu=False)
                )
        self.branch4 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, norm = norm_layer, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1,3), norm = norm_layer, stride=stride, padding=(0,1)),
                BasicConv(inter_planes, inter_planes, kernel_size=(1,3), norm = norm_layer, stride=stride, padding=(0,1)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, norm = norm_layer, stride=1, padding=5, dilation=5, relu=False)
                )
        self.ConvLinear = BasicConv(5*inter_planes, out_planes, kernel_size=1, stride=1, relu=False, norm = norm_layer)
        #self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        #self.norm = norm_layer(out_planes)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        out = torch.cat((x0,x1,x2,x3,x4),1)
        out = self.ConvLinear(out)
        #short = self.shortcut(x)
        out = out + x
        return out

class BasicRFB_b(nn.Module):

    def __init__(self, in_planes, norm_layer, stride=1, scale = 1):
        super(BasicRFB_b, self).__init__()
        self.scale = scale
        out_planes = in_planes
        inter_planes = in_planes //8

        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, norm = norm_layer, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, norm = norm_layer, stride=1, padding=1, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, norm = norm_layer, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(3,1), norm = norm_layer, stride=1, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, norm = norm_layer, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, norm = norm_layer, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1,3), norm = norm_layer, stride=stride, padding=(0,1)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, norm = norm_layer, stride=1, padding=3, dilation=3, relu=False)
                )
        self.ConvLinear = BasicConv(3*inter_planes, out_planes, kernel_size=1, stride=1, relu=False, norm = norm_layer)
        #self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        #self.norm = norm_layer(out_planes)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        #short = self.shortcut(x)
        out = out + x
        return out

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], use_parallel = True, learn_residual = False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.defog_model = Defogging_Net(64) #new dehaze model
        for param in self.defog_model.parameters():
            param.requires_grad = False

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        use_bias = True
        
        model_first = [nn.ReflectionPad2d(1),
                 nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.LeakyReLU(0.2, True)]

        n_downsampling = 1
        for i in range(n_downsampling):
            mult = 2**(i+1)
            model_trunk = [nn.ReflectionPad2d(1),
                 nn.Conv2d(ngf, ngf * mult, kernel_size=3,
                                stride=2, padding=0, bias=use_bias),
                      norm_layer(ngf * mult),
                      nn.LeakyReLU(0.2, True)]

        mult = 2**n_downsampling
        #model_trunk = []
        for i in range(int(n_blocks)):
            model_trunk += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        
        #for i in range(int(n_blocks/6)):
        #    model_trunk += [BasicRFB_a(ngf * mult, norm_layer=norm_layer)]
            
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            #model_trunk += [nn.ReflectionPad2d(1),
            #     nn.Conv2d(ngf * mult, ngf * mult * 2, 3, padding=0),
            #          nn.PixelShuffle(2),
            #          norm_layer(int(ngf * mult / 2)),
            #          nn.LeakyReLU(0.2, True)]
            model_trunk += [nn.ReflectionPad2d(1),
                 nn.Conv2d(ngf * mult, int(ngf * mult / 2), 3, padding=0),
                      nn.Upsample(scale_factor=2),
                      norm_layer(int(ngf * mult / 2)),
                      nn.LeakyReLU(0.2, True)]

        model = [nn.ReflectionPad2d(1)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=3, padding=0)]
        model += [nn.Tanh()]
        
        self.model_first = nn.Sequential(*model_first)
        self.model_trunk = nn.Sequential(*model_trunk)
        self.model = nn.Sequential(*model)
    def forward(self, e_input, r_type='train'):
        """Standard forward"""
        if r_type == 'test':
            e_input, loss = self.defog_model(e_input, e_input, False) 
            e_input -= 0.5
        output = self.model_first(e_input) 
        output = output + self.model_trunk(output) 
        output = self.model(output)
        if r_type == 'test':
            output.clamp_(min=-0.5,max=0.5)
        return output

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.LeakyReLU(0.2, True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        self.dehaze_model = Net(8, 6)
        for param in self.dehaze_model.parameters():
            param.requires_grad = False
            
        curr_input_size = [2,2]
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True,input_size=curr_input_size)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            curr_input_size = [x *2 for x in curr_input_size]
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout,input_size=curr_input_size)
        # gradually reduce the number of filters from ngf * 8 to ngf
        curr_input_size = [x *2 for x in curr_input_size]

        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,input_size=curr_input_size)

        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer,input_size=curr_input_size)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer,input_size=curr_input_size)
        curr_input_size = [x *2 for x in curr_input_size]
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer,input_size=curr_input_size)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        input,loss = self.dehaze_model(input, input)
        input.clamp_(max=1)
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,input_size=[64,64]):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        #downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        #upnorm =norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        use_bias = True
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                #norm_layer(ndf* nf_mult),
                nn.LeakyReLU(0.2, True)  #ndf * nf_mult
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            #norm_layer(ndf* nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)] # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
