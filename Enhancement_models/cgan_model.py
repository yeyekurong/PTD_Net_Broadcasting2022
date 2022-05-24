import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import cv2
import numpy
from .target_loss.model.utils.config import cfg
from .styleGan import StyleGAN_D
import torch.nn.functional as F
import copy
import torch.nn as nn
from collections import OrderedDict
from Defogging_models.ECLoss import L1_TVLoss, ColorLoss, MSCNLoss


class CGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=5.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=5.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_B', 'G_B', 'idt_B', 'trans_B', 'TV']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        #self.opt = opt
        if self.isTrain:
            visual_names_A = ['real_A']
            visual_names_B = ['real_B', 'fake_A']
        else:
            visual_names_A = []
            visual_names_B = ['fake_A']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize G_B(A) ad G_A(B)
            #visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain and int(opt.load_iter) < (int(opt.init_epoch)):
            self.model_names = ['G_B', 'D_B']
        elif self.isTrain and int(opt.load_iter) > int(opt.refine_epoch)-1:           
            self.model_names = ['G_B','D_B_low','D_B']
        elif self.isTrain and int(opt.load_iter) > int(opt.init_epoch)-1:
            self.model_names = ['G_B','D_B_low']
            #self.model_names = ['G_B']
        else:
            self.model_names = ['G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        #self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                                not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.defog_model)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.defog_model)
        #print(self.netG_B)
        if self.isTrain:  # define discriminators
            #self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
            #                                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            if opt.target_type == 'fasterrcnn':
                if opt.detection_type == 'vgg16':
                    self.netD_B_low = StyleGAN_D(128, True, self.gpu_ids)
                elif opt.detection_type == 'res101': 
                    self.netD_B_low = StyleGAN_D(256, True, self.gpu_ids)
            else:
                self.netD_B_low = StyleGAN_D(64, True, self.gpu_ids)
            #self.netD_B_high = StyleGAN_D(512, True, self.gpu_ids)
            
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionTrans1 = torch.nn.L1Loss()
            self.criterionTrans2 = torch.nn.MSELoss()
            self.mscnloss = MSCNLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            #self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_B = torch.optim.Adam(self.netG_B.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B_low = torch.optim.Adam(self.netD_B_low.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            #self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G_B)
            self.optimizers.append(self.optimizer_D_B)        
        self.bias = torch.Tensor(cfg.PIXEL_MEANS)
        self.bias = self.bias.squeeze(0).unsqueeze(2).unsqueeze(3).to(self.device)
        self.init_first = True
        self.refine_first = True
        
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths']
        
        self.A_boxes = input['A_boxes'].to(self.device)
        self.B_boxes = input['B_boxes'].to(self.device)
        self.A_num_box = input['A_num_box'].to(self.device)
        self.B_num_box = input['B_num_box'].to(self.device)
        self.info = input['im_info']
        
    def forward(self, critic_iters=0, init_epoch=0, refine_epoch=0, r_type='train'):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if r_type == 'train' and critic_iters < init_epoch:
            self.real_B = self.real_B - 0.5
            self.real_A = self.real_A - 0.5
            #self.fake_B = self.netG_A(self.real_A)  # G_A(A)
            #self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
            self.fake_A = self.netG_B(self.real_B)  # G_B(B)
            #self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        elif r_type == 'train':
            self.real_B = self.real_B - 0.5
            self.real_A = self.real_A - 0.5
            self.fake_A_copy = self.netG_B_copy(self.real_B).detach()
            self.fake_A = self.netG_B(self.real_B)
            #self.fake_B = self.netG_B(self.fake_B)
        elif r_type == 'Etest':
            #r_type = 'Etest'
            print(r_type)
            self.real_B = self.real_B - 0.5
            self.real_A = self.real_A - 0.5
            if self.opt.direction == 'AtoB':
                self.fake_A = self.netG_B(self.real_B, r_type)  
            else:
                self.fake_A = self.netG_A(self.real_B, r_type)              
        else:
            if self.opt.direction == 'AtoB':
                self.fake_A = self.netG_B(self.real_B, r_type)  
            else:
                self.fake_A = self.netG_A(self.real_B, r_type)  
            self.real_B = self.real_B - 0.5
            self.real_A = self.real_A - 0.5
        
    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real.detach())
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        if torch.isnan(loss_D):
            loss_D = torch.Tensor([0])
        else:    
            #self.loss_G.backward()
            loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self, detection_flag=1):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A, detection_flag)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_D_B_Low(self, detection_flag=1):
        """Calculate GAN loss for discriminator D_B"""
        self.gt_feature_low = self.gt_feature_low.detach()
        self.target_feature_low = self.target_feature_low.detach()
        self.loss_D_B_low = self.netD_B_low(self.gt_feature_low, self.target_feature_low, self.gt_mask_low, self.target_mask_low, False)
        # print("6:{}".format(torch.cuda.memory_allocated(0)))
        self.loss_D_B_low.backward()
    
    def backward_D_B_High(self, detection_flag=1):
        """Calculate GAN loss for discriminator D_B"""
        self.gt_feature_high = self.gt_feature_high.detach()
        self.target_feature_high = self.target_feature_high.detach()
        self.loss_D_B_high = self.netD_B_high(self.gt_feature_high, self.target_feature_high, self.gt_mask_high, self.target_mask_high, False)
        # print("7:{}".format(torch.cuda.memory_allocated(0)))
        self.loss_D_B_high.backward()
        
    def gradient(self,y):
        #gradient_h[:, :, :, :-1]=torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
        gradient_y=torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])
        return gradient_y
    
    def rgb_hsv(self, img):
        saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + 0.5 + 1e-8)
        return saturation
           
    def backward_single_G(self, refine, target_type):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        self.loss_G_A = 0
        self.loss_cycle_A = 0
        self.loss_cycle_B = 0
        self.loss_idt_A = 0
        self.loss_idt_B = 0 
        lambda_B = self.opt.lambda_B
        """Calculate the loss for generators G_B"""
        self.loss_TV = L1_TVLoss(self.fake_A-self.fake_A_copy) * float(5)
        #self.loss_trans_A = self.criterionTrans2(self.fake_B, self.real_A) * 20
        if target_type == 'fasterrcnn':
            if self.B_num_box == 0:
                self.loss_trans_B = self.criterionTrans2(self.fake_A, self.fake_A_copy) * 100 + self.criterionTrans1(self.fake_A.max(1)[0], self.fake_A_copy.max(1)[0]) * (25)
            else:
                self.loss_trans_B = self.criterionTrans2(self.fake_A, self.fake_A_copy) * 100 + self.criterionTrans1(self.fake_A.max(1)[0], self.fake_A_copy.max(1)[0]) * (80)
            size = self.real_A.size()
            im_bias = self.bias.expand(1, -1, size[2], size[3])
            real_A = (self.real_A[:,[2,1,0],:,:]+0.5)*255 - im_bias
            size = self.fake_A.size()
            im_bias = self.bias.expand(1, -1, size[2], size[3])
            fake_A = (self.fake_A[:,[2,1,0],:,:]+0.5)*255 - im_bias
            self.loss_target, self.gt_feature_low, self.gt_mask_low, self.target_feature_low, self.target_mask_low = self.criterionTarget(real_A, fake_A, self.info, self.A_boxes, self.A_num_box, self.B_boxes, self.B_num_box, self.opt.feature_culsum) #
        else:
            self.loss_trans_B = self.criterionTrans2(self.fake_A, self.fake_A_copy) * 10 + self.criterionTrans1(self.fake_A.max(1)[0], self.fake_A_copy.max(1)[0]) * (8)
            real_A = (self.real_A+0.5)
            fake_A = (self.fake_A+0.5)
            self.loss_target, self.gt_feature_low, self.gt_mask_low, self.target_feature_low, self.target_mask_low = self.criterionTarget(real_A, fake_A, self.A_boxes, self.A_num_box, self.B_boxes, self.B_num_box) # self.loss_gt
        
        #print(self.B_boxes)
        #from torchvision import utils
        #print(self.target_mask_low)
        #utils.save_image(self.target_mask_low, 'aaaaaaaaaaaaaaa.png')
        
        #self.loss_gt = 0.3 * self.loss_gt.mean()
        #print(self.A_boxes[0][:][4])
        #self.idt_B = self.netG_B(self.real_A)
        #self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        
        if not isinstance(self.loss_target,int):
            if self.B_num_box == 0:
                self.loss_target =  4 * self.loss_target.mean()
            else:
                scale = 8 if(target_type == 'yolo') else 1
                self.loss_target =  self.loss_target.mean() * 0.5 * scale
        self.loss_G_B_low = 0.1 * self.netD_B_low(self.gt_feature_low, self.target_feature_low, self.gt_mask_low, self.target_mask_low, True)
        self.loss_grad = L1_TVLoss(self.fake_A) * float(15) 
        self.loss_color = 40 * ColorLoss(self.fake_A)
        self.loss_mscn = self.mscnloss(self.fake_A)*0.1
        
        if refine:
            fake_A_sample = F.interpolate(self.fake_A, scale_factor=0.5, mode="bilinear")
            self.loss_G_B = 0.1  *(self.criterionGAN(self.netD_B(self.fake_A), True) + self.criterionGAN(self.netD_B(fake_A_sample), True))
            self.loss_G = self.loss_target + self.loss_G_B_low + self.loss_trans_B + self.loss_TV - self.loss_grad + self.loss_G_B #+ self.loss_idt_B
        else:
            self.loss_G = self.loss_target + self.loss_G_B_low + self.loss_trans_B + self.loss_TV - self.loss_grad + self.loss_color + self.loss_mscn#+ self.loss_DC #+ self.loss_idt_B #+  self.loss_G_B_high self.loss_gt
        
        if torch.isnan(self.loss_G) or self.loss_trans_B > 40:
            self.loss_G = torch.Tensor([0])
            print('===> Nan',self.loss_target, self.loss_G_B_low, self.loss_trans_B, self.loss_TV)
        else:
            self.loss_G.backward()
        return True        

    def optimize_parameters(self, critic_iters, init_epoch, refine_epoch, load_path, target_type):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        if self.init_first and critic_iters >= init_epoch:
            self.netG_B_copy = copy.deepcopy(self.netG_B)
            if critic_iters > init_epoch:
                print('===> Loading the model NetG_B_copy from %s' % load_path)
                state_dict = torch.load(load_path)
                if len(self.gpu_ids) == 1: 
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        if k[:6] == 'module':
                            k = k[7:] # remove `module.`
                        new_state_dict[k] = v
                    self.netG_B_copy.load_state_dict(new_state_dict)
            for param in self.netG_B_copy.parameters():
                param.requires_grad = False 
        
        self.forward(critic_iters, init_epoch, refine_epoch)      # compute fake images and reconstruction images.
        if critic_iters < init_epoch:
            # G_A and G_B
            self.set_requires_grad([self.netD_B], False)  # Ds require no gradients when optimizing Gs
            self.optimizer_G_B.zero_grad()  # set G_A and G_B's gradients to zero
            flag = self.backward_G()             # calculate gradients for G_A and G_B
            self.optimizer_G_B.step()       # update G_A and G_B's weights
            # D_A and D_B
            self.set_requires_grad([self.netD_B], True)
            self.optimizer_D_B.zero_grad()   # set D_A and D_B's gradients to zero
            #self.backward_D_A()      # calculate gradients for D_A
            self.backward_D_B()      # calculate graidents for D_B
            self.optimizer_D_B.step()  # update D_A and D_B's weights
        else:
            if self.init_first:
                self.init_first = False
                #del self.netG_A
                #del self.netD_A
                #del self.netD_B
                if target_type == 'fasterrcnn':
                    self.criterionTarget = networks.TargetLoss(self.opt.detection_type, self.opt.detection_model, self.opt.load_size, self.opt.batch_size, self.gpu_ids).to(self.device)  # define Target loss.
                else:
                    self.criterionTarget = networks.TargetLoss_yolo(self.opt.load_size, self.device).to(self.device)  # define Target loss.
                if len(self.gpu_ids)>1:
                    self.criterionTarget = torch.nn.DataParallel(self.criterionTarget, self.gpu_ids)
                self.visual_names = ['real_A', 'real_B', 'fake_A', 'fake_A_copy']
                self.loss_names = ['target', 'G_B_low', 'trans_B', 'TV', 'D_B_low', 'grad']
                self.model_names = ['G_B','D_B_low']
            self.set_requires_grad([self.netD_B, self.netD_B_low], False)
            self.optimizer_G_B.zero_grad() 
            flag = self.backward_single_G(critic_iters >= refine_epoch, target_type)  
            self.optimizer_G_B.step()       # update G_A and G_B's weights
            
            self.set_requires_grad([self.netD_B_low], True)
            self.optimizer_D_B_low.zero_grad()   # set D_A and D_B's gradients to zero
            self.backward_D_B_Low(0)      # calculate graidents for D_B
            self.optimizer_D_B_low.step()  # update D_A and D_B's weights
            if critic_iters >= refine_epoch:
                if self.refine_first:
                    self.loss_names = ['target', 'G_B_low', 'trans_B', 'TV', 'D_B_low','D_B']
                    self.model_names = ['G_B','D_B_low','D_B']
                    self.refine_first = False
                    if critic_iters == refine_epoch:
                        print('===> Loading the model NetD_B from %s' % load_path.replace("G_B", "D_B"))
                        state_dict = torch.load(load_path.replace("G_B", "D_B"))
                        if len(self.gpu_ids) == 1: 
                            new_state_dict = OrderedDict()
                            for k, v in state_dict.items():
                                if k[:6] == 'module':
                                    k = k[7:] # remove `module.`
                                new_state_dict[k] = v
                            self.netD_B.load_state_dict(new_state_dict)
                    #for param in self.netG_B.defog_model.parameters():
                    #    param.requires_grad = True
                    #print('===> Refining the defog model in netG_B')
                           
                self.set_requires_grad([self.netD_B], True)
                self.optimizer_D_B.zero_grad()
                self.backward_D_B()
                self.optimizer_D_B.step()
            