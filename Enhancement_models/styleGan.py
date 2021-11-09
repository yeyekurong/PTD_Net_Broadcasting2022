import torch
import itertools
from . import networks
import torch.nn as nn
import random
import numpy as np
class StyleGAN_G(nn.Module):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    def __init__(self, state):
        super(StyleGAN_G,self).__init__()
        self.isTrain = state
        self.netG_A = networks.define_G(3, 3, 64, 'resnet_9blocks', 'instance',
                                  False, 'normal', 0.02, [0])
        load_filename = 'cwgan/%s_net_%s.pth' % ('100', 'G_B') 
        state_dict = torch.load(load_filename)
        #self.netG_A.load_state_dict(state_dict)
        self.netG_A.load_state_dict({k[7:]:v for k,v in state_dict.items() })
        print('Loading pretrained weights from',load_filename)
        if not self.isTrain:
            for param in self.netG_A.parameters():
                param.requires_grad = False
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.

    def forward(self, im_data):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        Ori_h = im_data.size()[2]
        Ori_w = im_data.size()[3]
        
        h = int(round(Ori_h / 32) * 32)
        w = int(round(Ori_w / 32) * 32)
        if Ori_h != int(h/2) or Ori_w != int(w/2):
            im_data = torch.nn.functional.interpolate(im_data,size=(int(h/2), int(w/2)), mode = 'bilinear')
            
        #real_B = self.netG_A((im_data)/255,'dehaze')-0.5
        fake_A = (im_data/255) - 0.5
        fake_A = self.netG_A(fake_A)  # G_B(B)
        
        return fake_A
class ObjectGAN_G(nn.Module):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    def __init__(self, state):
        super(ObjectGAN_G,self).__init__()
        self.isTrain = state
        self.netG_A = networks.define_G(3, 3, 64, 'resnet_9blocks', 'instance',
                                  False, 'normal', 0.02, [0])
        load_filename = 'cwgan/%s_net_%s.pth' % ('100', 'G_B') 
        state_dict = torch.load(load_filename)
        #self.netG_A.load_state_dict(state_dict)
        self.netG_A.load_state_dict({k[7:]:v for k,v in state_dict.items() })
        print('Loading pretrained weights from',load_filename)
        self.criterionIdt = torch.nn.MSELoss()
        #self.idtloss = torch.Tensor([1.2201, 1.0607, 1.0798]) 
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.

    def forward(self, im_data, adaption_data, gt_boxes, unalign_boxes, gt_h, gt_w, style='unsupervised'):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        Ori_h = im_data.size()[2]
        Ori_w = im_data.size()[3]
        
        h = int(round(Ori_h / 32) * 32)
        w = int(round(Ori_w / 32) * 32)
        if Ori_h != int(h/2) or Ori_w != int(w/2):
            im_data = torch.nn.functional.interpolate(im_data,size=(int(h/2), int(w/2)), mode = 'bilinear')
            
        #real_B = self.netG_A((im_data)/255,'dehaze')-0.5
        fake_A = (im_data/255) - 0.5
        fake_A = self.netG_A(fake_A)  # G_B(B)
        
        if(style=='unsupervised'):
            unalign_mask = torch.zeros(Ori_h,Ori_w)
            gt_mask = torch.zeros(gt_h,gt_w)
        else:
            gt_mask = torch.zeros(Ori_h,Ori_w)
            unalign_mask = torch.zeros(gt_h,gt_w)
        for i in range(20):
            classes = gt_boxes[0][i][4]
            if classes == 7 or classes == 15:
                #if i % 3 == 0 or i % 3 == 2:
                #print('object',gt_boxes[0][i][1],gt_boxes[0][i][3],gt_boxes[0][i][0],gt_boxes[0][i][2])
                gt_mask[int(gt_boxes[0][i][1]):int(gt_boxes[0][i][3]),int(gt_boxes[0][i][0]):int(gt_boxes[0][i][2])] = 1
        #print(source_boxes.size(),len(source_boxes[0]))
        for i in range(len(unalign_boxes[0])):
            #print('object',gt_boxes[0][i][1],gt_boxes[0][i][3],gt_boxes[0][i][0],gt_boxes[0][i][2])
            if unalign_boxes[0][i][4]:
                unalign_mask[int(unalign_boxes[0][i][1]):int(unalign_boxes[0][i][3]),int(unalign_boxes[0][i][0]):int(unalign_boxes[0][i][2])] = 1
        gt_mask = gt_mask.unsqueeze(0).unsqueeze(0).cuda()
        unalign_mask = unalign_mask.unsqueeze(0).unsqueeze(0).cuda()
        #fake_A2 = (fake_A+0.5)*255
        if Ori_h != int(h/2) or Ori_w != int(w/2):
            fake_A2 = torch.nn.functional.interpolate((fake_A+0.5)*255,size=(Ori_h, Ori_w), mode = 'bilinear')
        
        Idt_loss = 200 * self.criterionIdt(fake_A, adaption_data) #MSE 200 
            #loss_G_A = self.criterionGAN(self.netD_A(fake_A_trans), True)
        return fake_A2, gt_mask, unalign_mask, Idt_loss

class StyleGAN_D(nn.Module):      
    def __init__(self, in_c, state, gpu_ids):  
        super(StyleGAN_D,self).__init__()
        self.netD_A = networks.define_D(in_c, 64, 'n_layers',
                                            1, 'batch', 'normal', 0.02, gpu_ids)
        self.criterionGAN = networks.GANLoss('lsgan').to(gpu_ids[0])
    def forward(self, real, fake, real_mask, fake_mask, state):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
            mask (tensor array) -- mask for detection the foreground
            label               -- object type. 0 denotes person, 1 denotes car
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        #print('fake',fake)
        #print('real',real.size())
        pred_fake = self.netD_A(fake)
        #pred_fake = pred_fake * (1 + mask)
        loss_D_fake = self.criterionGAN(pred_fake, state, fake_mask)
        loss_D = loss_D_fake
        if not state:
          pred_real = self.netD_A(real)
          #pred_fake = pred_fake * (1 + attention)
          loss_D_real = self.criterionGAN(pred_real, True, real_mask)
          loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D
        
        
        
        
        
        
        
        
        