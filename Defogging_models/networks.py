
# coding: utf-8
# In[16]:
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import numpy as np 
import torch.nn.init as init
import random
from Defogging_models import pytorch_msssim
from Defogging_models import pytorch_ssim
from torchvision.models import vgg16
from Defogging_models.ECLoss import DCLoss
from .L1_TVLoss_dif import L1_TVLoss_Charbonnier_diff

def xavier(param):
    init.xavier_uniform_(param)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False)
        self.fc2   = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.relu(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.relu(x)

class BottleBlock(nn.Module):
    def __init__(self, planes):
        super(BottleBlock, self).__init__()
        self.planes = planes*1
        self.ca = ChannelAttention(self.planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.ca(x) * x
        out = self.sa(out) * out
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv_block = self.build_conv_block(out_channels)

    def build_conv_block(self, dim):
        conv_block = []
        p = 0
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       nn.LeakyReLU(0.2, True)]
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x) * self.res_scale
        return out
        
class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = vgg16(pretrained=True).features
        vgg_pretrained_features = vgg_pretrained_features[:-1]
        #checkpoint = torch.load('/data2/GQ/faster-rcnn.pytorch-pytorch-1.0/data/pretrained_model/faster_vgg16.pth')
        #vgg_pretrained_features.load_state_dict(checkpoint)
        #if not requires_grad:
        #    for param in vgg_pretrained_features.parameters():
        #        param.requires_grad = False
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h = self.slice2(h)
        return h
        
class Defogging_Net(nn.Module):
    def __init__(self,dim):
        super(Defogging_Net,self).__init__()
        self.upsample = F.interpolate
        self.relu = nn.LeakyReLU(0.2, True)
        self.tanh = nn.Tanh()
        self.conv1 = nn.Conv2d(3,dim,kernel_size=3, padding=1,bias=True)
        inChannels = dim

        self.dense1 = ResBlock(inChannels,inChannels)
        self.dense2 = ResBlock(inChannels,inChannels)
        self.dense3 = ResBlock(inChannels,inChannels)
        self.dense4 = ResBlock(inChannels,inChannels)
        self.dense5 = ResBlock(inChannels,inChannels)
        self.dense6 = ResBlock(inChannels,inChannels)
        self.dense7 = ResBlock(inChannels,inChannels)
        self.dense8 = ResBlock(inChannels,inChannels)
                
        self.Bottleneck1 = BottleBlock(inChannels)
        self.Bottleneck2 = BottleBlock(inChannels)
        self.Bottleneck3 = BottleBlock(inChannels)
        self.Bottleneck4 = BottleBlock(inChannels)
            
        self.stride_conv1= nn.Conv2d(inChannels, inChannels, kernel_size=3,stride=2,padding=1)
        self.stride_conv2= nn.Conv2d(inChannels, inChannels, kernel_size=3,stride=2,padding=1)
        self.stride_conv3= nn.Conv2d(inChannels, inChannels, kernel_size=3,stride=2,padding=1)
        self.convt1 = nn.ConvTranspose2d(in_channels=inChannels, out_channels=inChannels, kernel_size=4, stride=2, padding=1, bias=True)
        self.convt2 = nn.ConvTranspose2d(in_channels=inChannels, out_channels=inChannels, kernel_size=4, stride=2, padding=1, bias=True)
        self.convt3 = nn.ConvTranspose2d(in_channels=inChannels, out_channels=inChannels, kernel_size=4, stride=2, padding=1, bias=True)
        
        self.refine0= nn.Conv2d(3, inChannels, kernel_size=3,stride=1,padding=1)
        self.refine1= nn.Conv2d(inChannels, inChannels, kernel_size=3,stride=1,padding=1)
        self.refine2= nn.Conv2d(inChannels*2, 3, kernel_size=3,stride=1,padding=1)
        self.refine3= nn.Conv2d(3, inChannels, kernel_size=3,stride=1,padding=1)
        self.refine4= nn.Conv2d(inChannels, 3, kernel_size=3,stride=1,padding=1)
        self.refine5= nn.Conv2d(inChannels*2, 3, kernel_size=3,stride=1,padding=1)

        self.conv1010 = nn.Conv2d(inChannels, int(inChannels/4), kernel_size=1,stride=1,padding=0)  
        self.conv1020 = nn.Conv2d(inChannels, int(inChannels/4), kernel_size=1,stride=1,padding=0)
        self.conv1030 = nn.Conv2d(inChannels, int(inChannels/4), kernel_size=1,stride=1,padding=0)
        self.conv1040 = nn.Conv2d(inChannels, int(inChannels/4), kernel_size=1,stride=1,padding=0)
        self.conv2010 = nn.Conv2d(inChannels, int(inChannels/4), kernel_size=1,stride=1,padding=0)  
        self.conv2020 = nn.Conv2d(inChannels, int(inChannels/4), kernel_size=1,stride=1,padding=0)
        self.conv2030 = nn.Conv2d(inChannels, int(inChannels/4), kernel_size=1,stride=1,padding=0)
        self.conv2040 = nn.Conv2d(inChannels, int(inChannels/4), kernel_size=1,stride=1,padding=0)
        
        self.TVLoss = L1_TVLoss_Charbonnier_diff()
        self.feature = Vgg16(requires_grad=False).cuda()
        self.ssim1 = pytorch_ssim.SSIM(window_size = 9).cuda()
        #self.ssim1 = pytorch_msssim.MSSSIM(window_size = 11).cuda()
        self.criterion = nn.MSELoss(reduction='mean').cuda()
        self.criterion1 = nn.L1Loss(reduction='mean').cuda()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
        init.constant_(self.refine2.bias, 1) 
        self.iteration = 0
        
    def gradient(self,y):
        gradient_h=torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
        gradient_y=torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])

        return gradient_h, gradient_y
        
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
        
    def forward(self,x,target,train=True):
        out0 = self.relu(self.conv1(x))
        out1 = self.dense1(out0)
        out2 = self.relu(self.stride_conv1(out1))
        out2 = self.dense2(out2)
        out3 = self.relu(self.stride_conv1(out2))
        out3 = self.dense3(out3)                                   
        out3 = self.Bottleneck1(out3)

        out3 = self.dense4(out3)
        out = self.relu(self.convt1(out3))
        out2 = self.Bottleneck2(out2)
        out2 = out2 + out
        
        out2 = self.dense5(out2)
        out = self.relu(self.convt2(out2))
        out1 = self.Bottleneck3(out1)
        out1 = out1 + out
        
        out1 = self.dense6(out1)
        #out = self.relu(self.refine0(x))
        out0 = self.Bottleneck4(out0)
        out = out0 + out1 
        
        defog = self.relu(self.refine1(out))
        shape_out = defog.data.size()
        shape_out = shape_out[2:4]
        x101 = F.avg_pool2d(defog, 8)
        x102 = F.avg_pool2d(defog, 4)
        x103 = F.avg_pool2d(defog, 2)
        x104 = F.avg_pool2d(defog, 1)
        x1010 = self.upsample(self.relu(self.conv1010(x101)),size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)),size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)),size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)),size=shape_out)
        defog = torch.cat((x1010, x1020, x1030, x1040, defog), 1)
        defog = self.tanh(self.refine2(defog))
        defog = self.relu(defog*x-defog+1)
        defog = self.relu(self.refine3(defog))
        defog = self.dense7(defog)
        defog = self.relu(self.refine4(defog))
        
        out = self.dense8(out)
        shape_out = out.data.size()
        shape_out = shape_out[2:4]
        x101 = F.avg_pool2d(out, 16)
        x102 = F.avg_pool2d(out, 8)
        x103 = F.avg_pool2d(out, 4)
        x104 = F.avg_pool2d(out, 2)
        x1010 = self.upsample(self.relu(self.conv2010(x101)),size=shape_out)
        x1020 = self.upsample(self.relu(self.conv2020(x102)),size=shape_out)
        x1030 = self.upsample(self.relu(self.conv2030(x103)),size=shape_out)
        x1040 = self.upsample(self.relu(self.conv2040(x104)),size=shape_out)
        out = torch.cat((x1010, x1020, x1030, x1040, out), 1)
        A = self.tanh(self.refine5(out))

        out = self.relu(defog*A)
        loss = 0
        if train:
            #loss1 = 5*self.criterion1(self.feature(out),self.feature(target))
            loss0 = self.TVLoss(out, target)
            loss1 = 0.01*DCLoss(out, 35) 
            loss2 = 0.1*(1-self.ssim1(out,target))
            loss3 = 1 *self.criterion(out, target)
            loss4 = 0.5 *self.criterion1(out, target)
            loss =  loss0 + loss1 + loss2 + loss3 + loss4
            if self.iteration % 50 == 0:
                print("loss0:{:.4f}, loss1:{:.4f}, loss2:{:.4f}, loss3:{:.4f}, loss4:{:.4f}".format(loss0, loss1,loss2,loss3,loss4))
            self.iteration += 1
        return out, loss

