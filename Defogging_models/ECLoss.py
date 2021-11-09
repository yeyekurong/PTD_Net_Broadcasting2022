import torch
import numpy as np
from PIL import Image
import math
import torch.nn as nn
from torch.nn import L1Loss, MSELoss
from torch.autograd import Variable
from torchvision import transforms
import pdb
import torch.nn.functional as F

class GradLayer(nn.Module):

    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]]
        kernel_h = [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def get_gray(self,x):
        ''' 
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        # x_list = []
        # for i in range(x.shape[1]):
        #     x_i = x[:, i]
        #     x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
        #     x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
        #     x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
        #     x_list.append(x_i)

        # x = torch.cat(x_list, dim=1)
        if x.shape[1] == 3:
            x = self.get_gray(x)

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x

class GradLoss(nn.Module):

    def __init__(self):
        super(GradLoss, self).__init__()
        self.grad_layer = GradLayer()

    def forward(self, x):
        output_grad = self.grad_layer(x)
        
        return 1 - torch.mean(output_grad).detach()
        
def L1_TVLoss(x):
    selfe = 0.000001 ** 2
    batch_size = x.size()[0]
    h_tv = torch.abs((x[:, :, 1:, :]-x[:, :, :-1, :]))
    #h_tv = h_tv*torch.exp(-1*h_tv)
    h_tv = torch.mean(torch.sqrt(h_tv ** 2 + selfe))
    w_tv = torch.abs((x[:, :, :, 1:]-x[:, :, :, :-1]))
    #w_tv = w_tv*torch.exp(-1*w_tv)
    w_tv = torch.mean(torch.sqrt(w_tv ** 2 + selfe))
    loss = 0.5*(h_tv + w_tv)
    return loss
        
def EntropyLoss(img):
    img_ = img+0.5
    img_ = 1/3*(img_[:,0,:,:] + img_[:,1,:,:] + img_[:,2,:,:])
    #print(img_.size())
    b, x, y = img_.size()
    loss_entorphy = 0
    for batch in range(b):
        tmp = torch.histc(img_[batch,:,:],256,0,1)
        tmp = tmp/(x*y)
        tmp = tmp.cpu()
        res = 0
        for i in range(len(tmp)):
            if(tmp[i] > 0):
                loss_entorphy += float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    loss_entorphy = loss_entorphy/b
    #print(8-loss_entorphy)
    return 8-loss_entorphy
    
def ColorLoss(img):
    tensor = img[:,0,:,:].view(img.size(0),-1)
    r_mean = tensor.mean(1)
    tensor = img[:,1,:,:].view(img.size(0),-1)
    g_mean = tensor.mean(1)
    tensor = img[:,2,:,:].view(img.size(0),-1)
    b_mean = tensor.mean(1)
    #print(r_mean)
    #print(g_mean)
    loss_color = 0
    for i in range(img.size(0)):
        color_list = [r_mean[i],g_mean[i],b_mean[i]] 
        max_index = color_list.index(max(color_list))
        min_index = color_list.index(min(color_list))
        mean_index = 3 - max_index - min_index
        loss_color += color_list[max_index] - color_list[mean_index]
    return loss_color/img.size(0)


def DCLoss(img, patch_size):
    """
    calculating dark channel of image, the image shape is of N*C*W*H
    """
    maxpool = nn.MaxPool3d((3, patch_size, patch_size), stride=1, padding=(0, patch_size//2, patch_size//2))
    dc = maxpool(0-img[:, None, :, :, :])
    
    target = Variable(torch.FloatTensor(dc.shape).zero_().cuda()) 
     
    loss = L1Loss(size_average=True)(-dc, target)
    return loss

def BCLoss(img, patch_size):
    """
    calculating bright channel of image, the image shape is of N*C*W*H
    """
    patch_size = 35
    dc = maxpool(img[:, None, :, :, :])
    
    target = Variable(torch.FloatTensor(dc.shape).zero_().cuda()+1) 
    loss = L1Loss(size_average=False)(dc, target)
    return loss
    
if __name__=="__main__":
    img = Image.open('clear_img.jpg')
    totensor = transforms.ToTensor()
    
    img = totensor(img)
    
    img = Variable(img[None, :, :, :].cuda(), requires_grad=True)    
    loss = DCLoss(img, 35)
    
    # loss.backward()



    



