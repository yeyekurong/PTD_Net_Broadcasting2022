import torch
import torch.nn as nn

class L1_TVLoss_Charbonnier_diff(nn.Module):

    def __init__(self):

        super(L1_TVLoss_Charbonnier_diff, self).__init__()

        self.e = 0.000001 ** 2



    def forward(self, x, target):

        batch_size = x.size()[0]

        h_tv = torch.abs((x[:, :, 1:, :]-x[:, :, :-1, :]))
        h_tv2 = torch.abs((target[:, :, 1:, :]-target[:, :, :-1, :]))
        h_tv = torch.abs(h_tv - h_tv2)
        h_tv = torch.mean(torch.sqrt(h_tv ** 2 + self.e))

        w_tv = torch.abs((x[:, :, :, 1:]-x[:, :, :, :-1]))
        w_tv2 = torch.abs((target[:, :, :, 1:]-target[:, :, :, :-1]))
        w_tv = torch.abs(w_tv - w_tv2)
        w_tv = torch.mean(torch.sqrt(w_tv ** 2 + self.e))

        return h_tv + w_tv

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