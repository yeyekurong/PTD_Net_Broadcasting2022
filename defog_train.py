
# coding: utf-8

import argparse, os
import torch
import random
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
from Defogging_models.networks import Defogging_Net 
from data.unaligned_dataset import Unaligned_Defog_Dataset
from PIL import Image
from torchvision import transforms
import torchvision
import math
import torch.nn.init as init
import sys
import torch.nn.functional as F
import time
import warnings
warnings.filterwarnings("ignore")

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every step epochs"""
    lr = opt.lr * (0.9 ** (epoch // opt.step))
    return lr


def train(opt, device):
    cuda = opt.cuda
    opt.seed = random.randint(1, 10000)
    print("===> Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    criterion = nn.MSELoss(reduction = 'mean').cuda()
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    else:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        cudnn.benchmark = True
        torch.cuda.manual_seed(opt.seed)

    print("===> Loading datasets")
    defog_train_dataset = Unaligned_Defog_Dataset(hazy=opt.hazy_dir, gt=opt.gt_dir, train=True, transform=transforms.Compose([transforms.Resize((300,352), interpolation=Image.BICUBIC), transforms.ToTensor()]))
    defog_training_data_loader = DataLoader(defog_train_dataset, batch_size=opt.batchSize, num_workers=opt.threads, shuffle=True, pin_memory=True, drop_last=True)
    defog_test_dataset = Unaligned_Defog_Dataset(hazy=opt.symthesized_hazy_test, gt=opt.symthesized_gt_test, train=False, transform=transforms.Compose([transforms.Resize((400,512), interpolation=Image.BICUBIC), transforms.ToTensor()]))
    defog_testing_data_loader = DataLoader(defog_test_dataset, batch_size=1, num_workers=1, shuffle=False)

    print("===> Building model")
    defog_model = Defogging_Net(64)
    # optionally copy weights from a checkpoint
    pre_epoch = 0
    if opt.pretrained:
        if os.path.isfile('checkpoints/defog_model/'+opt.pretrained):
            print("=> loading model '{}'".format('checkpoints/defog_model/'+opt.pretrained))
            weights = torch.load('checkpoints/defog_model/'+opt.pretrained)
            pre_epoch = weights["epoch"]
            if isinstance(weights['defog_model'], torch.nn.DataParallel):
                defog_model.load_state_dict(weights['defog_model'].module.state_dict())
            else:
                defog_model.load_state_dict(weights['defog_model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    if torch.cuda.is_available():
        defog_model.to(device)
        if torch.cuda.device_count() > 1:
            gpu_ids=range(torch.cuda.device_count())
            defog_model = nn.DataParallel(defog_model, device_ids=gpu_ids)
            print("===> multiple GPU")
        # criterion = criterion.to(device)
        print("===> Setting GPU")

    print("===> Training")
    print("===> Setting Optimizer")
    optimizer = optim.Adam(defog_model.parameters(), lr=opt.lr)
    defog_model.train()
    print("===> Length of training dataset: ", len(defog_training_data_loader))
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        epoch += pre_epoch
        epoch_loss = 0
        avg_psnr = 0
        lr = adjust_learning_rate(optimizer, epoch - 1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        print("===> Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
        for iteration, batch in enumerate(defog_training_data_loader, 1):
            hazy, target = batch
            if opt.cuda:
                hazy = hazy.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            output,loss_sum = defog_model(hazy, target)
            loss = loss_sum.mean()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(defog_model.parameters(), 0.1)
            optimizer.step()
            if iteration % 10 == 0:
                print("===> Epoch[{}]({}/{}): Loss: {:.8f}".format(epoch, iteration, len(defog_training_data_loader), loss.item()))
        print("===> Epoch {} Complete: Avg. Loss: {:.8f}".format(epoch, epoch_loss / len(defog_training_data_loader)))
        avg_loss = epoch_loss / len(defog_training_data_loader)
        for iteration, batch in enumerate(defog_testing_data_loader, 1):
            hazy, target = batch
            if opt.cuda:
                hazy = hazy.cuda()
                target = target.cuda()
            prediction, loss = defog_model(hazy, target)
            mse = criterion(prediction, target)
            psnr = 20 * math.log10(1 / math.sqrt(mse.item()))
            if iteration % 20 == 0:
                print("===> ({}/{}):psnr:{:.8f}".format(iteration, len(defog_testing_data_loader), psnr))
            avg_psnr += psnr
        image = prediction.cpu().clone()
        image = image.squeeze(0)
        isExists=os.path.exists("checkpoints/defog_model/")
        if not isExists:
            os.makedirs("checkpoints/defog_model/") 
        torchvision.utils.save_image(image, "checkpoints/defog_model/" + str(epoch) + '.bmp')
        print("===> Avg. PSNR: {:.8f} dB".format(avg_psnr / len(defog_testing_data_loader)))
        psnr = avg_psnr / len(defog_testing_data_loader)
        model_out_path = "checkpoints/defog_model/defog_model_{}_epoch_{}_loss{:.8f}psnr{}.pth".format(opt.lr, epoch, avg_loss, psnr)
        state = {"epoch": epoch, "defog_model": defog_model}
        torch.save(state, model_out_path)

        print("Checkpoint saved to {}".format(model_out_path))

def test(opt, device):
    cuda = opt.cuda
    opt.seed = random.randint(1, 10000)
    print("===> Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    criterion = nn.MSELoss(reduction = 'mean').cuda()
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    else:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        cudnn.benchmark = True
        torch.cuda.manual_seed(opt.seed)

    print("===> Loading datasets")
    defog_test_dataset = Unaligned_Defog_Dataset(hazy=opt.real_hazy_test, gt=opt.real_hazy_test, train=False, real=True, transform=transforms.Compose([transforms.ToTensor()]))
    defog_testing_data_loader = DataLoader(defog_test_dataset, batch_size=1, num_workers=0, shuffle=False)

    print("===> Building model")
    defog_model = Defogging_Net(64)
    # optionally copy weights from a checkpoint
    pre_epoch = 0
    if opt.pretrained:
        if os.path.isfile('checkpoints/defog_model/'+opt.pretrained):
            print("=> loading model '{}'".format('checkpoints/defog_model/'+opt.pretrained))
            weights = torch.load('checkpoints/defog_model/'+opt.pretrained)
            pre_epoch = weights["epoch"]
            if isinstance(weights['defog_model'], torch.nn.DataParallel):
                defog_model.load_state_dict(weights['defog_model'].module.state_dict())
            else:
                defog_model.load_state_dict(weights['defog_model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))
    else:
        print("=> no model found at '{}'".format(opt.pretrained))
        sys.exit(0)
    if torch.cuda.is_available():
        defog_model.to(device)

    print("===> Testing")
    defog_model.eval()
    print(len(defog_testing_data_loader))
    with torch.no_grad():
        seq_time = time.time()
        for iteration, batch in enumerate(defog_testing_data_loader, 1):
            hazy, target, hazy_size, name = batch
            if opt.cuda:
                hazy = hazy.cuda()
                target = target.cuda()
            prediction, loss = defog_model(hazy, target, False)
            prediction.clamp_(min=0,max=1)
            #prediction = F.interpolate(prediction, size=[hazy_size[1],hazy_size[0]]) % not use it
            image = prediction.cpu().clone().squeeze(0)
            pre = transforms.ToPILImage()(image)
            pre = pre.resize(hazy_size, Image.BICUBIC)
            image = transforms.ToTensor()(pre)
            isExists=os.path.exists("results/defog_model/")
            if not isExists:
                os.makedirs("results/defog_model/") 
            torchvision.utils.save_image(image, "results/defog_model/" + name[0])
            print("===> Results save to: ", "results/defog_model/" + name[0])
    stop_time = time.time()
    print('===> Spend time: ',stop_time-seq_time)
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Badweather_Enhance")
    parser.add_argument("--hazy_dir", type=str, default='datasets/defog/hazy_train', help="Train dataset")
    parser.add_argument("--gt_dir", type=str, default='datasets/defog/gt_train', help="Train dataset")
    parser.add_argument("--symthesized_hazy_test", type=str, default='datasets/defog/hazy_test', help="Synthesize test dataset")
    parser.add_argument("--symthesized_gt_test", type=str, default='datasets/defog/gt_test', help="Synthesize test dataset")
    parser.add_argument("--real_hazy_test", type=str, default='datasets/defog/hazy_test', help="Real test dataset")
    parser.add_argument("--batchSize", type=int, default=15, help="Training batch size")
    parser.add_argument("--nEpochs", type=int, default=200, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=0.0006")
    parser.add_argument("--step", type=int, default=10,
                        help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    parser.add_argument("--cuda", action="store_true", help="Use cuda?")
    parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument("--threads", type=int, default=6, help="Number of threads for data loader to use, Default: 4")
    parser.add_argument('--pretrained', default='defog_model_0.0001_epoch_101_loss0.08797820psnr15.909477740494202.pth', type=str, help='path to pretrained model (default: none)')
    parser.add_argument('--test', action="store_true", help='state of the module (default: train)')
    opt, unknown = parser.parse_known_args()
    opt.cuda = True
    print(opt)
    if opt.test:
        test(opt, device)
    else:
        train(opt, device)


