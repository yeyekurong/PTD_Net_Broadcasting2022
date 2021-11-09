import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch

def default_loader(path, real):
    img = Image.open(path).convert('RGB')
    size = img.size
    if real:
        width = 8*int(img.size[0]/8)
        height = 8*int(img.size[1]/8)
        if(max(width,height)>1500):
            width = int(width / 2 /8) * 8
            height = int(height / 2 /8) *8
        img = img.resize([width,height])
    return img, size
    
class Unaligned_Defog_Dataset(BaseDataset):
    def __init__(self, hazy, gt, train=True, real=False, transform=None):
        self.hazy = hazy
        self.gt = gt
        self.real = real
        self.transform = transform
        self.train = train  # training set or test set
        self.img_name_list_hazy = os.listdir(hazy)
        self.img_name_list_gt = os.listdir(gt)
        self.size = [512,512]
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        hazy = []
        gt = []
        if self.train:
            hazy_i = os.path.join(self.hazy, self.img_name_list_hazy[index])
            img_name = self.img_name_list_hazy[index][:4]+ ".jpg.png"
            gt_i = os.path.join(self.gt, img_name)
            #print(hazy_i)
            #print(gt_i)
        else:
            hazy_i = os.path.join(self.hazy, self.img_name_list_hazy[index])
            img_name = self.img_name_list_hazy[index]
            gt_i = os.path.join(self.gt, self.img_name_list_hazy[index])
            
        hazy_i, self.size = default_loader(hazy_i, self.real) 
        gt_i, self.size = default_loader(gt_i, self.real)
        if self.transform is not None and self.real:
            img_hazy = self.transform(hazy_i)
            img_gt = self.transform(gt_i)
            return img_hazy, img_gt, self.size, img_name
        else:  # self.usage == 'test'
            img_hazy = self.transform(hazy_i)
            img_gt = self.transform(gt_i)
            return img_hazy, img_gt
        
    def __len__(self):
        return len(self.img_name_list_hazy)

class Unaligned_Enhance_Dataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.isTrain = opt.isTrain
        if self.isTrain:
            self.dir_A = os.path.join(opt.dataroot, opt.train_cleardir)  # create a path '/path/to/data/trainA'  clear images
            self.dir_B = os.path.join(opt.dataroot, opt.train_fogdir)  # create a path '/path/to/data/trainB'  fog images
            self.list_A = os.path.join(opt.dataroot, 'enhance/train_A.txt')
            self.list_B = os.path.join(opt.dataroot, 'enhance/train_B.txt')
        else:
            self.dir_A = os.path.join(opt.dataroot, opt.test_dir)
            self.dir_B = self.dir_A
            self.list_A = os.path.join(opt.dataroot, 'enhance/test.txt')
            self.list_B = os.path.join(opt.dataroot, 'enhance/test.txt')
        self.A_paths = sorted(make_dataset(self.dir_A, self.list_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, self.list_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        size = B_img.size
        #print('input image: ',B_path)
        #if(max(size[1],size[0])>1800):
        #    B_img = B_img.resize((960, 540), Image.BICUBIC)
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'Orisize':A_img.size, 'A_paths': A_path, 'A_boxes':torch.Tensor([0]), 'A_num_box':0, 'B_boxes':torch.Tensor([0]), 'B_num_box':0, 'im_info':[0, 0]}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
        #return 100
