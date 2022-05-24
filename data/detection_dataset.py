from data.image_folder import make_dataset
from data.base_dataset import BaseDataset, get_transform
import torch
from torch.utils import data
from PIL import Image
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
import os
import torch
import random
import numpy as np
from math import floor
import random

class Detection_Enhance_Dataset(BaseDataset):
    """
    This dataset class can load images with annotations.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        rescale_length = opt.load_size
        step = 32
        self.batch_size = opt.batch_size
        self.isTrain = opt.isTrain
        if self.isTrain:
            self.dir_A = os.path.join(opt.dataroot, opt.train_vocdir)  # create a path '/path/to/data/trainA'  gt_train2 4322 clear images, 
            self.dir_B = os.path.join(opt.dataroot, opt.train_fogdir_l)  # create a path '/path/to/data/trainB'  fog images
            self.list_A = os.path.join(opt.dataroot, 'enhance/train_A.txt')
            self.list_B = os.path.join(opt.dataroot, 'enhance/train_B.txt')
        else:
            self.dir_A = os.path.join(opt.dataroot, opt.test_dir)
            self.dir_B = self.dir_A
            self.list_A = os.path.join(opt.dataroot, 'enhance/test.txt')
            self.list_B = os.path.join(opt.dataroot, 'enhance/test.txt')
        self.A_paths, self.A_scales = self.ratio_sorted(make_dataset(self.dir_A, self.list_A, opt.max_dataset_size),rescale_length, step)
        self.B_paths, self.B_scales = self.ratio_sorted(make_dataset(self.dir_B, self.list_B, opt.max_dataset_size),rescale_length, step)
        self.size = len(self.B_paths)
        self.transform_A = transforms.Compose([transforms.ToTensor()])
        self.transform_C = transforms.Compose([transforms.Lambda(lambda img: self.scale_height(img, rescale_length, step, Image.BICUBIC)),transforms.ToTensor()]) #标准600运行不起来
        self.transform_D = transforms.Compose([transforms.Lambda(lambda img: self.scale_height(img, rescale_length, step, Image.BICUBIC)),transforms.Lambda(lambda img: self.flip(img)),transforms.ToTensor()]) #标准600运行不起来
        self._classes = ('__background__',  # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self.A_index_pre = -2
        self.A_index_begin = -2
        self.A_imgs = []
        self.B_index_pre = -2
        self.B_index_begin = -2
        self.B_imgs = []

    def __getitem__(self, index, Astyle=True, Bstyle=True, Flip=False):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, boxes
            A (tensor)       -- an image in the input domain
            boxes (tensor)   -- an annotations data of the image A
        """
        # randomize the index for domain B to avoid fixed pairs.
        while True: 
            index_B = (index % (len(self.B_paths))).item()  
            B_path = self.B_paths[index_B]
            
            if index_B - self.B_index_pre==1:
                index_A = self.A_index_pre + 1
            else:
                index_A = random.randint(0,len(self.A_paths)-self.batch_size-1)
            if not self.isTrain:
                index_A = index
            #index_A = (index % (len(self.A_paths))).item() 
            A_path = self.A_paths[index_A]  # make sure index is within then range
            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')
            A_name = os.path.split(A_path)[1]
            B_name = os.path.split(B_path)[1]
            #classes = ['person','car','bicycle','bus','motorbike']
            classes = ['person','car']
            #print(B_path)
            #print(index_A)
            #[w, h] = B_img.size
            A_tensor = self.transform_C(A_img)
            #no_flip = random.random() > 0.5
            no_flip = not Flip
            if no_flip:
                B_tensor = self.transform_C(B_img)
            else:
                B_tensor = self.transform_D(B_img)
            
            if index_B-self.B_index_pre==1 and index_B in self.B_imgs:
                self.A_imgs = list(range(self.A_index_begin, self.A_index_begin+self.batch_size))
                self.B_imgs = list(range(self.B_index_begin, self.B_index_begin+self.batch_size))
            else:
                self.A_index_begin = index_A
                self.A_imgs = list(range(index_A, index_A+self.batch_size))
                self.B_index_begin = index_B
                self.B_imgs = list(range(index_B, index_B+self.batch_size))
            self.B_index_pre = index_B
            self.A_index_pre = index_A
            #print('B index',self.B_scales[self.B_imgs])
            #print('B size',B_tensor.size())
            #print('A index',self.A_scales[self.A_imgs])
            #print('A size',A_tensor.size())
            
            A_size = self.A_scales[self.A_imgs].max(0)
            B_size = self.B_scales[self.B_imgs].max(0)
            #print('B index',A_size)
            #print('A index',B_size)
            
            padding_A_img = torch.FloatTensor(3, A_size[1], A_size[0]).zero_() 
            padding_B_img = torch.FloatTensor(3, B_size[1], B_size[0]).zero_() 
            padding_A_img[:, :A_tensor.size()[1], :A_tensor.size()[2]] = A_tensor
            padding_B_img[:, :B_tensor.size()[1], :B_tensor.size()[2]] = B_tensor
            A_scale = A_tensor.size()[1] / A_img.size[1]  
            B_scale = B_tensor.size()[1] / B_img.size[1]
            #w_scale = B_tensor.size()[2] / w  
            A_boxes = torch.Tensor([0])
            B_boxes = torch.Tensor([0])
            A_actual_obj = 0
            B_actual_obj = 0
            if(Bstyle):
                xml_file = self.opt.detection_annotation + B_name[:-3]  + 'xml'
                #print(xml_file)
                if os.path.exists(xml_file):
                    tree = ET.parse(xml_file)
                    objs = tree.findall('object')
                    sizes = tree.findall('size')
                    B_boxes = torch.zeros(40, 5)
                    B_actual_obj = 0
                    for ix, obj in enumerate(objs):
                        w_xml = float(sizes[0].find('width').text.lower().strip())
                        h_xml = float(sizes[0].find('height').text.lower().strip())
                        #print(B_img.size[0] / w_xml)
                        cls = obj.find('name').text.lower().strip()
                        dif = int(obj.find('difficult').text.lower().strip())
                        if cls in classes and dif==0 and B_actual_obj < 40:
                            bbox = obj.find('bndbox')
                            # Make pixel indexes 0-based
                            if no_flip:
                                x1 = float(bbox.find('xmin').text) * B_scale * (B_img.size[0] / w_xml)  - 1
                                y1 = float(bbox.find('ymin').text) * B_scale * (B_img.size[1] / h_xml) - 1
                                x2 = float(bbox.find('xmax').text) * B_scale * (B_img.size[0] / w_xml) - 1
                                y2 = float(bbox.find('ymax').text) * B_scale * (B_img.size[1] / h_xml) - 1
                            else:
                                x1 = (B_img.size[0]-float(bbox.find('xmax').text)) * B_scale * (B_img.size[0] / w_xml) - 1
                                y1 = float(bbox.find('ymin').text) * B_scale * (B_img.size[1] / h_xml) - 1
                                x2 = (B_img.size[0]-float(bbox.find('xmin').text)) * B_scale * (B_img.size[0] / w_xml) - 1
                                y2 = float(bbox.find('ymax').text) * B_scale * (B_img.size[1] / h_xml) - 1           
                            B_boxes[B_actual_obj, :] = torch.Tensor([x1, y1, x2, y2, self._classes.index(cls)])
                            #print(B_path, no_flip, B_img.size, float(bbox.find('xmin').text), B_scale, [x1, y1, x2, y2])               
                            B_actual_obj += 1
                    if B_actual_obj == 0:
                        self.B_paths = np.delete(self.B_paths,index_B)
                        self.B_scales = np.delete(self.B_scales,index_B,0)
                        print("No target in ", self.B_paths[index_B])
                        print(cls)
                        continue
                else:
                    print("No Bstyle image in ", xml_file)
                    
            if(Astyle):
                xml_file = self.opt.detection_annotation_source + A_name[:-3]  + 'xml'
                #print(xml_file)
                if os.path.exists(xml_file):
                    tree = ET.parse(xml_file)
                    objs = tree.findall('object')
                    A_boxes = torch.zeros(40, 5)
                    A_actual_obj = 0
                    for ix, obj in enumerate(objs):
                        cls = obj.find('name').text.lower().strip()
                        dif = int(obj.find('difficult').text.lower().strip())
                        if cls in classes and dif==0 and A_actual_obj < 40:
                            bbox = obj.find('bndbox')
                            # Make pixel indexes 0-based
                            x1 = float(bbox.find('xmin').text) * A_scale - 1
                            y1 = float(bbox.find('ymin').text) * A_scale - 1
                            x2 = float(bbox.find('xmax').text) * A_scale - 1
                            y2 = float(bbox.find('ymax').text) * A_scale - 1
                            A_boxes[A_actual_obj, :] = torch.Tensor([x1, y1, x2, y2, self._classes.index(cls)])  
                            A_actual_obj += 1
                    if A_actual_obj == 0:
                        self.A_paths = np.delete(self.A_paths,index_A)
                        self.A_scales = np.delete(self.A_scales,index_A)
                        print("No target in ", self.A_paths[index_A])
                        print(cls,dif,B_actual_obj)
                        continue
                else:
                    print("No Astyle image in ", xml_file)
                
            break            
        return {'A': padding_A_img, 'A_paths': A_path, 'B': padding_B_img, 'Orisize':A_img.size, 'A_boxes':A_boxes, 'A_num_box':A_actual_obj, 'B_boxes':B_boxes, 'B_num_box':B_actual_obj, 'im_info':[A_scale, B_scale]}

    def __len__(self):
        """Return the total number of images in the dataset.

        """
        return self.size
    
    def ratio_sorted(self, img_list, target_width, step):
        ratio_list = []
        size_list = []
        for img in img_list:
            [ow, oh] = Image.open(img).size
            if ow < oh:
                w = target_width 
                h = int(oh*target_width / ow /step)*step
                h = h if(h<1000) else 960
            else:
                h = target_width 
                w = int(ow*target_width / oh /step)*step
                w = w if(w<1000) else 960
            ratio_list.append(w/ow)
            size_list.append([w,h])
        img_list = np.array(img_list)
        ratio_list = np.array(ratio_list)
        size_list = np.array(size_list)
        ratio_index = np.argsort(ratio_list) 
        return img_list[ratio_index], size_list[ratio_index]
    
    def flip(self, img):
            return img.transpose(Image.FLIP_LEFT_RIGHT)      
    
    def scale_height(self, img, target_width, step, method=Image.BICUBIC):
        ow, oh = img.size
        if ow < oh:
            w = target_width 
            h = int(oh*target_width / ow /step)*step
            h = h if(h<1000) else 960
        else:
            h = target_width 
            w = int(ow*target_width / oh /step)*step
            w = w if(w<1000) else 960
        #print('real size',[w,h])    
        return img.resize((w, h), method)
