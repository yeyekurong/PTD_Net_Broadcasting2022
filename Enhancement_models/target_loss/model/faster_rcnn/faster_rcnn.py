import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import sys

from Enhancement_models.target_loss.model.utils.config import cfg
from Enhancement_models.target_loss.model.rpn.rpn import _RPN
from Enhancement_models.target_loss.model.roi_layers import ROIAlign, ROIPool

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg
from Enhancement_models.target_loss.model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from Enhancement_models.target_loss.model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classi?ed examples (p > .5), 
                                   putting more focus on hard, misclassi?ed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=1, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
            #self.alpha[7] = 0.75
            #self.alpha[15] = 0.75
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, module_type, class_agnostic):
        super(_fasterRCNN, self).__init__()
        
        self.module_type = module_type
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)
        #if self.module_type == 'vgg16':
        #    self.feature = torch.zeros(4096).cuda()
        #else:
        #    self.feature = torch.zeros(2048).cuda()
        self.focalloss = FocalLoss(21)
        
    def forward(self, im_data, im_info, gt_boxes, num_boxes, person_feat, car_feat, is_detect = True, detection_loss=True, culsum_loss=False):
        batch_size = im_data.size(0)

        # feed image data to base model to obtain base feature map
        if self.module_type == 'vgg16':
            base_feat_layer1 = self.RCNN_base[:7](im_data)
            #base_feat_layer2 = self.RCNN_base[7:19](base_feat_layer1)
            base_feat = self.RCNN_base[7:](base_feat_layer1)  
        else:
            #print(self.RCNN_base[:5])
            base_feat_layer1 = self.RCNN_base[:5](im_data)
            base_feat = self.RCNN_base[5:](base_feat_layer1)
        if is_detect is False:
            return base_feat_layer1, None, None, None 
        else:
            self.training = self.training and num_boxes
            # feed base feature map tp RPN to obtain rois
            #print('im_info',im_info)
            #print(self.training)
            #print('boxes',gt_boxes)
            rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

            # if it is training phrase, then use ground trubut bboxes for refining
            if self.training:
                roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
                rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

                rois_label = Variable(rois_label.view(-1).long())
                rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
                rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
                rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
            else:
                rois_label = None
                rois_target = None
                rois_inside_ws = None
                rois_outside_ws = None
                rpn_loss_cls = 0
                rpn_loss_bbox = 0

            rois = Variable(rois)
            # do roi pooling based on predicted rois
            if cfg.POOLING_MODE == 'align':
                pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
            elif cfg.POOLING_MODE == 'pool':
                pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

            # feed pooled features to top model
            pooled_feat = self._head_to_tail(pooled_feat)

            # compute bbox offset
            bbox_pred = self.RCNN_bbox_pred(pooled_feat)
            if self.training and not self.class_agnostic:
                # select the corresponding columns according to roi labels
                bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
                bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
                bbox_pred = bbox_pred_select.squeeze(1)

            # compute object classification probability
            cls_score = self.RCNN_cls_score(pooled_feat)
            cls_prob = F.softmax(cls_score, 1)
        
            car_feature = 0
            person_feature = 0
            if culsum_loss:
              if self.training is False:
                  pooled_feat = pooled_feat[:128,:]
                  cls_prob = cls_prob[:128,:]
          
              if self.module_type == 'vgg16':
                  car_feature = torch.zeros(4096).cuda()  #
                  person_feature = torch.zeros(4096).cuda()
              else:
                  car_feature = torch.zeros(2048).cuda()  #
                  person_feature = torch.zeros(2048).cuda()
              if isinstance(car_feat,int):
                  car_feature = 0
              else:
                  car_feature.copy_(car_feat.detach_())
              if isinstance(person_feat,int):
                  person_feature = 0
              else:
                  person_feature.copy_(person_feat.detach_())
              delta = 0
              if self.training:
                  max_index = rois_label
              else:
                  max_index = cls_prob.argmax(dim = 1)
              #print(max_index)
              select_feat_car = pooled_feat[max_index == 7]
              select_feat_person = pooled_feat[max_index == 15]
              if(len(select_feat_car)):
                  select_feat_car=select_feat_car.mean(dim=0)
                  #print(select_feat_car)
                  if(isinstance(car_feat,int)):
                      car_feature = select_feat_car
                      print('===> Car_feature inital')
                  else:
                      delta = torch.cosine_similarity(select_feat_car, car_feat, dim = 0).item()
                      car_feature = delta * select_feat_car + (1 - delta) * car_feat
              if(len(select_feat_person)):
                  select_feat_person=select_feat_person.mean(dim=0)
                  if(isinstance(person_feat,int)):
                      person_feature = select_feat_person
                      print('===> Person_feature inital')
                  else:
                      delta = torch.cosine_similarity(select_feat_person, person_feat, dim = 0).item()
                      person_feature = delta * select_feat_person + (1 - delta) * person_feat

            RCNN_loss_cls = 0
            RCNN_loss_bbox = 0
            target_loss = 0
        
            if self.training and detection_loss:
                # classification loss
                RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
                #print(cls_score)
                #print(rois_label)
                #RCNN_loss_cls = self.focalloss(cls_score, rois_label)
                
                # bounding box regression L1 loss
                RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
            
                # the detection loss
                target_loss = rpn_loss_cls + rpn_loss_bbox + RCNN_loss_cls + RCNN_loss_bbox
        
            return base_feat_layer1, target_loss, person_feature, car_feature 
        

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
        
'''        
class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, module_type, class_agnostic):
        super(_fasterRCNN, self).__init__()
        
        self.module_type = module_type
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)
        #if self.module_type == 'vgg16':
        #    self.feature = torch.zeros(4096).cuda()
        #else:
        #    self.feature = torch.zeros(2048).cuda()
        self.focalloss = FocalLoss(21)
        
    def forward(self, im_data, im_info, gt_boxes, num_boxes, person_feat, car_feat, detection_loss=True, culsum_loss=False):
        batch_size = im_data.size(0)

        # feed image data to base model to obtain base feature map
        base_feat_layer1 = self.RCNN_base[:6](im_data)
        base_feat = self.RCNN_base[6:](base_feat_layer1)  
        self.training = self.training and num_boxes
        # feed base feature map tp RPN to obtain rois
        #print('im_info',im_info)
        #print(self.training)
        #print('boxes',gt_boxes)
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)
        
        car_feature = 0
        person_feature = 0
        if culsum_loss:
          if self.training is False:
              pooled_feat = pooled_feat[:128,:]
              cls_prob = cls_prob[:128,:]
          
          if self.module_type == 'vgg16':
              car_feature = torch.zeros(4096).cuda()  #
              person_feature = torch.zeros(4096).cuda()
          else:
              car_feature = torch.zeros(2048).cuda()  #
              person_feature = torch.zeros(2048).cuda()
          if isinstance(car_feat,int):
              car_feature = 0
          else:
              car_feature.copy_(car_feat.detach_())
          if isinstance(person_feat,int):
              person_feature = 0
          else:
              person_feature.copy_(person_feat.detach_())
          delta = 0
          if self.training:
              max_index = rois_label
          else:
              max_index = cls_prob.argmax(dim = 1)
          #print(max_index)
          select_feat_car = pooled_feat[max_index == 7]
          select_feat_person = pooled_feat[max_index == 15]
          if(len(select_feat_car)):
              select_feat_car=select_feat_car.mean(dim=0)
              #print(select_feat_car)
              if(isinstance(car_feat,int)):
                  car_feature = select_feat_car
                  print('===> Car_feature inital')
              else:
                  delta = torch.cosine_similarity(select_feat_car, car_feat, dim = 0).item()
                  car_feature = delta * select_feat_car + (1 - delta) * car_feat
          if(len(select_feat_person)):
              select_feat_person=select_feat_person.mean(dim=0)
              if(isinstance(person_feat,int)):
                  person_feature = select_feat_person
                  print('===> Person_feature inital')
              else:
                  delta = torch.cosine_similarity(select_feat_person, person_feat, dim = 0).item()
                  person_feature = delta * select_feat_person + (1 - delta) * person_feat

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        target_loss = 0
        
        if self.training and detection_loss:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
            
            # the detection loss
            target_loss = rpn_loss_cls + rpn_loss_bbox + RCNN_loss_cls + RCNN_loss_bbox
        
        return target_loss, person_feature, car_feature 

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
'''