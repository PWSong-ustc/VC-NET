import torch
import torch.nn.functional as F


def label_downsample(labels,fea_h,fea_w):

    '''
    labels: N*H*W
    '''
    labels = labels.float().cuda()
    labels = F.interpolate(labels, size=fea_w, mode='nearest')
    labels = labels.permute(0, 2, 1).contiguous()
    labels = F.interpolate(labels, size=fea_h, mode='nearest')
    labels = labels.permute(0, 2, 1).contiguous()  # n*fea_h*fea_w
    labels = labels.int()
    return labels

def update_class_center_iter(cla_src_feas,batch_src_labels,class_center_feas,m):

    '''
    batch_src_feas  : n*c*h*w
    barch_src_labels: n*h*w
    '''
    batch_src_feas     = cla_src_feas.detach()
    batch_src_labels   = batch_src_labels.cuda()
    n,c,fea_h,fea_w    = batch_src_feas.size()
    batch_y_downsample = label_downsample(batch_src_labels, fea_h, fea_w)  # n*fea_h*fea_w
    batch_y_downsample = batch_y_downsample.unsqueeze(1)  # n*1*fea_h*fea_w
    batch_class_center_fea_list = []
    for i in range(3):
        fea_mask        = torch.eq(batch_y_downsample,i).float().cuda()  #n*1*fea_h*fea_w
        class_feas      = batch_src_feas * fea_mask  # n*c*fea_h*fea_w
        class_fea_sum   = torch.sum(class_feas, [0, 2, 3])  # c
        class_num       = torch.sum(fea_mask, [0, 1, 2, 3])
        if class_num == 0:
            batch_class_center_fea = class_center_feas[i,:].detach()
        else:
            batch_class_center_fea = class_fea_sum/class_num
        batch_class_center_fea = batch_class_center_fea.unsqueeze(0) # 1 * c
        batch_class_center_fea_list.append(batch_class_center_fea)
    batch_class_center_feas = torch.cat(batch_class_center_fea_list,dim=0) # n_class * c
    class_center_feas = m * class_center_feas + (1-m) * batch_class_center_feas

    return class_center_feas


def mpcl_loss_calc(feas,labels, class_center_feas, loss_func):

    '''
    feas:  batch*c*h*w
    label: batch*img_h*img_w
    class_center_feas: n_class*n_feas
    '''

    n,c,fea_h,fea_w = feas.size()
    labels      = labels.float().detach()
    labels      = F.interpolate(labels, size=fea_w, mode='nearest')
    labels      = labels.permute(0,2,1).contiguous()
    labels      = F.interpolate(labels, size=fea_h, mode='nearest')
    labels      = labels.permute(0, 2, 1).contiguous()         # batch*fea_h*fea_w

    labels  = labels.cuda()
    labels  = labels.view(-1).long()

    feas = torch.nn.functional.normalize(feas.detach(),p=2,dim=1)
    feas = feas.transpose(1,2).transpose(2,3).contiguous() #batch*c*h*w->batch*h*c*w->batch*h*w*c
    feas = torch.reshape(feas,[n*fea_h*fea_w,c]) # [batch*h*w] * c
    feas = feas.unsqueeze(1) # [batch*h*w] 1 * c

    class_center_feas = torch.nn.functional.normalize(class_center_feas,p=2,dim=1)
    class_center_feas = torch.transpose(class_center_feas, 0, 1)  # n_fea*n_class

    loss =  loss_func(feas,labels,class_center_feas)
    return loss
