import sys
import torch
from torch import nn
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target
from torch.nn.modules.loss import CrossEntropyLoss
from medpy import metric
from sklearn.metrics import confusion_matrix
import numpy as np
from MPSCL.func import *
import math


def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    losses = {}
    ce_loss = CrossEntropyLoss(weight=loss_weight)
    for name, x in inputs.items():
        loss = ce_loss(x, target[:].long())
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        # loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
        if dice is True:
            dice_target = build_target(target, num_classes, ignore_index)
            loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
        losses[name] = loss

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        jc = metric.binary.jc(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, hd95, jc, asd
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0, 1, 0
    else:
        return 0, 0, 0, 0


def caculate_dice_for_classes(pred, target, num_classes):
    dice_scores = []
    for class_id in range(1, num_classes):
        binary_target = np.where(target == class_id, 1, 0)
        binary_predicted = np.where(pred == class_id, 1, 0)
        dice_score = metric.binary.dc(binary_target, binary_predicted)
        dice_scores.append(dice_score)
    return dice_scores


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    dice_1 = []
    # metric_list = []
    metric1 = 0
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            metric_list = []
            image, target = image.to(device), target.to(device)
            class_features, output = model(image)
            output = output['out']
            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

            # dice_2=metric.binary.dc(output.argmax(1).detach().cpu().numpy(),target.detach().cpu().numpy())
            # dice_1.append(dice_2)
            # dice_2 = caculate_dice_for_classes(output.argmax(1).detach().cpu().numpy(),target.detach().cpu().numpy(), num_classes)
            # dice_1.append(sum(dice_2)/len(dice_2))

            # neglect  Background
            for i in range(0, 3):
                metric_list.append(calculate_metric_percase(output.argmax(1).detach().cpu().numpy() == i,
                                                            target.detach().cpu().numpy() == i))

            metric1 += np.array(metric_list)

        # metric_list = metric_list / 2
        metric1 = metric1 / len(data_loader)
        # print('dice', np.mean(metric1, axis=0)[0])
        # for i in range(0, 3):
        #     print('class %d dice %f hd95 %f jc %f asd %f' % (i, metric1[i ][0], metric1[i ][1], metric1[i ][2], metric1[i ][3]))

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()
        # print('dice________1',sum(dice_1)/len(dice_1))

    return confmat, metric1


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, class_center_feats, m, mpcl_loss, Bank, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None

    contrast = []

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        label = target
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            class_features, output = model(image)  # class_features[8,128,45,45], output[8,3,720,720]
            # 更新类心 [3, 128]
            new_class_center_feats = update_class_center_iter(class_features, target, class_center_feats, m=m)
            Bank._dequeue_and_enqueue(class_features.detach())
            Bank._dequeue_and_enqueue_label(target.detach())
            # print(Bank.iter)
            if Bank.iter < 24 / 8:    # after 64 / 8 iter, bank will full
           #  if True:
                # 特征用的是经过proj和mlp后的特征
                CL_loss = mpcl_loss_calc(feas=class_features, labels=target,
                                         class_center_feas=new_class_center_feats, loss_func=mpcl_loss)
            else:
                # print('*'*10, 'enter memory bank', '*'*10)
                CL_loss = mpcl_loss_calc(feas=(Bank.queue).permute(3, 2, 0, 1), labels=(Bank.label).permute(2, 0, 1),
                                         class_center_feas=new_class_center_feats, loss_func=mpcl_loss)

            label = label.long().to(device)
            label = label.view(-1)

            reweight_epoch_min = 100
            reweight_epoch_max = 200
            alpha = 2

            if epoch <= reweight_epoch_min:
                # now_power = 0
                loss_weight = None

            elif epoch > reweight_epoch_max:
                # now_power = ((epoch - reweight_epoch_min) / (reweight_epoch_max - self.reweight_epoch_min)) ** self.alpha

                class_count = torch.bincount(label).float()
                if class_count.numel() == 2:
                    class_count = torch.cat((class_count, torch.tensor([1]).to(device)))
                loss_weight = (1 - class_count / label.size(0))

            else:
                now_power = ((epoch - reweight_epoch_min) / (reweight_epoch_max - reweight_epoch_min)) ** alpha
                class_count = torch.bincount(label).float()
                if class_count.numel() == 2:
                    class_count = torch.cat((class_count, torch.tensor([1]).to(device)))
                loss_weight = (1 - class_count / label.size(0))
                loss_weight = loss_weight.cpu().numpy()
                # per_cls_weights = per_cls_weights * self.class_extra_weight
                loss_weight = [math.pow(num, now_power) for num in loss_weight]
                loss_weight = torch.tensor(loss_weight).to(device)
            #print(loss_weight)

            loss = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255)

        total_loss = loss + CL_loss
        # '''for Test'''
        # print('*'*30)
        # print('CL loss: ', CL_loss.item(), 'Total loss:', total_loss.item())
        # print('*'*30)
        contrast.append(CL_loss.item())

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, sum(contrast) / len(contrast), lr, new_class_center_feats
    # return metric_logger.meters["loss"].global_avg,sum(contrast)/len(contrast),lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
