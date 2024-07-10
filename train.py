import time
import os
import datetime

import torch
import numpy as np
from src.deeplabv3 import DeepLabV3
from Contrastive_Learning.ConLoss import MPCL
from Contrastive_Learning.Queue_class import Queue
from train_utils import train_one_epoch, evaluate, create_lr_scheduler, init_distributed_mode, save_on_master, mkdir
from my_dataset import DriveDataset, DriveDataset1, DriveDataset_mutil_class
import transforms as T
from train_utils.dice_coefficient_loss import dice_loss, build_target
from tensorboardX import SummaryWriter


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        # if hflip_prob > 0:
        #     trans.append(T.RandomHorizontalFlip(hflip_prob))
        # if vflip_prob > 0:
        #     trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            # T.ToTensor(),
            # T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            # T.ToTensor(),
            # T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # base_size = 565
    # crop_size = 480
    base_size = 1024
    crop_size = 720  # 720 640  672

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


def create_model(num_classes):

    model = DeepLabV3(num_classes=num_classes)

    return model


def main(args):
    init_distributed_mode(args)
    print(args)

    # device = torch.device(args.device)
    # segmentation nun_classes + ba ckground
    strGPUs = [str(x) for x in [1]]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(strGPUs)
    device = torch.device("cuda")
    num_classes = args.num_classes + 1

    max_dice = 0
    max_k = 0
    sum_dice = []

    for k in range(200):
        # 载入类心
        if False:
            class_center_feas = np.load(
                '/home/xmj/4T/work/unet-vessel/unet_CL-2024.1.26/output/2024-1-26/noallpic-dice2-lr0.01-epoch1000-imbalance-2/0/best_center.npy').squeeze()  # 使用预训练权重初始化类心
            class_center_feas = torch.from_numpy(class_center_feas).float().cuda()
        else:
            class_center_feas = np.random.random((3, 128))  # 随机初始化类心(classes, dim)
            class_center_feas = torch.from_numpy(class_center_feas).float().cuda()
        # 初始化对比损失类
        mpcl_loss = MPCL(device, num_class=num_classes, temperature=1.0,
                         base_temperature=1.0, m=args.margin)
        # 初始化 MemoryBank
        bank = Queue(device=device, dim=128, K=48)  # 24 128

        mean = (0.709, 0.381, 0.224)
        std = (0.127, 0.079, 0.043)

        # 用来保存coco_info的文件
        results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        results_file = os.path.join(args.output_dir, str(k), results_file)
        writer_path = os.path.join(args.output_dir, str(k), 'writer')
        writer = SummaryWriter(writer_path)

        data_root = args.data_path
        # check data root
        # if os.path.exists(os.path.join(data_root, "DRIVE")) is False:
        #     raise FileNotFoundError("DRIVE dose not in path:'{}'.".format(data_root))

        train_dataset = DriveDataset_mutil_class(args.data_path,
                                                 train=True,
                                                 transforms=get_transform(train=True, mean=mean, std=std))

        val_dataset = DriveDataset_mutil_class(args.data_path,
                                               train=False,
                                               transforms=get_transform(train=False, mean=mean, std=std))

        print("Creating data loaders")
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        else:
            train_sampler = torch.utils.data.RandomSampler(train_dataset)
            test_sampler = torch.utils.data.SequentialSampler(val_dataset)

        train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=train_sampler, num_workers=args.workers,
            collate_fn=train_dataset.collate_fn, drop_last=True)

        val_data_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1,
            sampler=test_sampler, num_workers=args.workers,
            collate_fn=train_dataset.collate_fn)

        print("Creating model")
        # create model num_classes equal background + foreground classes
        model = create_model(num_classes=num_classes)
        model = torch.nn.DataParallel(model).to(device)

        # model.to(device)
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        params_to_optimize = [p for p in model_without_ddp.parameters() if p.requires_grad]

        optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        scaler = torch.cuda.amp.GradScaler() if args.amp else None

        # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
        lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs, warmup=True)

        # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
        if args.resume:
            # If map_location is missing, torch.load will first load the module to CPU
            # and then copy each parameter to where it was saved,
            # which would result in all processes on the same machine using the same set of devices.
            checkpoint = torch.load(args.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
            model_without_ddp.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.amp:
                scaler.load_state_dict(checkpoint["scaler"])

        if args.test_only:
            confmat = evaluate(model, val_data_loader, device=device, num_classes=num_classes)
            val_info = str(confmat)
            print(val_info)
            return

        best_dice = 0.
        best_epoch = 0.
        print("Start training")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            mean_loss, contrast_loss, lr, new_class_center_feats = train_one_epoch(model, optimizer, train_data_loader,
                                                                                   device, epoch, num_classes,
                                                                                   lr_scheduler=lr_scheduler,
                                                                                   class_center_feats=class_center_feas,
                                                                                   m=args.margin, mpcl_loss=mpcl_loss,
                                                                                   Bank=bank,
                                                                                   print_freq=args.print_freq,
                                                                                   scaler=scaler)
            # confmat, dice = evaluate(model, val_data_loader, device=device, num_classes=num_classes)
            
            confmat, metric1 = evaluate(model, val_data_loader, device=device, num_classes=num_classes)
            val_info = str(confmat)


            if args.save_best is True:
                if best_dice < sum(metric1[1:, 0]) / len(metric1[1:, 0]):
                    best_dice = sum(metric1[1:, 0]) / len(metric1[1:, 0])
                    best_epoch = epoch

                    save_file = {'model': model_without_ddp.state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 'lr_scheduler': lr_scheduler.state_dict(),
                                 'args': args,
                                 'epoch': epoch}
                    save_on_master(save_file, os.path.join(args.output_dir, str(k), 'best_model.pth'))

                    np.save(os.path.join(args.output_dir, str(k), 'best_center.npy'),
                            new_class_center_feats.cpu().detach().numpy())

                else:
                    continue

            # print(f"best dice: {best_dice:.3f}")
            # print('best epoch: ', best_epoch)
            # print('dice', sum(metric1[1:, 0]) / len(metric1[1:, 0]))

            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                train_info = f"[epoch: {epoch}]\n" \
                             f"train_loss: {mean_loss:.4f}\n" \
                             f"lr: {lr:.6f}\n" \
                             f"dice: {metric1[0, 0]:.3f}  {metric1[1, 0]:.3f} {metric1[2, 0]:.3f}\n" \
                             f"mean_dice coefficient: {sum(metric1[1:, 0]) / len(metric1[1:, 0]):.3f}\n" \
                             f"hd95: {metric1[0, 1]:.3f}  {metric1[1, 1]:.3f} {metric1[2, 1]:.3f}\n" \
                             f"mean_hd95: {sum(metric1[1:, 1]) / len(metric1[1:, 1]):.3f}\n" \
                             f"jc: {metric1[0, 2]:.3f}  {metric1[1, 2]:.3f} {metric1[2, 2]:.3f}\n" \
                             f"mean_jc: {sum(metric1[1:, 2]) / len(metric1[1:, 2]):.3f}\n" \
                             f"asd: {metric1[0, 3]:.3f}  {metric1[1, 3]:.3f} {metric1[2, 3]:.3f}\n" \
                             f"mean_asd:    {sum(metric1[1:, 3]) / len(metric1[1:, 3]):.3f}\n" \
                             f"best dice: {best_dice:.3f}\n" \
                             f"best epoch: {best_epoch:.1f}\n"
                f.write(train_info + val_info + "\n\n")

            if args.output_dir:
                # 只在主节点上执行保存权重操作
                save_file = {'model': model_without_ddp.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'lr_scheduler': lr_scheduler.state_dict(),
                             'args': args,
                             'epoch': epoch}
                if args.amp:
                    save_file["scaler"] = scaler.state_dict()

                # if args.save_best is True:
                #     save_on_master(save_file,
                #                    os.path.join(args.output_dir,str(k), 'latest.pth'))
                #     np.save(os.path.join(args.output_dir,str(k), 'latest_center.npy'),new_class_center_feats)
                # else:
                #     save_on_master(save_file,
                #                    os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        writer.close()

        max_dice_file = "max_dice.txt"
        max_dice_file = os.path.join(args.output_dir, max_dice_file)

        if max_dice < best_dice:
            max_dice = best_dice
            max_k = k

        sum_dice.append(best_dice)
        sum_dice1 = sum(sum_dice) / len(sum_dice)
        with open(max_dice_file, "a") as f:
            f.write(str(max_dice) + "\n" + str(max_k) + "\n" + str(sum_dice1) + "\n\n\n\n")

        print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练文件的根目录(DRIVE)
    parser.add_argument('--data-path', default='/home/xl/work/unet-vessel-2024/unet-2024.1.29-mutilmodel/',
                        help='dataset')
    # 训练设备类型
    parser.add_argument('--device', default='cuda', help='device')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=2, type=int, help='num_classes')
    # 每块GPU上的batch_size
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    # 是否使用同步BN(在多个GPU之间同步)，默认不开启，开启后训练速度会变慢
    parser.add_argument('--sync_bn', type=bool, default=False, help='whether using SyncBatchNorm')
    # 数据加载以及预处理的线程数
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # 训练学习率，这里默认设置成0.01(使用n块GPU建议乘以n)，如果效果不好可以尝试修改学习率
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 只保存dice coefficient值最高的权重
    parser.add_argument('--save-best', default=True, type=bool, help='only save best weights')
    # 训练过程打印信息的频率
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    # 文件保存地址
    parser.add_argument('--output-dir',default='./output/',help='path where to save')
    # 基于上次的训练结果接着训练
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # 不训练，仅测试
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # 分布式进程数
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    parser.add_argument("--CLASS_CENTER_FEA_INIT", default=None, type=str,
                        help="是否使用预训练类心文件")
    parser.add_argument("--margin", default=0.4, type=float,
                        help="初始化margin参数")

    args = parser.parse_args()

    # 如果指定了保存文件地址，检查文件夹是否存在，若不存在，则创建
    if args.output_dir:
        mkdir(args.output_dir)

    main(args)
