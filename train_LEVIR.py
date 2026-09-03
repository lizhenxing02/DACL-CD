import argparse
from utils.func import Logger, save_model, format_logs, get_train_loader, list2device, generate_mixed_images, generate_salience_mask
from utils.metrics import *
import torch
import torch.nn as nn
import os
import time
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import models
import itertools
import utils.optim_weight_ema as optim_weight_ema
from loaders.datasets import LEVIRDataset
from utils.mask_gen import BoxMaskGenerator, AddMaskParamsToBatch
from utils.lr_schedules import make_lr_schedulers
import torchvision.transforms as tvt
from utils.ValEpoch import ValEpoch
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.optim.lr_scheduler import CosineAnnealingLR


def get_arguments():
    """Parse all the arguments provided from the CLI.解析命令行参数

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="train process")
    # ########## 新增：阶段学习率参数
    # parser.add_argument("--lr_freeze", type=float, default=1e-4,
    #                     help="Learning rate for frozen backbone stage (first 3 epochs)")
    # parser.add_argument("--lr_unfreeze", type=float, default=1e-5,
    #                     help="Learning rate for unfrozen stage (after 3 epochs)")
    # ############
    parser.add_argument("--work_dirs", type=str, default='./semi_checkpoints/LEVIR99/ConvNeXt-DTC-CD') # 训练模型保存路径
    parser.add_argument('--ratio', '--labeled_ratio', type=float, default=0.5) # 有标签数据的比例（如0.1表示使用10%的数据作为有标签样本）
    parser.add_argument('--mask_size', type=list, default=[64, 64]) # 掩码的大小（像素）
    parser.add_argument("--log", type=str, default='cutmix_0.5') #日志文件名前缀，用于区分不同实验配置（cutmix_0.1表示10%标记数据）
    parser.add_argument("--data_root", type=str, default='./LEVIR-CD256-mini') # 数据集路径

    parser.add_argument('--train_split_path', type=str, default=None) # 指定已生成的数据划分文件（.pkl）路径
    # "train_split_path" is the location of a pkl file which records the split of labeled and unlabeled data
    # For the first run, you can set it to None, and the train_split.pkl will be generated automatically in work_dirs
    # After that, you can set the path of generated train_split.pkl to ensure each experiment is conducted with the same split for a dataset
    parser.add_argument('--num_epochs', '--num_epochs', type=int, default=8) # 无监督训练的总轮数（整个训练周期的长度）
    parser.add_argument('--epoch_start_unsup', type=int, default=4) #从第几轮开始引入无监督训练
                                                                       # flexible for different datasets
    # parser.add_argument('--cons_weight', type=float, default=1.0) # 一致性损失（无监督损失）的权重系数，平衡有监督和无监督损失
    parser.add_argument('--cons_weight', type=float, default=0.5) # ConvNeXt调整为0.5
    # parser.add_argument('--conf_thresh', type=float, default=0.97) # 0.97 置信度阈值
    parser.add_argument('--conf_thresh', type=float, default=0.7) # ConvNeXt调整为0.7
    parser.add_argument('--conf_per_pixel', type=bool, default=False) # 是否为每个像素计算单独的置信度掩码（默认 False，使用全局阈值）
    parser.add_argument('--cons_loss_fn', type=str, default='var') # 一致性损失函数类型（如 'var' 可能表示方差损失，衡量学生与教师预测的差异）

    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="Base learning rate for training with polynomial decay.") #基础学习率，控制参数更新步长
    parser.add_argument('--lr_sched', type=str, default='cosine') # 学习率调度策略，可选 ['none', 'stepped', 'cosine', 'poly']
    parser.add_argument('--lr_step_epochs', type=str, default='')
    parser.add_argument('--lr_step_gamma', type=float, default=0.1)
    parser.add_argument('--lr_poly_power', type=float, default=0.9)
    parser.add_argument('--aug_strong_colour', default=True) #是否启用强颜色增强（如亮度、对比度、饱和度变化）
    parser.add_argument('--freeze_bn', default=False) #是否冻结批归一化（BatchNorm）层的参数（默认 False，即训练时更新）

    parser.add_argument("--batch-size", type=int, default=2,
                        help="train dataset batch size.") #训练批次大小
    parser.add_argument("--val-batch-size", type=int, default=2,
                        help="val dataset batch size.") #验证批次大小
    parser.add_argument("--opt_type", type=str, default='adam',
                        help="val dataset batch size.") #优化器类型（如 'adam' 或 'sgd'）

    return parser.parse_args()


def train():
    num_classes = 2
    torch_device = torch.device('cuda')
    args = get_arguments()

    # logger 创建保存日志的目录，并初始化日志记录器
    save_dir = os.path.join(args.work_dirs, args.log)
    os.makedirs(save_dir, exist_ok=True)
    logger = Logger(os.path.join(save_dir, "train.log"))
    logger.write(str(args))

    # build network 初始化学生网络和教师网络，使用预训练的 ConvNeXt_CD 模型
    student_net = models.ConvNeXt_CD(num_classes, pretrained=True).to(torch_device)
    teacher_net = models.ConvNeXt_CD(num_classes, pretrained=True).to(torch_device)

    #定义学生网络的优化器为 Adam，学习率为 args.learning_rate
    # student_optim = torch.optim.Adam(student_net.parameters(), lr=args.learning_rate)

    ################### 针对ConvNeXt-tiny修改
    student_optim = torch.optim.AdamW(student_net.parameters(),lr=5e-4, 
    weight_decay=1e-5,  # 权重衰减抑制深度卷积的过拟合
    betas=(0.9, 0.999)  # 保持默认betas即可
)

    # teacher_optim 冻结教师网络的参数，并定义教师网络的权重更新优化器
    for p in teacher_net.parameters():
        p.requires_grad = False
    # 创建一个EMAWeightOptimizer指数移动平均（EMA）优化器，用于根据学生网络的参数逐步更新教师网络的权重，平滑参数变化
    teacher_optim = optim_weight_ema.EMAWeightOptimizer(teacher_net, student_net, ema_alpha=0.99)
    eval_net = teacher_net #将教师网络指定为评估网络

    # # CELoss 定义监督损失函数为交叉熵损失
    # supervised_loss = nn.CrossEntropyLoss()
############ 替换原supervised_loss
    supervised_loss = FocalLoss(alpha=0.7, gamma=2)  # alpha>0.5增强正样本权重

    # val 初始化评估指标
    metric = ChangeMetrics(False)

    print("Build network")

    # 根据是否启用强颜色增强，定义无监督数据的变换
    if args.aug_strong_colour:
        train_unsup_transforms = tvt.Compose([
            tvt.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            tvt.ToTensor(),
            tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_unsup_transforms = tvt.Compose([
            tvt.ToTensor(),
            tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # dataset 加载训练集、无监督训练集和验证集
    train_dataset = LEVIRDataset(args.data_root,"train", supervised_train=True, transforms_unsup=None)  # label [B, H, W]
    train_unsup_dataset = LEVIRDataset(args.data_root,"train", supervised_train=False, transforms_unsup=train_unsup_transforms)
    val_dataset = LEVIRDataset(args.data_root,"val")

    # random mask (not used) 掩码生成器（未使用，代码中可能预留）
    mask_generator = BoxMaskGenerator((0.25, 0.25), random_aspect_ratio=False) # 创建一个生成矩形掩码的工具
    add_mask_params_to_batch = AddMaskParamsToBatch(mask_generator) # 将生成的掩码参数添加到批次数据中，用于后续数据增强或混合操作

    # loader 获取数据加载器
    loaders = get_train_loader(train_dataset, train_unsup_dataset, add_mask_params_to_batch,
                        args.batch_size, args.ratio, train_split_path=args.train_split_path, work_dir=args.work_dirs)

    # train_sup_loader：有标签数据（带真实标签），train_unsup_loader_0和train_unsup_loader_1：未标记数据的两个分支（用于生成混合图像）
    # train_unsup_loader_0: 无变化标签，有随机掩码参数；train_unsup_loader_1: 无变化标签
    train_sup_loader, train_unsup_loader_0, train_unsup_loader_1 = loaders
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.val_batch_size,
                            num_workers=1,
                            shuffle=False,
                            pin_memory=True)

    # Create iterators 创建迭代器（用于按批次读取数据）
    train_sup_iter = iter(train_sup_loader)
    train_unsup_iter_0 = iter(train_unsup_loader_0) if train_unsup_loader_0 is not None else None
    train_unsup_iter_1 = iter(train_unsup_loader_1) if train_unsup_loader_1 is not None else None

    # scheduler 计算每轮迭代次数（无标签数据量 / 批量大小）
    unlabel_size = len(train_dataset) * (1 - args.ratio) # 无标签数据的样本数量 
    iters_per_epoch = int(unlabel_size // args.batch_size)  # 每个epoch中无标签数据的迭代次数
    sup_iters_per_epoch = int(len(train_dataset) * args.ratio // args.batch_size) # 每个epoch中有标签数据的迭代次数

    # # 总迭代次数
    # total_iters = iters_per_epoch * args.num_epochs

    ####### 修改后代码：根据ratio选择迭代次数基准
    if args.ratio == 1.0:
        # 全量有标签数据时，总迭代次数基于有标签数据的迭代次数
        total_iters = sup_iters_per_epoch * args.num_epochs
    else:
        # 半监督时使用原逻辑
        total_iters = iters_per_epoch * args.num_epochs
    #######

    # 定义学习率调度器（返回两个调度器，基于epoch的学习率调度器和基于迭代次数的学习率调度器）
    lr_epoch_scheduler, lr_iter_scheduler = make_lr_schedulers(
        optimizer=student_optim, total_iters=total_iters, schedule_type=args.lr_sched,
        step_epochs=args.lr_step_epochs, step_gamma=args.lr_step_gamma, poly_power=args.lr_poly_power
    )

    iter_i = 0 # 初始化迭代计数器
    print('Training...')
    best_val_metric = 0 # 初始化最佳验证指标 
    # bms = 0
    bms = {}
    for epoch_i in range(args.num_epochs):
        # 每个epoch开始时检查是否存在基于epoch的学习率调度器，存在则调用step方法更新学习率
        if lr_epoch_scheduler is not None:
            # lr_epoch_scheduler.step(epoch_i)
            lr_epoch_scheduler.step()

        t1 = time.time()
        ramp_val = 1.0

        student_net.train() # 设置学生网络为训练模式
        if teacher_net is not student_net:
            teacher_net.train() # 设置教师网络为训练模式
    
        # 如果参数中设置了冻结批归一化层
        if args.freeze_bn:
            student_net.freeze_batchnorm() # 冻结学生网络的BatchNorm层
            if teacher_net is not student_net:
                teacher_net.freeze_batchnorm() # 冻结教师网络的BatchNorm层

        # 初始化损失累加器
        sup_loss_acc = 0.0 #监督损失
        consistency_loss_acc = 0.0 #一致性损失
        feat_contras_loss_acc = 0.0 #特征对比损失
        conf_rate_acc = 0.0 # 置信度
        # 初始化有监督和无监督批次计数器
        n_sup_batches = 0
        n_unsup_batches = 0

        # 如果当前epoch小于无监督学习开始的epoch
        if args.epoch_start_unsup > 0 and epoch_i < args.epoch_start_unsup:
            iters = sup_iters_per_epoch # 使用有监督学习的迭代次数
            eval_net = student_net # 使用学生网络进行验证
        else:
            # 否则，使用总的迭代次数和教师网络进行验证
            iters = iters_per_epoch
            eval_net = teacher_net
        
        # 初始化验证器，用于在验证集上评估网络性能
        val_runner = ValEpoch(num_classes, eval_net, supervised_loss, metric)

        # load best model of the supervised epoches.加载监督训练阶段的最佳模型
        # 如果当前epoch等于无监督训练的起始epoch
        if epoch_i == args.epoch_start_unsup:
            print("load best model of the supervised epoches.")
            # 加载保存在指定路径下的最佳模型检查点
            checkpoint = torch.load(os.path.join(save_dir, 'best.pth'))
            # 将检查点中的网络参数加载到学生网络中
            student_net.load_state_dict(checkpoint['net'])
            # 将检查点中的网络参数加载到教师网络中
            teacher_net.load_state_dict(checkpoint['net'])

        # 使用tqdm包装迭代器，以可视化训练进度
        for sup_batch in tqdm(itertools.islice(train_sup_iter, iters)):  # 一个epoch包含的iteration数
            # 如果学习率迭代调度器存在，则在每个迭代中更新学习率
            # if lr_iter_scheduler is not None:
            #     # lr_iter_scheduler.step(iter_i)
            #     lr_iter_scheduler.step()
            # 清零学生网络优化器的梯度，避免梯度累积
            student_optim.zero_grad()

            #
            # Supervised branch 有监督分支
            #
            # 将批次数据中的图像列表移动到当前设备torch_device上
            batch_x = list2device(sup_batch['image'], torch_device)
            # 将批次数据中的标签移动到当前设备torch_device上
            batch_y = sup_batch['labels'].to(torch_device)

            # 使用学生网络对图像进行前向传播，获取预测结果
            logits_sup = student_net(batch_x)
            # 计算学生网络预测结果的监督损失
            sup_loss = supervised_loss(logits_sup, batch_y)
            # 对监督损失进行反向传播，计算梯度
            sup_loss.backward()

            # 如果一致性权重大于0，且当前epoch数大于或等于无监督训练开始的epoch数，则执行无监督分支逻辑
            if args.cons_weight > 0.0 and epoch_i >= args.epoch_start_unsup:

                #
                #  Unsupervised branch CutmixCD 无监督分支
                #
                # 从无监督数据迭代器中获取两组无监督批次数据
                unsup_batch0 = next(train_unsup_iter_0)
                unsup_batch1 = next(train_unsup_iter_1)


                # 教师网络使用样本0（弱增强），学生网络使用样本1（强增强）

                # 将样本0的图像数据（弱增强）加载到设备中，用于教师网络
                batch_ux0_tea = list2device(unsup_batch0['sample0']['image'], torch_device)
                # 将样本1的图像数据（强增强）加载到设备中，用于学生网络
                batch_ux0_stu = list2device(unsup_batch0['sample1']['image'], torch_device)
                # batch_um0 = unsup_batch0['sample0']['mask'].to(torch_device)

                # 将样本0的图像数据（弱增强）加载到设备中，用于教师网络
                batch_ux1_tea = list2device(unsup_batch1['sample0']['image'], torch_device)
                # 将样本1的图像数据（强增强）加载到设备中，用于学生网络
                batch_ux1_stu = list2device(unsup_batch1['sample1']['image'], torch_device)

                # batch_mix_masks = unsup_batch0['mask'].to(torch_device)   # (N,1,H,W)    中间的mask为0

                # Get teacher predictions for original images获取教师网络对原始图像的预测
                with torch.no_grad(): # 禁用梯度计算
                    logits_u0_tea = teacher_net(batch_ux0_tea).detach()
                    logits_u1_tea = teacher_net(batch_ux1_tea).detach()

                # generate salience mask from logits_u1_tea  [B, 2, H, W]
                # 生成显著性掩码，指定的掩码大小args.mask_size
                batch_mix_masks = generate_salience_mask(logits_u1_tea, args.mask_size).to(torch_device)    # [B,1,H,W] 中间的mask为0
                # 混合图像：CutMixCD核心操作，用掩码混合两个未标记样本
                batch_ux_stu_mixed = generate_mixed_images(batch_ux0_stu, batch_ux1_stu, batch_mix_masks)

                # 学生网络获取预测结果logits_cons_stu和特征feat_mixed，用于计算特征对比损失
                logits_cons_stu, feat_mixed = student_net(batch_ux_stu_mixed, return_features=True)   # output the Feature contrastive loss

                # 教师网络对两个未标记样本的预测logits_u0_tea和logits_u1_tea按权重混合，生成混合预测logits_cons_tea
                logits_cons_tea = logits_u0_tea * batch_mix_masks + logits_u1_tea * (1 - batch_mix_masks)

                # Logits -> probs 将教师网络和学生网络的预测结果（logits）转换为概率分布
                prob_cons_tea = F.softmax(logits_cons_tea.detach(), dim=1)
                prob_cons_stu = F.softmax(logits_cons_stu, dim=1)


                # Confidence thresholding
                if args.conf_thresh > 0.0:
                    # 教师网络预测的置信度计算
                    conf_tea = prob_cons_tea.max(dim=1)[0]
                    # 置信度掩码生成
                    conf_mask = (conf_tea >= args.conf_thresh).float()[:, None, :, :]
                    # 记录置信率
                    conf_rate_acc += float(conf_mask.mean())
                    # 平均置信度掩码
                    if not args.conf_per_pixel:
                        conf_mask = conf_mask.mean()
                    # 损失掩码赋值
                    loss_mask = conf_mask

                # Compute per-pixel consistency loss
                # Note that the way we aggregate the loss across the class/channel dimension (1)
                # depends on the loss function used. Generally, summing over the class dimension
                # keeps the magnitude of the gradient of the loss w.r.t. the logits
                # nearly constant w.r.t. the number of classes. When using logit-variance,
                # dividing by `sqrt(num_classes)` helps.

                # 计算一致性损失（学生预测与混合教师预测的差异）
                if args.cons_loss_fn == 'var': # 学生网络和教师网络概率分布的差值平方和
                    delta_prob = prob_cons_stu - prob_cons_tea
                    consistency_loss = delta_prob * delta_prob
                    consistency_loss = consistency_loss.sum(dim=1, keepdim=True)
                elif args.cons_loss_fn == 'kld': # Kullback-Leibler 散度（KL 散度）
                    consistency_loss = F.kl_div(F.log_softmax(logits_cons_stu, dim=1), prob_cons_tea, reduce=False)
                    consistency_loss = consistency_loss.sum(dim=1, keepdim=True)
                else:
                    raise ValueError('Unknown consistency loss function {}'.format(args.cons_loss_fn))

                # 如损失掩码存在，则将其应用于一致性损失，然后对所有像素和图像取平均值
                if loss_mask:
                    consistency_loss = (consistency_loss * loss_mask).mean()
                else:
                    consistency_loss = consistency_loss.mean() # 添加else，无论是否有掩码，都对所有像素和图像取平均值，确保为标量
                

                # 特征对比损失计算  教师网络的预测概率和学生网络的特征图之间的对比损失
                feat_contras_loss = Alg_loss(prob_cons_tea, feat_mixed, args.conf_thresh)

                # # 无监督总损失：一致性损失 + 特征对比损失
                # unsup_loss = consistency_loss * args.cons_weight + feat_contras_loss.mean()

                ####### 原代码中feat_contras_loss直接与total_loss相加，改为带动态权重
                contras_weight = min(epoch_i / 50, 0.5)  # 前50个epoch权重从0线性增至0.5
                unsup_loss = consistency_loss * args.cons_weight + feat_contras_loss.mean() * contras_weight
                #######

                unsup_loss.backward() # 反向传播无监督损失
                
                # 累加一致性损失和特征对比损失的值
                consistency_loss_acc += float(consistency_loss.detach())
                feat_contras_loss_acc += float(feat_contras_loss.detach())

                n_unsup_batches += 1 # 增加无监督批次计数器

            # 学生模型优化器进行参数更新
            student_optim.step()
            # 如果教师模型优化器存在，则进行参数更新
            if teacher_optim is not None:
                teacher_optim.step()
            
            # 移动212行的 lr_iter_scheduler.step() 到 optimizer.step() 之后
            # 如果学习率迭代调度器存在，则在每个迭代中更新学习率
            if lr_iter_scheduler is not None:
                # lr_iter_scheduler.step(iter_i)
                lr_iter_scheduler.step()

            # 将监督学习损失从Tensor转换为浮点数
            sup_loss_val = float(sup_loss.detach())
            # 检查损失值是否为NaN，以判断模型是否失效
            if np.isnan(sup_loss_val):
                print('NaN detected; network dead, bailing.')
                return

            # 累加有监督损失值
            sup_loss_acc += sup_loss_val
            # 更新有监督批次计数器 n_sup_batches 和迭代计数器 iter_i
            n_sup_batches += 1
            iter_i += 1

        sup_loss_acc /= n_sup_batches # 计算平均监督损失
        if n_unsup_batches > 0: #如果存在无监督批次，计算平均一致性损失、特征对比损失和置信率
            consistency_loss_acc /= n_unsup_batches
            feat_contras_loss_acc /= n_unsup_batches
            conf_rate_acc /= n_unsup_batches

        t2 = time.time()

        # train results打印当前训练轮次的损失信息，并将这些信息存储到字典中
        print('Epoch {}: took {:.3f}s, TRAIN clf loss={:.6f}, consistency loss={:.6f}, contras loss={:.6f}, conf rate={:.3%}'.format(
              epoch_i + 1, t2 - t1, sup_loss_acc, consistency_loss_acc, feat_contras_loss_acc, conf_rate_acc))
        train_log = {'sup_loss': sup_loss_acc, 'consistency_loss':consistency_loss_acc, 'feat_contras_loss':feat_contras_loss_acc,'conf_rate':conf_rate_acc}

        # Eval this epoch 对当前模型进行验证
        val_log = val_runner.run(val_loader)
        val_metric = val_log['f1'] # 获取F1分数作为验证指标


        # 保存当前训练轮次的模型参数及相关信息（最新）
        save_model(eval_net, os.path.join(save_dir, 'latest.pth'), epoch_i, val_log['loss'], val_metric)

        # 保存最佳metric模型
        if val_log['f1'] > best_val_metric: # 检查当前验证日志中的F1分数是否大于历史最佳F1分数
            best_val_metric = val_log['f1'] # 更新最佳F1分数
            bms = val_log
            # 保存当前评估网络的最佳模型参数
            save_model(eval_net, os.path.join(save_dir, 'best.pth'), epoch_i, val_log['loss'], val_metric)
            # 保存相应的student模型
            torch.save({
                'net': student_net.module.state_dict() if hasattr(student_net, 'module') else student_net.state_dict(),
            }, os.path.join(save_dir, 'student.pth'))

        # 日志记录
        logger.write('Epoch:\t' + str(epoch_i))
        logger.write('Train:\t' + format_logs(train_log))
        logger.write('Val:\t' + format_logs(val_log))
        logger.write("Best:\t" + format_logs(bms))
        logger.write("\n")

        print("train:", train_log)
        print("val:", val_log)
        print("best_metric:\t" + format_logs(bms))



# def Alg_loss(weak_prob_ul, strong_feat_ul, threshold):
#     """
#     特征对比损失函数
#     该函数通过对比弱标签概率图和强特征图来计算损失，旨在提高模型的泛化能力。
#     参数:
#     weak_prob_ul : 未标记数据的弱标签概率图。
#     strong_feat_ul: 未标记数据的强特征图。
#     threshold : 用于生成二值掩码的阈值。
#     返回:
#     loss_pos: 对比损失和正样本距离损失之和。
#     """
#     feat_size = strong_feat_ul.size()[-2:] # 获取特征图尺寸（H, W）
#     # weak_prob_ul = F.interpolate(weak_prob_ul, size=feat_size, mode='nearest',) # 上采样概率图到特征图尺寸
#     weak_prob_ul = F.interpolate(weak_prob_ul, size=feat_size, mode='nearest', align_corners=None)  # 上采样概率图到特征图尺寸

#     mask_unit = weak_prob_ul.ge(threshold).float() # 生成掩码，突出显示超过阈值的区域
#     weight = (mask_unit.sum(dim=-1).sum(dim=-1) + 1e-5).unsqueeze(dim=-1)  # 计算每个样本的掩码总和，避免除以零
#     # 调整维度：特征图（B, C, H, W）→（B, 1, C, H, W），掩码（B, 2, 1, H, W）
#     mask_unit = mask_unit.unsqueeze(dim=2)       # [B, 2, 1, H=32, W=32]
#     feat_ul = strong_feat_ul.unsqueeze(dim=1)    # [B, 1, 512, H=32, W=32]

#     # 计算每个类别的特征中心：对每个类的特征加权平均
#     class_centers = (mask_unit * feat_ul).sum(-1).sum(-1) / weight
#     # 对特征中心进行归一化
#     class_centers = F.normalize(class_centers, dim=-1)  # [B, 2, 512]
#     # 计算对比损失：强制两类中心差异最大化（负平方距离）

#     # loss_contras = - (class_centers[:, 0, :] - class_centers[:, 1, :]) ** 2
#     loss_contras = torch.clamp( (class_centers[:, 0, :] - class_centers[:, 1, :])** 2, min=0.1 ).mean() # 修改
#     loss_contras = loss_contras.mean()

#     # 计算正样本对之间的距离
#     dist_pos = torch.bmm(class_centers.permute(1, 0, 2), class_centers.permute(1, 0, 2).permute(0, 2, 1))
#     mask_pos = 1 - (dist_pos == 0).float() # 生成掩码以排除自身点积
#     loss_pos = ((0.5 - dist_pos / 2) * mask_pos).mean() # 计算正样本对损失
#     # 结合对比损失和正样本对损失
#     loss_pos += loss_contras

#     return loss_pos

# 优化后的Alg_loss函数，适配ConvNeXt特征
def Alg_loss(weak_prob_ul, strong_feat_ul, threshold):
    if isinstance(strong_feat_ul, tuple):
        # 从元组中提取特征张量，第一个元素为768通道特征输出
        strong_feat_ul = strong_feat_ul[0]
    # 1. 适配ConvNeXt特征通道（统一到512维）
    if strong_feat_ul.size(1) != 512:
        # 动态创建1x1卷积适配器（仅在需要时创建，避免额外参数量）
        feat_adapter = nn.Conv2d(strong_feat_ul.size(1), 512, kernel_size=1, bias=False).to(strong_feat_ul.device)
        nn.init.kaiming_normal_(feat_adapter.weight, mode='fan_out', nonlinearity='relu')
        strong_feat_ul = feat_adapter(strong_feat_ul)
    
    # 2. 上采样概率图到特征图尺寸
    feat_size = strong_feat_ul.size()[-2:]
    weak_prob_ul = F.interpolate(weak_prob_ul, size=feat_size, mode='nearest', align_corners=None)
    
    # 3. 动态阈值生成（适配ConvNeXt的概率分布）
    batch_max_prob = weak_prob_ul.max(dim=1, keepdim=True)[0].mean()
    batch_threshold = threshold * batch_max_prob  # 自适应调整阈值
    mask_unit = weak_prob_ul.ge(batch_threshold).float()  # 生成有效区域掩码
    
    # 4. 优化权重计算（避免除以零或数值溢出）
    weight = mask_unit.sum(dim=-1).sum(dim=-1)  # [B, 2]
    weight = torch.where(weight < 1e-5, torch.tensor(1e-5).to(weight.device), weight)
    weight = weight.unsqueeze(dim=-1)  # [B, 2, 1]
    
    # 5. 计算类别中心（特征+掩码加权平均）
    mask_unit = mask_unit.unsqueeze(dim=2)  # [B, 2, 1, H, W]
    feat_ul = strong_feat_ul.unsqueeze(dim=1)  # [B, 1, 512, H, W]
    class_centers = (mask_unit * feat_ul).sum(-1).sum(-1) / weight  # [B, 2, 512]
    class_centers = F.normalize(class_centers, dim=-1)  # 归一化类中心
    
    # 6. 类间距离损失（动态钳位）
    class_dist_sq = (class_centers[:, 0, :] - class_centers[:, 1, :]) ** 2  # [B, 512]
    min_clamp = torch.quantile(class_dist_sq, 0.25)  # 动态钳位最小值
    loss_contras = torch.clamp(class_dist_sq, min=min_clamp).mean()
    
    # 7. 正样本对损失（适配ConvNeXt特征分布）
    dist_pos = torch.bmm(
        class_centers.permute(1, 0, 2),
        class_centers.permute(1, 2, 0)
    )  # [2, B, B]
    mask_pos = 1 - torch.eye(dist_pos.size(1)).to(dist_pos.device).unsqueeze(0)  # [1, B, B]
    loss_pos = ((0.6 - dist_pos / 2) * mask_pos).mean()
    
    # 8. 平衡损失权重
    return 0.7 * loss_contras + 0.3 * loss_pos

# 在定义损失函数处替换
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2):
        super().__init__()
        self.alpha = alpha  # 正样本权重
        self.gamma = gamma

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


if __name__ == "__main__":
    train()