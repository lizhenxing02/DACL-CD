import os
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Sampler
import itertools
import pickle
import random
from utils.mask_gen import BoxMaskGenerator
import torch.nn.functional as F
from torchvision.transforms import Resize, GaussianBlur


class Logger:
    def __init__(self, log_path):
        self.log_path = log_path

    def write(self, txt):
        with open(self.log_path, 'a') as f:
            f.write(txt)
            f.write("\r\n")


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']

def get_mem():
    return '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)

def get_next(dataloader, i):
    try:
        batch = next(i)
    except:
        trainloader_iter = iter(dataloader)
        batch = next(trainloader_iter)
    return batch

def list2device(x, device):
    """
    将一个列表中的所有元素移动到指定的设备上，如GPU。
    如果输入不是一个列表，则直接将其移动到指定设备上。
    此函数旨在支持深度学习模型中，批量处理数据从CPU内存移动到GPU（如果使用的话）。
    """
    if isinstance(x, list):
        y = []
        for i in x:
            y.append(i.to(device))
        return y
    else:
        return x.to(device)


class RepeatSampler(Sampler):
    r"""Repeated sampler

    Arguments:
        data_source (Dataset): dataset to sample from
        sampler (Sampler): sampler to draw from repeatedly
        repeats (int): number of repetitions or -1 for infinite
    """

    def __init__(self, sampler, repeats=-1):
        if repeats < 1 and repeats != -1:
            raise ValueError('repeats should be positive or -1')
        self.sampler = sampler
        self.repeats = repeats

    def __iter__(self):
        if self.repeats == -1:
            reps = itertools.repeat(self.sampler)
            return itertools.chain.from_iterable(reps)
        else:
            reps = itertools.repeat(self.sampler, self.repeats)
            return itertools.chain.from_iterable(reps)

    def __len__(self):
        if self.repeats == -1:
            return 2 ** 62
        else:
            return len(self.sampler) * self.repeats

def get_train_val_loader(train_dataset, val_dataset, train_bs, val_bs, labeled_ratio, train_split_path, work_dir):
    """
    Training data is divided into two parts：train_loader and train_loader_remain
    train_loader: with change labels
    train_loader_remain: without change labels
    """
    num_workers = 2
    train_dataset_size = len(train_dataset)
    print('dataset size: ', train_dataset_size)
    partial_size = int(labeled_ratio * train_dataset_size)
    print('partial size: ', partial_size)

    if train_split_path:
        train_ids = pickle.load(open(train_split_path, 'rb'))

    else:
        train_ids = np.arange(train_dataset_size)
        np.random.shuffle(train_ids)
        pickle.dump(train_ids, open(os.path.join(work_dir, 'train_split.pkl'), 'wb'))

    train_sampler = RepeatSampler(SubsetRandomSampler(train_ids[:partial_size]))
    train_remain_sampler = RepeatSampler(SubsetRandomSampler(train_ids[partial_size:]))


    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_bs,
                            num_workers=num_workers,
                            shuffle=False,
                            pin_memory=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=train_bs,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True)

    train_loader_remain = DataLoader(train_dataset,
                                     batch_size=train_bs,
                                     sampler=train_remain_sampler,
                                     num_workers=num_workers,
                                     pin_memory=False)


    return train_loader, train_loader_remain, val_loader


def get_train_loader(train_dataset, train_unsup_dataset, mask_collate_fn, train_bs, labeled_ratio, train_split_path, work_dir):
    """
    Outputs:
    train_sup_loader: with change labels有变化标签
    train_unsup_loader_0:  without change labels, with random mask parameter.)无变化标签，有随机掩码参数
    train_unsup_loader_1:  without change labels无变化标签
    """
    num_workers = 2
    train_dataset_size = len(train_dataset)
    print('dataset size: ', train_dataset_size)
    partial_size = int(labeled_ratio * train_dataset_size)
    print('partial size: ', partial_size)

    if train_split_path:
        train_ids = pickle.load(open(train_split_path, 'rb'))

    else:
        train_ids = np.arange(train_dataset_size)
        np.random.shuffle(train_ids)
        pickle.dump(train_ids, open(os.path.join(work_dir, 'train_split.pkl'), 'wb'))

    # train_sup_sampler = SubsetRandomSampler(train_ids[:partial_size])
    train_sup_sampler = RepeatSampler(SubsetRandomSampler(train_ids[:partial_size]))

    train_sup_loader = DataLoader(train_dataset,
                              batch_size=train_bs,
                              sampler=train_sup_sampler,
                              collate_fn=mask_collate_fn,  # 添加 mask参数
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True)
    
    # train_remain_sampler = SubsetRandomSampler(train_ids[partial_size:])
    train_remain_sampler = RepeatSampler(SubsetRandomSampler(train_ids[partial_size:]))

    # 修改--如果labeled_ratio为1.0，表示没有无标签数据，则只返回有标签的加载器
    if labeled_ratio == 1.0:
        return train_sup_loader, None, None  # 禁用无标签加载器
    else:
        train_unsup_loader_0 = DataLoader(train_unsup_dataset,
                                    batch_size=train_bs,
                                    sampler=train_remain_sampler,
                                    collate_fn=mask_collate_fn,  # 添加 mask参数
                                    num_workers=num_workers,
                                    pin_memory=False)

        train_unsup_loader_1 = DataLoader(train_unsup_dataset,
                                        batch_size=train_bs,
                                        sampler=train_remain_sampler,   # sampler传递的是函数，此步骤确保了unsup_loader_0和unsup_loader_1的输出顺序不同
                                        num_workers=num_workers,
                                        pin_memory=False)

        return train_sup_loader, train_unsup_loader_0, train_unsup_loader_1



def save_model(model, save_path, iteration, loss, metric):
    torch.save({
        'net': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'epoch': iteration,
        'loss': loss,
        'metric': metric
    }, save_path)

def format_logs(logs):
    str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
    s = ', '.join(str_logs)
    return s



# def generate_mixed_images(sample1, sample2, mask):
#     """
#     根据两个样本和掩码生成混合图像。

#     该函数接收两个样本图像（每个样本包含两个时间点的图像）和一个掩码，
#     并生成一个混合图像，其中每个时间点的图像都是根据掩码对两个样本对应时间点图像的加权平均。

#     参数:
#     sample1: 第一个样本，包含两个时间点的图像数据。
#     sample2: 第二个样本，包含两个时间点的图像数据。
#     mask: 掩码，用于决定两个样本在生成混合图像时的贡献比例。

#     返回:
#     一个列表，包含两个混合后的图像，分别对应两个时间点。
#     """
#     # [x1 y1], [x2, y2] -> [mix_x, mix_y]    x, y represent bi-temporal images
#     # 根据掩码混合两个样本的第一个时间点图像
#     mix_x = sample1[0] * mask + sample2[0] * (1-mask)
#     # 根据掩码混合两个样本的第二个时间点图像
#     mix_y = sample1[1] * mask + sample2[1] * (1-mask)
#     # 返回混合后的两个时间点图像
#     return [mix_x, mix_y]



from torchvision import transforms
import torchvision

def generate_mixed_images(sample1, sample2, mask, 
                          apply_radio_A=True, apply_geo_B=True,):
    """
    对样本对1中的时相A进行模拟辐射差异，对时相B进行模拟几何偏移，
    然后利用mask与样本对2中的时相A、B进行混合得到混合样本对。

    参数:
    sample1: 样本对1（A1, B1），类型为列表[torch.Tensor, torch.Tensor]
    sample2: 样本对2（A2, B2），类型为列表[torch.Tensor, torch.Tensor]
    mask: CutMix掩码张量（形状为[B, 1, H, W]）
    apply_radio_A: 是否对sample1的A1应用辐射差异
    apply_geo_B: 是否对sample1的B1应用几何偏移
    """

    A1, B1 = sample1
    A2, B2 = sample2

    # 获取可用的GPU设备，如果没有则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将所有输入张量移至指定设备
    A1 = A1.to(device)
    B1 = B1.to(device)
    A2 = A2.to(device)
    B2 = B2.to(device)
    mask = mask.to(device)

    # 定义颜色抖动变换，用于改变图像的颜色属性（亮度、对比度、饱和度、色调）
    colorjit = torchvision.transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.2)
     # 定义仿射变换，用于对图像进行小范围的旋转、缩放、平移和剪切
    affine = torchvision.transforms.RandomAffine(degrees=(-5, 5), scale=(1, 1.02),translate=(0.02, 0.02), shear=(-5, 5))

    # 对样本对1的时相A应用辐射差异（颜色抖动）
    if apply_radio_A:
        # 检查是否为批量输入
        is_batch = A1.dim() == 4
        if is_batch:
            # 批量处理模式
            A1_transformed = []
            for i in range(A1.shape[0]):
                img = A1[i]  # 获取单张图像 [C, H, W]
                img_pil = transforms.ToPILImage()(img)  # 转为PIL
                img_transformed = colorjit(img_pil)  # 应用颜色抖动
                img_tensor = transforms.ToTensor()(img_transformed).to(device)  # 转回张量
                A1_transformed.append(img_tensor)
            A1_transformed = torch.stack(A1_transformed)  # 重新组合成批量
        else:
            # 单张图像模式
            A1_pil = transforms.ToPILImage()(A1)
            A1_transformed = colorjit(A1_pil)
            A1_transformed = transforms.ToTensor()(A1_transformed).to(device)
    else:
        A1_transformed = A1

    # 对样本对1的时相B应用几何偏移（仿射变换）
    if apply_geo_B:
        # 确保B1是4D张量（批量模式）
        if B1.dim() == 3:
            B1 = B1.unsqueeze(0)  # 添加批量维度
        # 应用仿射变换
        B1_transformed = []
        for i in range(B1.shape[0]):
            img_pil = transforms.ToPILImage()(B1[i])
            img_transformed = affine(img_pil)
            img_tensor = transforms.ToTensor()(img_transformed).to(device)
            B1_transformed.append(img_tensor)
        B1_transformed = torch.stack(B1_transformed)
        # 如果输入是单张图像，移除批量维度
        if not is_batch:
            B1_transformed = B1_transformed.squeeze(0)
    else:
        B1_transformed = B1

    # 根据mask进行混合
    mix_x = A1_transformed * mask + A2 * (1 - mask)  # 混合时相A
    mix_y = B1_transformed * mask + B2 * (1 - mask)  # 混合时相B
    return [mix_x, mix_y]




def generate_mixed_images_intra(sample, mask):
    # [x1 y1], [y1, x1] -> [mix_x, mix_y]    x, y represent bi-temporal images
    mix_x = sample[0] * mask + sample[1] * (1-mask)
    mix_y = sample[1] * mask + sample[0] * (1-mask)
    return [mix_x, mix_y]

def generate_mixed_images_withlabels(sample1, sample2, label, mask):
    # [x1 y1], [x2, y2] -> [mix_x, mix_y]    x, y represent bi-temporal images
    mix_x = sample1[0] * mask + sample2[0] * (1-mask)
    mix_y = sample1[1] * mask + sample2[1] * (1-mask)
    mix_label = label[0] * mask + label[1] * (1-mask)
    return mix_x, mix_y, mix_label


# masks = generate_salience_mask(logits_u1_tea)    # [B,1,H,W]
def generate_salience_mask(logits, mask_size):
    """
    生成显著性掩膜。

    参数:
    logits: 张量，形状为[B, 2, H, W]，表示模型的输出 logits。
    mask_size: 一个包含两个整数的列表或元组，表示掩膜的大小，例如[64, 64]。

    返回:
    一个形状为[B, 1, H, W]的张量，表示生成的显著性掩膜。
    """
    # mask size [64, 64]
    # logits [B, 2, H, W]  -> mask [B, 1, H, W]
    # 将 logits 转换为概率
    prob = F.softmax(logits, dim=1)
    # 仅保留类别为1的概率
    prob_change = prob[:,1,:,:]
    # 创建一个调整大小的变换，将概率图调整到指定的掩膜大小
    torch_resize = Resize(mask_size)
    small_prob_change = torch_resize(prob_change)
    # 创建一个高斯模糊的变换，用于模糊调整大小后的概率图
    AddGaussianBlur = GaussianBlur((5,5),(0.1, 2.0))
    small_prob_change_blurred = AddGaussianBlur(small_prob_change) # [B, mask_size, mask_size]

    # 获取输入 logits 的形状
    B, _, H, W = logits.shape

    # 初始化掩膜为全1张量
    mask = torch.ones((B, H, W))

    # 对于每一批次的数据，找到概率图中最高的点，并在其周围绘制掩膜
    for batch in range(B):
        heatmap = small_prob_change_blurred[batch]
        # 找到概率图中概率最高的点的索引
        max_index = torch.argmax((heatmap))
        # 根据索引计算点坐标
        row = torch.floor(max_index/mask_size[1])
        col = max_index % mask_size[1]
        # 放大后的行列号
        row = row / mask_size[0] * H
        col = col / mask_size[1] * W
        # 计算掩膜的左上角和右下角坐标
        top_left = [i.to(torch.int) for i in [row - mask_size[0] / 2, col - mask_size[1] / 2]]
        bottom_right = [i.to(torch.int) for i in [row + mask_size[0] / 2, col + mask_size[1] / 2]]
        # 确保掩膜的坐标在图像范围内
        top_left[0], top_left[1] = max(top_left[0], 0), max(top_left[1], 0)
        bottom_right[0], bottom_right[1] = min(bottom_right[0], 256), min(bottom_right[1], 256)
        # 在掩膜中将计算出的区域设置为0
        mask[batch, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = 0

    # 在通道维度上增加一个维度，以匹配输入 logits 的形状
    mask = mask.unsqueeze(1)
    # 最终输出一个形状为 [B, 1, H, W] 的二值掩膜张量
    return mask

def generate_object_mask(logits, threshold=0.5):
    # mask size [64, 64]
    # logits [B, 2, H, W]  -> mask [B, 1, H, W]
    prob = F.softmax(logits, dim=1)
    prob_change = prob[:, 1, :, :]
    mask = (prob_change < threshold).float() # change的region为0

    mask = mask.unsqueeze(1)

    return mask


