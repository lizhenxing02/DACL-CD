import os
import torch
import numpy as np
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
import glob
import cv2 as cv

# ImageNet数据集的预处理方式，用于将图像转换为Tensor并进行标准化
imagenet_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# 对标签图像进行处理，将其转换为适合模型训练的格式
def resize(label):
    label = label / 255  # 将像素值归一化到0-1之间
    label = label.reshape([1, label.shape[0], label.shape[1]])  # 调整形状为 [1, H, W]
    label = np.concatenate((1 - label, label), axis=0)  # 类别数为2 按类别数扩展维度为 [2, H, W]
    return label

# 加载数据集分割文件，返回文件路径列表
def load(split_path):
    res = []
    with open(split_path, 'r') as f:
        lines = f.readlines() # 读取文件中的每一行
        # 遍历每一行，去除换行符和反斜杠，并将结果添加到列表中
        for line in lines:
            v = line.replace("\n", "")
            v = v.replace("\\", "/")
            res.append(v)
    return res

# S2looking数据集类，用于处理S2looking数据集
class S2lookingDataset_all(data.Dataset):
    def __init__(self, root_dir, split, supervised_train=True, transforms_unsup=None):
        assert split in ["train", "test", "val"]
        self.split = split
        self.supervised_train = supervised_train
        self.transforms = transforms_unsup
        # root_dir = "/data0/qidi/S2looking"
        # 初始化数据集路径
        root_dir = os.path.join(root_dir, split)
        # 加载图像和标签路径
        T1_image_path = glob.glob(root_dir + '/A' + '/*.png')
        T2_image_path = glob.glob(root_dir + '/B' + '/*.png')
        label_path = glob.glob(root_dir + '/label' + '/*.png')
        # 对路径进行排序
        T1_image_path.sort()
        T2_image_path.sort()
        label_path.sort()
        # 存储路径
        # self.ids = load(txt_dir)
        self.T1_image_path = T1_image_path
        self.T2_image_path = T2_image_path
        self.label_path = label_path


    def __getitem__(self, idx):
        sample = {}
        # 加载图像和标签
        image1 = Image.open(self.T1_image_path[idx])
        image2 = Image.open(self.T2_image_path[idx])
        label = cv.imread(self.label_path[idx], 0)
        label = (label != 0).astype('uint8')
        label = torch.from_numpy(label).long()
        # 根据训练模式进行数据预处理
        if self.supervised_train or self.split == "test":
            image1 = imagenet_preprocess(image1)
            image2 = imagenet_preprocess(image2)
            sample['image'] = [image1, image2]
            sample['labels'] = label
        else:     # 'sample0' no augmentation  'sample1' strong augmentation--- ‘sample0’无增强  ‘sample1’强增强
            sample0 = {}
            sample1 = {}
            image11 = imagenet_preprocess(image1)
            image22 = imagenet_preprocess(image2)
            sample0['image'] = [image11, image22]
            image1 = self.transforms(image1)
            image2 = self.transforms(image2)
            sample1['image'] = [image1, image2]

            sample['sample0'] = sample0
            sample['sample1'] = sample1

        # return sample, self.ids[idx]
        return sample

    def __len__(self):
        return len(self.T1_image_path)

# LEVIR数据集类，用于处理LEVIR数据集
class LEVIRDataset(data.Dataset):
    def __init__(self, root_dir, split, supervised_train=True, transforms_unsup=None):
        """
         root_dir: 数据集根目录
         split: 数据集分割（"train", "test", "val"）
         supervised_train: 是否为监督训练
         transforms_unsup: 无监督训练的图像变换方法
        """
        assert split in ["train", "test", "val"]
        self.split = split
        self.supervised_train = supervised_train
        self.transforms = transforms_unsup
        # 初始化数据集路径
        # root_dir = "/data0/qidi/LEVIR-CD256"
        root_dir = os.path.join(root_dir, split)
        # 加载图像和标签路径
        T1_image_path = glob.glob(root_dir + '/A' + '/*.png')
        T2_image_path = glob.glob(root_dir + '/B' + '/*.png')
        label_path = glob.glob(root_dir + '/label' + '/*.png')
        # 对路径进行排序
        T1_image_path.sort()
        T2_image_path.sort()
        label_path.sort()
        # 存储路径
        self.T1_image_path = T1_image_path
        self.T2_image_path = T2_image_path
        self.label_path = label_path


    def __getitem__(self, idx):
        sample = {}
        # 加载图像和标签
        image1 = Image.open(self.T1_image_path[idx])
        image2 = Image.open(self.T2_image_path[idx])
        label = cv.imread(self.label_path[idx], 0)
        label = (label != 0).astype('uint8')
        label = torch.from_numpy(label).long()
        # 根据训练模式进行数据预处理
        if self.supervised_train or self.split == "test":
            image1 = imagenet_preprocess(image1)
            image2 = imagenet_preprocess(image2)
            sample['image'] = [image1, image2]
            sample['labels'] = label
        else:      # 'sample0' no augmentation  'sample1' strong augmentation--- ‘sample0’无增强  ‘sample1’强增强
            sample0 = {}
            sample1 = {}
            image11 = imagenet_preprocess(image1)
            image22 = imagenet_preprocess(image2)
            sample0['image'] = [image11, image22]
            image1 = self.transforms(image1)
            image2 = self.transforms(image2)
            sample1['image'] = [image1, image2]

            sample['sample0'] = sample0
            sample['sample1'] = sample1

        return sample

    def __len__(self): # 获取数据集长度
        return len(self.T1_image_path)

