# -*- coding:utf-8 -*-
# @Author: wxG2
# @Time: 2024/8/22 下午2:46
# @File: load_data.py
# -*-coding:utf-8-*-
import os
from os import getcwd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MyDataset(Dataset):
    # 初始化
    def __init__(self, path_dir, transform=None):
        self.path_dir = path_dir
        self.transform = transform
        self.images = os.listdir(self.path_dir)
    # 返回整个数据集大小
    def __len__(self):
        return len(self.images)
    # 根据索引index返回图像及标签
    def __getitem__(self, index):
        image_index = self.images[index]
        # 获取图像的路径或目录
        img_path = os.path.join(self.path_dir, image_index)
        # 获取图像，灰度图要设为‘L’，彩色图设为‘RGB’
        img = Image.open(img_path).convert('L')
        # 按照文件名提取标签，绝对引用用\\，相对引用用/，相对引用就是指从这个脚本文件所在的文件夹路径开始往后加。
        # 即从D:\EEG_code_project\EEG_project\pytorch_project\CNN_PD往后加
        label = img_path.split('/')[-1].split('.')[0]
        # 设置pd标签为1，hc标签为0
        label = 1 if 'pd' in label else 0
        if self.transform is not None:
            img = self.transform(img)
        return img, label

