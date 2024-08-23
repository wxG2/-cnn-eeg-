# -*- coding:utf-8 -*-
# @Author: wxG2
# @Time: 2024/8/19 下午9:58
# @File: cnn_net_module.py
import torch
from torch import nn
#配置神经网络
class CNN(nn.Module):#网络架构
    def __init__(self):
        super(CNN, self).__init__()
        self.module = nn.Sequential(
            # 输入为（4，1，128，128），经过第一个卷积层(4, 32, 128, 128)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=11, stride=1, padding=5),  # 假设padding=5以保持尺寸，但可能需要调整
            nn.ReLU(),
            # 经过第一个池化层（kernel_size = 2, stride = 2），输出尺寸减半，变为(4, 32, 64, 64)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 经过第二个卷积层（kernel_size = 9, padding = 4），输出尺寸不变，仍为(4, 64, 64, 64)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=9, stride=1, padding=4),  # 类似地，假设padding
            nn.ReLU(),
            # 经过第二个池化层，输出尺寸再次减半，变为(4, 64, 32, 32)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 经过第三个卷积层（kernel_size = 7, padding = 3），输出尺寸不变，仍为(4, 128, 32, 32)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            # 经过第三个池化层，输出尺寸再次减半，变为(4, 128, 16, 16)
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        x = self.module(x)
        return x
if __name__ == '__main__':
    net = CNN()
    img = torch.ones((4, 1, 128, 128))
    output = net(img)
    # 返回（4，2）表示4个数据在2个类别的概率是什么样子的
    print(output.shape)