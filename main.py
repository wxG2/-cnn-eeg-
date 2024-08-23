# -*- coding:utf-8 -*-
# @Author: wxG2
# @Time: 2024/8/22 下午3:53
# @File: main.
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import time
from CNN_PD import cnn_net_module
from cnn_net_module import *
from torch.utils.tensorboard import SummaryWriter
from load_data import *
import numpy as np
'加载数据'
train_path = 'Dataset/train/hc'
test_path = 'Dataset/test'
# 训练集 返回的数据为（4，1，128，128）
transforms = transforms.Compose([transforms.ToTensor()])
# 读取img的形状为（1，128，128）,dataset包含img和label
dataset = MyDataset(train_path, transform=transforms)
# dataloader为（4，1，128，128），4指一次取的样本数
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
# 测试集
test_data = MyDataset(test_path, transform=transforms)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

'训练'
# 设置用GPU训练
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 构建CNN网络
cnn = cnn_net_module.CNN()
cnn.to(device)
# 设置损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
# 设置学习率
learning_rate = 0.001
# 定义优化器
optmizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate)

#记录训练次数
train_step = 0

#记录测试次数
test_step = 0

#记录训练的轮数
epoch = 30

#利用tensorboard表示
start_time = time.time()
for i in range(epoch):
    print("----训练第{}轮开始----".format(i+1))
    cnn.train()
    #训练步骤开始
    for data in train_dataloader:
        img, label = data
        img = img.to(device)
        label = label.to(device)
        output = cnn(img)
        #计算损失
        loss = loss_fn(output, label)
        optmizer.zero_grad()
        loss.backward()
        #优化器优化
        optmizer.step()
        train_step += 1
        if train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("共训练次数{}，loss：{}".format(train_step, loss.item()))#用.item()，是因为这样输出不会带tensor()

    '测试'
    cnn.eval()
    test_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0.0
    with torch.no_grad():
        for data in test_dataloader:
            img, label = data
            img = img.to(device)
            label = label.to(device)
            output = cnn(img)
            #计算损失
            loss = loss_fn(output, label)
            test_loss = loss.item() + test_loss
            accuracy = (output.argmax(1) == label).sum()
            total_accuracy = total_accuracy + accuracy
            num_batches += 1
    test_loss /= num_batches
    print("测试集的正确率：{}".format(total_accuracy/len(test_dataloader)))
    test_step += 1
    torch.save(cnn, 'cnn_model/cnn_{}.pth'.format(i))
    print("模型已保存")
