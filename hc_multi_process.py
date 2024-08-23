# -*- coding:utf-8 -*-
# @Author: wxG2
# @Time: 2024/8/19 下午9:59
# @File: hc_multi_process.py
import mne
import numpy as np
from Processing_module import *

# 读取hc数据集
for i in range(16):
    data_path = '../dataset/sande_PD/hc/sub-hc'
    sub_path = '/ses-hc/eeg/sub-hc'
    raw_path = data_path + str(i + 1) + sub_path + str(i + 1) + '_ses-hc_task-rest_eeg.bdf'
    raw = mne.io.read_raw_bdf(raw_path, preload=True)
    # 对每个数据进行预处理，得到的shape：（32，32，2560）
    processed_raw = processing(raw)
    # 转化为数组
    processed_raw = np.array(processed_raw)
    '提取特征'
    # 创建特征列表
    feature = []
    for time in processed_raw:
        # time.shape = (32,2560)
        # 提取每个通道时域特征
        time_feature = get_time_domain_feature(time)
        # time_featue.shape = (32,16)表示每个通道共有16个特征
        feature.append(time_feature)
    # 此时feature.shape(32,32,16)
    feature = np.array(feature)
    # 将特征矩阵降维成一维时间序列
    feature = np.reshape(feature, (-1))
    # 转化为pd的dataframe
    feature = pd.DataFrame(feature)
    # 进行Nan值去除和均值补值
    feature = feature.fillna(feature.mean())
    # 转化回numpy
    feature = np.array(feature)
    '生成灰度图'
    # 将长为L的时间序列转成m*n的矩阵， L = m*n
    m = 128
    n = 128
    result = feature.reshape((m, n))
    # 矩阵归一化,调用Image
    result = (result - np.min(result)) / (np.max(result) - np.min(result))
    im = Image.fromarray(result * 255.0)
    # 将每个被试数据生成的灰度图统一保存在hc_data文件夹中
    im_gray = im.convert('L')
    im_gray.save("hc_data/hc_"+str(i+1)+'.jpg', format='jpeg')