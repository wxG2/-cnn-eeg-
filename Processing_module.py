# -*- coding:utf-8 -*-
# @Author: wxG2
# @Time: 2024/8/19 下午9:58
# @File: Processing_module.py
import mne
import numpy as np
from matplotlib import pyplot as plt
from mne.preprocessing import ICA
from scipy import stats
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
from pyts.image import  GramianAngularField

import csv
import torch
import scipy
import pandas as pd

import numpy as np
from PIL import Image
import math
def processing(raw):
    # read_custom_montage支持tsv
    montage = mne.channels.make_standard_montage("standard_1020")
    # 传⼊数据的电极位置信息
    raw.set_montage(montage, on_missing = "ignore")
    # print(ch)
    # 不加block = True，mne会闪退显示不了。
    # raw.plot(duration=5, n_channels=32, clipping=None, block=True)
    '滤波'
    #50-60hz处有工频干扰
    # raw.plot_psd(average=True)
    # 陷波
    fi_raw = raw.notch_filter(freqs=(60))
    # 带宽滤波
    fi_raw = fi_raw.filter(l_freq=0.5, h_freq=32)
    # raw.plot_psd(average=True)
    # '去坏段'
    # fig = raw.plot(block=True)
    # fig.fake_keypress('a')

    '去坏导'
    # 标记坏道
    fi_raw.info['bads'].extend(['EXG1','EXG2','EXG3','EXG4','EXG5','EXG6','EXG7','EXG8','Status'])
    # 补坏导
    # fi_raw = fi_raw.interpolate_bads()
    # 基于索引删除坏导
    good_picks = [pick for pick in range(len(raw.info['ch_names'])) if raw.info['ch_names'][pick] not in raw.info['bads']]
    # 使用pick_info创建新的Info对象
    new_info = mne.pick_info(raw.info, sel=good_picks)

    # 使用新的Info对象创建不包含坏通道的Raw对象
    fi_raw = mne.io.RawArray(raw._data[good_picks, :], new_info)
    '重参考'
    processed_raw = fi_raw.set_eeg_reference(ref_channels='average')
    # raw.plot(duration=5, n_channels=32, clipping=None, block=True)

    # 添加注释
    my_annot = mne.Annotations(onset=[0, 6.519531],
                               duration=[0.0, 0.0],
                               description=['0', '1'])
    processed_raw = processed_raw.set_annotations(my_annot)

    # # 'ICA主成分分析'
    # ica = ICA(max_iter='auto')
    # # # 对⾼通0.5Hz的数据进⾏ICA及相关成分剔除
    # # ica.fit(processed_raw)
    # reconst_raw = processed_raw.copy()
    # ica.apply(reconst_raw)
    # 绘制地形图
    # ica.plot_components()
    # 提取事件
    # events = mne.events_from_annotations(reconst_raw)
    # epochs = mne.Epochs(reconst_raw, events, event_id=2, tmin=-1, tmax=4, baseline=(-0.5, 0), preload=True, reject=dict(eeg=2e-4))
    '分段'
    # # 定义分段参数

    # events, event_id = mne.events_from_annotations(raw)
    # 取刺激前1s刺激后4s的数据，共5s
    # epochs = mne.Epochs(raw, events, event_id, tmin=-1, tmax=4, baseline=(-0.5, 0),
    #  preload=True)
    # 叠加平均
    # evoked = epochs.average()
    # processed_raw.plot(duration=5, n_channels=32, clipping=None, block=True)
    # 保存为csv文件
    # raw.to_csv('preprocessed_data.csv', index=False)
    '这里是将raw数据中的分段全部添加segment里去了，运行后，利用segment进行分析即可'
    # 定义分段时间窗口时长
    window_size = 5
    overlap = 2
    sfreq = raw.info['sfreq']
    # # 计算每个时间窗口包含的样本数
    window_size_sample = int(window_size * sfreq)
    overlap_sample = int(overlap * sfreq)
    start = 0
    stop = window_size_sample
    step = window_size_sample - overlap_sample
    segments = []
    # 计算理论上可以得到的段数
    max_segments = (processed_raw.n_times - window_size_sample) // step + 1

    # 限制所有数据得到的段数不超过32，这样就可以得到固定尺寸的图片
    max_segments = min(max_segments, 32)
    # 提取段
    for _ in range(max_segments):
        if stop <= processed_raw.n_times:
            segment, _ = processed_raw[:, start:stop]
            segments.append(segment)
            start += step
            stop += step
        else:
            break

            # (32,32,2560),32个段 32个通道 每段包含2560个数据点
    # print(segments.shape)

    return segments
def get_time_domain_feature(data):
    """
    提取 15个 时域特征,1个频域特征
    @param data: shape 为 (m, n) 的 2D array 数据，其中，m 为样本个数， n 为样本（信号）长度
    @return: shape 为 (m, 16)的 2D array 数据，其中，m 为样本个数。即 每个样本的15个时域特征，1个频域特征
    """
    rows, cols = data.shape

    # 有量纲统计量
    max_value = np.amax(data, axis=1)  # 最大值
    peak_value = np.amax(abs(data), axis=1)  # 最大绝对值
    min_value = np.amin(data, axis=1)  # 最小值
    mean = np.mean(data, axis=1)  # 均值
    p_p_value = max_value - min_value  # 峰峰值
    abs_mean = np.mean(abs(data), axis=1)  # 绝对平均值
    rms = np.sqrt(np.sum(data ** 2, axis=1) / cols)  # 均方根值
    square_root_amplitude = (np.sum(np.sqrt(abs(data)), axis=1) / cols) ** 2  # 方根幅值
    # variance = np.var(data, axis=1)  # 方差
    std = np.std(data, axis=1)  # 标准差
    kurtosis = stats.kurtosis(data, axis=1)  # 峭度
    skewness = stats.skew(data, axis=1)  # 偏度
    # mean_amplitude = np.sum(np.abs(data), axis=1) / cols  # 平均幅值 == 绝对平均值

    # 无量纲统计量
    clearance_factor = peak_value / square_root_amplitude  # 裕度指标
    shape_factor = rms / abs_mean  # 波形指标
    impulse_factor = peak_value / abs_mean  # 脉冲指标
    crest_factor = peak_value / rms  # 峰值指标
    # kurtosis_factor = kurtosis / (rms**4)  # 峭度指标
    # 频域特征
    data_fft = np.fft.fft(data, axis=1)
    m, N = data_fft.shape  # 样本个数 和 信号长度
    # 傅里叶变换是对称的，只需取前半部分数据，否则由于 频率序列 是 正负对称的，会导致计算 重心频率求和 等时正负抵消
    mag = np.abs(data_fft)[:, : N // 2]  # 信号幅值
    ps = mag ** 2 / N  #
    mf = np.mean(ps, axis=1)
    features = [max_value, peak_value, min_value, mean, p_p_value, abs_mean, rms, square_root_amplitude,
                std, kurtosis, skewness, clearance_factor, shape_factor, impulse_factor, crest_factor, mf]

    return np.array(features).T

'单被式预处理生成图像'
# i = 0
# data_path = '../dataset/sande_PD/hc/sub-hc'
# sub_path = '/ses-hc/eeg/sub-hc'
# raw_path = data_path + str(i+1) + sub_path + str(i+1) +'_ses-hc_task-rest_eeg.bdf'
# raw = mne.io.read_raw_bdf(raw_path, preload=True)
# processed_raw = processing(raw)
# # processed_raw = processed_raw.get_data()
# processed_raw = np.array(processed_raw)
# # processed_raw = np.transpose(processed_raw,(1, 0, 2))
# feature = []
# for time in processed_raw:
#     # time.shape = (32,2560)
#     # 提取每个通道时域特征
#     time_feature = get_time_domain_feature(time)
#     # time_featue.shape = (32,16)
#     feature.append(time_feature)
# feature = np.array(feature)
# # feature.shape = (32,32,16),意思共有32段，每段有32个通道，每个通道有15个时域特征
# print(feature.shape)
# feature = np.reshape(feature,(-1))
# # 转化为pd的dataframe
# feature = pd.DataFrame(feature)
# # 进行Nan值去除和均值补值
# feature = feature.fillna(feature.mean())
# # 转化回numpy
# feature = np.array(feature)
#
# # 将长为L的时间序列转成m*n的矩阵， L = m*n,生成128*128的图片
# m = 128
# n = 128
# result = feature.reshape((m, n))
# # 矩阵归一化,调用Image
# result = (result - np.min(result)) / (np.max(result) - np.min(result))
# im = Image.fromarray(result * 255.0)
# # 保存图像为1.jpg
# im.convert('L').save("1.jpg", format='jpeg')




