import math
import numpy as np
from scipy.ndimage import convolve1d,gaussian_filter1d
import torch

def unbiased_rmse(y_true, y_pred):
    predmean = np.nanmean(y_pred)
    targetmean = np.nanmean(y_true)
    predanom = y_pred-predmean
    targetanom = y_true - targetmean
    return np.sqrt(np.nanmean((predanom-targetanom)**2))


def r2_score(y_true, y_pred):
    mask = y_true == y_true
    a, b = y_true[mask], y_pred[mask]
    unexplained_error = np.nansum(np.square(a-b))
    total_error = np.nansum(np.square(a - np.nanmean(a)))
    return 1. - unexplained_error/total_error

def nanunbiased_rmse(y_true, y_pred):
    predmean = np.mean(y_pred)
    targetmean = np.mean(y_true)
    predanom = y_pred-predmean
    targetanom = y_true - targetmean
    return np.sqrt(np.mean((predanom-targetanom)**2))

def _rmse(y_true,y_pred):
    predanom = y_pred
    targetanom = y_true
    return np.sqrt(np.nanmean((predanom-targetanom)**2))
def _bias(y_true,y_pred):
    bias = np.nanmean(np.abs(y_pred-y_true))
    return bias

def _ACC(y_true,y_pred):
    y_true_anom = y_true-np.nanmean(y_true)
    y_pred_anom = y_pred-np.nanmean(y_pred)
    numerator = np.sum(y_true_anom*y_pred_anom)
    denominator = np.sqrt(np.sum(y_true_anom**2))*np.sqrt(np.sum(y_pred_anom**2))
    acc = numerator/denominator
    return acc


def get_bin_idx(label, bins, bins_edges):
    if math.isnan(label):
        return bins - 1
    if label > 0.6:
        return bins - 1
    else:
        return np.where(bins_edges >= label)[0][0] - 1


def calculate_kde_weight_renorm(y, cfg, mask, scaler_y):
    a, b = np.array(scaler_y[0]), np.array(scaler_y[1])
    a = a[np.newaxis, :, :, :]
    b = b[np.newaxis, :, :, :]
    y_renorm = y * (b - a) + a
    y_renorm = y_renorm.transpose(0, 3, 1, 2)
    y_renorm = y_renorm.reshape(y_renorm.shape[0], y_renorm.shape[2] * y_renorm.shape[3])
    mask = mask.reshape(mask.shape[0] * mask.shape[1])
    y_renorm = y_renorm[:, mask == 1]
    # nt,ngrid
    y = y_renorm[cfg['seq_len'] + cfg["forcast_time"] - 1:, :]
    bins = 3000  # 1000  #直方图计算，定义区间数
    value_list, bins_edges = np.histogram(y, bins=bins, range=(0, 0.6))  # value_list：每个区间内的数据点数目 bins_edges：区间的边界值
    print(y.shape)
    # value_list,bins_edges = np.histogram(y,bins=bins,range=(np.nanmax(y),np.nanmin(y)))
    ks = 75  # 55 核大小
    sigma = 5  # 10 标准差
    half_ks = (ks - 1) // 2
    base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
    lds_kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / np.sum(
        gaussian_filter1d(base_kernel, sigma=sigma))  # 归一化：确保核的总和为 1
    smoothed_value = convolve1d(value_list, weights=lds_kernel_window, mode='constant')  # 通过一维卷积操作来平滑 value_list
    num_per_label = [smoothed_value[get_bin_idx(label, bins, bins_edges)] for label in y.reshape(-1)]
    print(np.max(num_per_label))
    print(np.min(num_per_label))
    num_per_label = (num_per_label - np.min(num_per_label)) / (np.max(num_per_label) - np.min(num_per_label))  # 归一化
    print(num_per_label)
    # weight = (1/num_per_label).reshape(y.shape)
    a = 0.9  # 调整权重比例的系数
    b = 0.1  # b 是一个最小值阈值
    fw1 = 1 - a * num_per_label  # 使得频数较高的标签得到较低的 fw1 值，而频数较低的标签得到较高的 fw1 值
    fw2 = np.maximum(fw1, b)  # 将 fw1 中的每个值与 b 进行比较，取两者中的最大值，得到一个新的数组 fw2。不会导致权重过低
    density = fw2 / (np.sum(fw2) / fw2.size)  # 归一化后的密度值
    weight = density.reshape(y.shape)  # 将 density 重塑为与原始 y 相同的形状，得到最终的权重数组 weight。

    return weight
    
    
def lat_weight(y,mask):
    nt,_,_,_ = y.shape
    mask = mask.reshape(mask.shape[0] * mask.shape[1])
    latitudes_1deg = np.linspace(-88, 88, 45)  # 1° 分辨率，纬度共 180 个格点
    longitudes_1deg = np.linspace(2, 358, 90)  # 1° 分辨率，经度共 360 个格点

    # 生成纬度权重（低纬度权重高，高纬度权重低）
    def generate_latitude_weights(latitudes):
        # 权重从0到1线性递减
        weights = np.clip((90 - np.abs(latitudes)) / 90, 0, 1)
        return weights

    weights_lat_1deg = generate_latitude_weights(latitudes_1deg)
    weights_matrix_1deg = np.repeat(weights_lat_1deg[:, np.newaxis], len(longitudes_1deg), axis=1)

    # 扩展到 (时间步, 纬度, 经度)
    time_steps = nt  # 示例时间步
    weights_array_1deg = np.repeat(weights_matrix_1deg[np.newaxis, :, :], time_steps, axis=0)

    def normalize_weights(weights):
        min_weight = np.min(weights)
        max_weight = np.max(weights)
        return (weights - min_weight) / (max_weight - min_weight)

    weights_normalized_1deg = normalize_weights(weights_array_1deg)

    weights_flattened_1deg = weights_normalized_1deg.reshape(time_steps, -1)
    weights_flattened_1deg_mask = weights_flattened_1deg[:, mask == 1]

    return weights_flattened_1deg_mask
    
def _rv(y_true, y_pred):
    # �����ֵ
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    # ������Ա�����
    rv = np.sqrt(np.mean((y_pred - mean_pred) ** 2) / np.mean((y_true - mean_true) ** 2))

    return rv


def _fhv(y_true, y_pred):
    # ���۲�ֵ��ģ��ֵ�����һ�𣬷�������
    combined = np.column_stack((y_true, y_pred))
    # ����ģ��ֵ��������
    sorted_combined = combined[combined[:, 1].argsort()]
    # ����ǰ2%�ĸ߷�������Χ
    percentile_2 = int(0.02 * len(y_true))
    # ��ȡǰ2%�ĸ߷�����
    fhv_values = sorted_combined[-percentile_2:]
    # ����ٷ�ƫ��
    fhv = np.mean((fhv_values[:, 1] - fhv_values[:, 0]) / fhv_values[:, 0])

    return fhv


def _flv(y_true, y_pred):
    # ���۲�ֵ��ģ��ֵ�����һ�𣬷�������
    combined = np.column_stack((y_true, y_pred))

    # ����ģ��ֵ��������
    sorted_combined = combined[combined[:, 1].argsort()]

    # ����ײ�30%�ĵ�������Χ
    percentile_30 = int(0.3 * len(y_true))

    # ��ȡ�ײ�30%�ĵ�����
    flv_values = sorted_combined[:percentile_30]

    # ����ٷ�ƫ��
    flv = np.mean((flv_values[:, 1] - flv_values[:, 0]) / flv_values[:, 0])

    return flv

def _ACC(y_true,y_pred):
    y_true_anom = y_true-np.nanmean(y_true)
    y_pred_anom = y_pred-np.nanmean(y_pred)
    numerator = np.sum(y_true_anom*y_pred_anom)
    denominator = np.sqrt(np.sum(y_true_anom**2))*np.sqrt(np.sum(y_pred_anom**2))
    acc = numerator/denominator
    return acc





def GetKGE(Qo, Qs):
    # Input variables
    # Qs: Simulated runoff
    # Qo: Observed runoff
    # Output variable
    # KGE: Kling-Gupta edddddfficiency coefficient
    if isinstance(Qs, torch.Tensor):
        Qs = Qs.cpu()
        Qo = Qo.cpu()
        Qs = Qs.numpy()  # Convert Qs to a NumPy array
        Qo = Qo.numpy()  # Convert Qo to a NumPy array

    if len(Qs) == len(Qo):
        mask = Qo != 0
        Qo = Qo[mask]
        Qs = Qs[mask]
        QsAve = np.mean(Qs)
        QoAve = np.mean(Qo)
        # COV = np.cov(np.array(Qs).flatten(), np.array(Qo).flatten())
        # COV = np.cov(Qs, Qo)
        # CC = COV[0, 1] / np.std(Qs) / np.std(Qo)
        CC = np.corrcoef(Qo, Qs)[0, 1]
        BR = QsAve / QoAve
        RV = (np.std(Qs) / QsAve) / (np.std(Qo) / QoAve)
        KGE = 1 - np.sqrt((CC - 1) ** 2 + (BR - 1) ** 2 + (RV - 1) ** 2)
        return KGE
from scipy.stats import pearsonr


def GetRMSE(y_pred,y_true):
    mask = y_true != 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    predanom = y_pred
    targetanom = y_true
    return np.sqrt(np.nanmean((predanom-targetanom)**2))

def GetMAE(y_pred,y_true):
    
    n = len(y_true)
    # ���ݵ������
    mae = sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n
    return mae

def GetPCC(Qs, Qo):

    if isinstance(Qs, torch.Tensor):
        Qs = Qs.cpu()
        Qo = Qo.cpu()
        Qs = Qs.numpy()  # Convert Qs to a NumPy array
        Qo = Qo.numpy()  # Convert Qo to a NumPy array

    if len(Qs) == len(Qo):
        # COV = np.cov(Qs, Qo)
        # COV = np.cov(np.array(Qs).flatten(), np.array(Qo).flatten())
        # # PCC = np.corrcoef(Qo, Qs)[0, 1]
        #
        # PCC = COV / (np.std(Qs) * np.std(Qo))
        mask = Qo != 0

        Qs[np.isnan(Qs)] = 0
        Qo[np.isnan(Qo)] = 0

        PCC, _ = pearsonr(Qs, Qo)
        return PCC

def GetNSE(simulated,observed):

    if isinstance(observed, torch.Tensor):
        observed = observed.cpu()
        simulated = simulated.cpu()
        observed = observed.numpy()
        simulated = simulated.numpy()
    mask = observed != 0
    observed = observed[mask]
    simulated = simulated[mask]
    mean_observed = np.mean(observed)
    numerator = np.sum((simulated - observed)**2)
    denominator = np.sum((observed - mean_observed)**2)
    nse = 1 - numerator / denominator
    return nse