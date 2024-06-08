import numpy as np
from scipy.ndimage import median_filter, gaussian_filter


def suppress_noise_with_filters(coordinates, window_size=3, sigma=1):
    """
    对坐标点序列应用中值滤波器和高斯平滑处理以抑制噪声。

    参数:
    - coordinates: 二维数组，形状为[num_frames, num_keypoints]，代表关键点的坐标序列。
    - window_size: 中值滤波器的窗口大小。
    - sigma: 高斯平滑处理的标准差。

    返回:
    - filtered_coordinates: 处理后的坐标点序列。
    """
    # 应用中值滤波
    median_filtered = median_filter(coordinates, size=(window_size, 1))
    
    # 应用高斯平滑处理
    gaussian_smoothed = gaussian_filter(median_filtered, sigma=sigma)
    
    return gaussian_smoothed
