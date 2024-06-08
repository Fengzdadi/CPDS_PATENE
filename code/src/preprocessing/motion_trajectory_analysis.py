import numpy as np

def calculate_curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    curvature = np.abs(dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)
    return curvature

def autocorrelation(x):
    mean_x = np.mean(x)
    autocorr_f = np.correlate(x - mean_x, x - mean_x, mode='full')
    autocorr_f = autocorr_f[autocorr_f.size // 2:]
    return autocorr_f / autocorr_f[0]

def create_feature_vector(curvature, autocorr):
    features = [np.mean(curvature)]  # 示例：使用曲率的平均值作为特征
    # 添加自相关函数的特征，例如特定τ下的值
    features.extend(autocorr[:5])  # 示例：取自相关函数的前5个值
    return np.array(features)
