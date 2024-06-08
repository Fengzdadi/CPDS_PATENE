import numpy as np

def calculate_angle_displacement(v_t, v_t_plus_delta):
    """
    计算两个向量之间的角位移。
    
    参数:
    - v_t: 时间t的向量
    - v_t_plus_delta: 时间t+Δt的向量
    
    返回:
    - 角位移，以度为单位
    """
    cos_theta = np.dot(v_t, v_t_plus_delta) / (np.linalg.norm(v_t) * np.linalg.norm(v_t_plus_delta))
    # 限制cos_theta的值在-1到1之间，以防止由于浮点误差导致的问题
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return np.degrees(angle)  # 将结果从弧度转换为度

def build_angle_displacement_histogram(keypoints, delta_t, num_bins=10):
    """
    构建角位移直方图特征。
    
    参数:
    - keypoints: 关键点数据，假设为[num_frames, num_keypoints*3]的numpy数组
    - delta_t: 时间间隔Δt
    - num_bins: 直方图的bin数量
    
    返回:
    - 角位移直方图特征向量
    """
    angle_displacements = []

    # 假设关键点向量的定义和父子关系已知，这里需要根据实际情况调整
    for t in range(len(keypoints) - delta_t):
        # 示例：计算一个身体部位在时间t和t+Δt的角位移
        # 这里需要根据实际的父子关系调整索引
        v_t = keypoints[t, :2] - keypoints[t, 3:5]
        v_t_plus_delta = keypoints[t + delta_t, :2] - keypoints[t + delta_t, 3:5]
        angle = calculate_angle_displacement(v_t, v_t_plus_delta)
        angle_displacements.append(angle)
    
    # 构建直方图特征
    hist, _ = np.histogram(angle_displacements, bins=num_bins, range=(0, 180))
    # 归一化直方图
    hist = hist / np.sum(hist)
    
    return hist
