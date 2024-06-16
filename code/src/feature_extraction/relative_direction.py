import numpy as np

def calculate_relative_direction(joint1, joint2):
    """
    计算两个关节之间的相对方向。
    
    参数:
    - joint1: 第一个关节的坐标，格式为(x, y)
    - joint2: 第二个关节的坐标，格式为(x, y)
    
    返回:
    - 两个关节之间的相对方向，以度为单位
    """
    dy = joint2[1] - joint1[1]
    dx = joint2[0] - joint1[0]
    angle = np.arctan2(dy, dx)
    return np.degrees(angle)  # 将结果从弧度转换为度


def build_relative_direction_histogram(keypoints, limb_joints, num_bins=2):
    """
    为每个肢体构建相对方向的直方图。
    
    参数:
    - keypoints: 关键点数据，格式为[num_frames, num_keypoints*2]的numpy数组
    - limb_joints: 每个肢体的关节对列表，例如[(0, 1), (1, 2)]代表肢体由关节0到1和1到2组成
    - num_bins: 直方图的bin数量
    
    返回:
    - 所有肢体的综合直方图特征向量
    """
    histograms = []
    
    for limb in limb_joints:
        relative_directions = []
        for joint_pair in limb:
            for frame in keypoints:
                joint1 = frame[joint_pair[0]*2:joint_pair[0]*2+2]
                joint2 = frame[joint_pair[1]*2:joint_pair[1]*2+2]
                angle = calculate_relative_direction(joint1, joint2)
                relative_directions.append(angle)
        
        # 构建直方图并归一化
        hist, _ = np.histogram(relative_directions, bins=num_bins, range=(-180, 180))
        hist = hist / np.sum(hist)
        histograms.extend(hist)
    
    return np.array(histograms)
