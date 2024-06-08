import numpy as np

def normalize_and_rotate_keypoints(keypoints, standard_height=1.0):
    """
    对关键点进行归一化和旋转以减少身体大小和斜侧的影响。

    参数:
    - keypoints: 关键点数据，假设为[num_frames, num_keypoints*3]的numpy数组。
    - standard_height: 身高的标准化值。

    返回:
    - 归一化和旋转后的关键点数据。
    """
    # 使用适当的关键点索引估算身高H
    # 选择下腿、上腿、臀部到颈部和头部这些身体部分的关键点
    segments = [
        (22, 10),  # RBigToe - RKnee
        (19, 13),  # LBigToe - LKnee
        (10, 9),   # RKnee - RHip
        (13, 12),  # LKnee - LHip
        (9, 8),    # RHip - MidHip
        (12, 8),   # LHip - MidHip
        (1, 8)     # Neck - MidHip
    ]
    
    H = np.sum([np.linalg.norm(keypoints[:, 3*p_start:3*p_start+2] - keypoints[:, 3*p_end:3*p_end+2], axis=1) for p_start, p_end in segments], axis=0)
    H_mean = np.mean(H)

    # 计算缩放因子S
    S = standard_height / H_mean

    # 缩放关键点
    keypoints[:, 0::3] *= S  # x坐标
    keypoints[:, 1::3] *= S  # y坐标

    # 计算旋转角度α
    neck = keypoints[:, 1*3:1*3+2]  # 颈部坐标
    mid_hip = keypoints[:, 8*3:8*3+2]  # MidHip坐标
    direction = neck - mid_hip
    angle = np.arctan2(direction[:, 1], direction[:, 0]) - np.pi/2

    # 应用旋转变换
    for i in range(len(keypoints)):
        for j in range(0, keypoints.shape[1], 3):
            x, y = keypoints[i, j], keypoints[i, j+1]
            keypoints[i, j] = np.cos(angle[i]) * x - np.sin(angle[i]) * y
            keypoints[i, j+1] = np.sin(angle[i]) * x + np.cos(angle[i]) * y

    # 转换坐标相对于初始帧
    initial_frame = keypoints[0, :].reshape(-1, 3)
    for i in range(1, len(keypoints)):
        keypoints[i, 0::3] -= initial_frame[:, 0]
        keypoints[i, 1::3] -= initial_frame[:, 1]

    return keypoints
