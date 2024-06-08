import numpy as np

def calculate_orientation_angles(joint_positions):
    num_frames, num_joints, _ = joint_positions.shape
    orientation_angles = np.zeros((num_frames, num_joints))

    for j in range(1, num_joints):  # 假设第0个关节没有父关节
        for i in range(num_frames):
            parent_j = j - 1  # 简化的父关节索引，实际应根据关节结构确定
            dy = joint_positions[i, j, 1] - joint_positions[i, parent_j, 1]
            dx = joint_positions[i, j, 0] - joint_positions[i, parent_j, 0]
            orientation_angles[i, j] = np.arctan2(dy, dx)
    
    return orientation_angles

def fft_orientation_angles(orientation_angles):
    num_frames, num_joints = orientation_angles.shape
    fft_features = np.zeros((num_joints, num_frames//2+1))  # 取FFT的一半，因为FFT是对称的

    for j in range(num_joints):
        fft_result = np.fft.rfft(orientation_angles[:, j])
        fft_features[j, :] = np.abs(fft_result)
    
    return fft_features.flatten()  # 将特征向量展平

def fft_orientation_angles_new(orientation_angles, num_features=10):
    num_frames, num_joints = orientation_angles.shape
    # 更新：只选择每个关节FFT结果的前num_features个频率分量
    fft_features = np.zeros((num_joints, num_features))

    for j in range(num_joints):
        fft_result = np.fft.rfft(orientation_angles[:, j])
        fft_features[j, :] = np.abs(fft_result)[:num_features]  # 只取前num_features个特征
    
    return fft_features.flatten()  # 将特征向量展平
