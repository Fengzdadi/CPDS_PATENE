import numpy as np

def calculate_displacement_signal(keypoints):
    """
    计算每个关节的位移信号。
    
    参数:
    - keypoints: 关键点数据，格式为[num_frames, num_keypoints*2]的numpy数组
    
    返回:
    - 位移信号，格式为[num_frames-1, num_keypoints]的numpy数组
    """
    num_frames, num_keypoints = keypoints.shape[0], keypoints.shape[1] // 2
    displacement_signal = np.zeros((num_frames-1, num_keypoints))
    
    for k in range(num_keypoints):
        for t in range(num_frames-1):
            displacement_signal[t, k] = np.linalg.norm(keypoints[t+1, k*2:k*2+2] - keypoints[t, k*2:k*2+2])
            
    return displacement_signal

def extract_fft_features(displacement_signal):
    """
    对位移信号应用FFT，并提取每个频率分量的幅值。
    
    参数:
    - displacement_signal: 位移信号，格式为[num_frames-1, num_keypoints]的numpy数组
    
    返回:
    - FFT特征向量
    """
    num_keypoints = displacement_signal.shape[1]
    fft_features = []
    
    for k in range(num_keypoints):
        fft_result = np.fft.fft(displacement_signal[:, k])
        amplitude = np.abs(fft_result)
        fft_features.extend(amplitude)
        
    return np.array(fft_features)
