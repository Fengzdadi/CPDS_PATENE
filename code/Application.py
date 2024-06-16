import numpy as np
import pandas as pd
import joblib
import json

from src.preprocessing.interpolation import linear_interpolate_keypoints
from src.preprocessing.abnormal_keypoints_interpolation import interpolate_abnormal_keypoints
from src.preprocessing.noise_suppression import suppress_noise_with_filters
from src.preprocessing.normalize_rotate_keypoints import normalize_and_rotate_keypoints
from src.feature_extraction.angle_displacement import build_angle_displacement_histogram
from src.feature_extraction.relative_direction import build_relative_direction_histogram
from src.feature_extraction.fft_joint_orientation import calculate_orientation_angles, fft_orientation_angles, fft_orientation_angles_new
from src.preprocessing.motion_trajectory_analysis import autocorrelation, calculate_curvature, create_feature_vector
from src.feature_extraction.fft_joint_displacement import calculate_displacement_signal, extract_fft_features

from sklearn.preprocessing import StandardScaler



delta_t = 1
num_bins = 20
features_list = []


def load_keypoints_from_npy(filepath, video_id="1"):
    """
    从.npy文件加载关键点数据，并将其转换为与从JSON加载的数据相似的格式。
    
    参数:
    - filepath: 存储.npy文件的路径。
    - video_id: 视频的ID（默认为"video_001"，可以根据实际需要进行更改）。
    
    返回:
    - 包含每帧关键点数据的列表，每个元素是一个字典。
    """
    # 加载关键点数据
    keypoints_data = np.load(filepath)
    all_keypoints = []
    
    # 创建每一帧的关键点字典
    for frame_id, keypoints in enumerate(keypoints_data):
        # 检查关键点是否全部为0
        if not np.all(keypoints == 0):
            # 扁平化关键点数据以匹配原有格式
            keypoints_list = keypoints.flatten().tolist()
            keypoints_info = {
                'video_id': video_id,
                'frame_id': str(frame_id),
                'keypoints': keypoints_list
            }
            all_keypoints.append(keypoints_info)
    
    return all_keypoints

def expand_keypoints(keypoints_data):
    """
    扩展18个关键点到25个关键点。
    
    参数:
    keypoints_data: DataFrame，包含两列 'frame_id', 'keypoints'，其中 'keypoints' 是包含18个关键点的列表。
    
    返回:
    expanded_keypoints_df: DataFrame，同样格式但包含25个关键点。
    """
    # 初始化新的DataFrame
    expanded_data = []

    # 遍历每一帧
    for index, row in keypoints_data.iterrows():
        frame_id = row['frame_id']
        video_id = row['video_id']
        keypoints = np.array(row['keypoints']).reshape(18, 3)  # 重新塑形为(18, 3)
        new_keypoints = np.zeros((25, 3))  # 初始化25个关键点

        # 复制已有的18个关键点
        new_keypoints[:18, :] = keypoints

        # 获取RAnkle和LAnkle关键点
        r_ankle = keypoints[11, :]
        l_ankle = keypoints[14, :]

        # 估算新的关键点
        # RBigToe
        new_keypoints[22, :] = [r_ankle[0], r_ankle[1] + 0.05, r_ankle[2]]
        # RSmallToe
        new_keypoints[23, :] = [r_ankle[0] + 0.05, r_ankle[1] + 0.05, r_ankle[2]]
        # RHeel
        new_keypoints[24, :] = [r_ankle[0], r_ankle[1] - 0.05, r_ankle[2]]

        # LBigToe
        new_keypoints[19, :] = [l_ankle[0], l_ankle[1] + 0.05, l_ankle[2]]
        # LSmallToe
        new_keypoints[20, :] = [l_ankle[0] - 0.05, l_ankle[1] + 0.05, l_ankle[2]]
        # LHeel
        new_keypoints[21, :] = [l_ankle[0], l_ankle[1] - 0.05, l_ankle[2]]

        # 添加到新的DataFrame数据中
        expanded_data.append({
            'video_id': video_id,
            'frame_id': frame_id,
            'keypoints': new_keypoints.flatten().tolist()
        })

    return pd.DataFrame(expanded_data)

def preprocess_and_extract_features(filepath):
    print("Loading and preprocessing data...")
    # 数据输入改动
    keypoints_data = load_keypoints_from_npy(filepath)
    
    # print(np.shape(keypoints_data))
    # print(keypoints_data)
    # print("Loaded keypoints data:", keypoints_data)

    if not isinstance(keypoints_data, pd.DataFrame):
        keypoints_data = pd.DataFrame(keypoints_data)

    # print(keypoints_data)

    keypoints_data = expand_keypoints(keypoints_data)

    interpolated_data = linear_interpolate_keypoints(keypoints_data)
    interpolated_data = interpolate_abnormal_keypoints(keypoints_data)


    for index, row in interpolated_data.iterrows():
        keypoints = np.array(row['keypoints']).reshape(-1, 3)  # 假设关键点数据是扁平化的
        x_coordinates = keypoints[:, 0]
        y_coordinates = keypoints[:, 1]

        x_filtered = suppress_noise_with_filters(x_coordinates.reshape(-1, 1))
        y_filtered = suppress_noise_with_filters(y_coordinates.reshape(-1, 1))

        interpolated_data.at[index, 'keypoints'] = np.column_stack((x_filtered, y_filtered, keypoints[:, 2])).flatten().tolist()

    keypoints_array = np.array(interpolated_data['keypoints'].tolist()) 
    # print("Processed keypoints:", keypoints_array)
    # print("Processed keypoints shape:", np.shape(keypoints_array))
    processed_keypoints = normalize_and_rotate_keypoints(keypoints_array)

    interpolated_data['keypoints'] = processed_keypoints.tolist()

    print("Extracting features...")

    for video_id, group in keypoints_data.groupby('video_id'):
        
        keypoints_list = group['keypoints'].tolist()
        keypoints_data = np.array(keypoints_list)
    
        keypoints_list = interpolated_data['keypoints'].tolist()
        keypoints_array = np.array(keypoints_list)
        angle_displacement_feature = build_angle_displacement_histogram(keypoints_array, delta_t, num_bins)
        # print("Extracted angle displacement feature:", angle_displacement_feature)

        # 计算相对方向直方图特征
        right_arm = [(2, 3), (3, 4)]
        left_arm = [(5, 6), (6, 7)]
        right_leg = [(9, 10), (10, 11)]
        left_leg = [(12, 13), (13, 14)]
        torso = [(8, 1), (1, 0), (1, 15), (1, 16)]
        limb_joints = [right_arm, left_arm, right_leg, left_leg, torso]

        relative_direction_feature  = build_relative_direction_histogram(keypoints_array, limb_joints, num_bins)
        # print("Extracted relative direction feature:", relative_direction_feature)

        # 计算FFT关节位移
        displacement_signal = calculate_displacement_signal(keypoints)
        fft_features = extract_fft_features(displacement_signal)
        # print("Extracted FFT features:", fft_features)

        # 计算FFT关节相对位移
        keypoints_list = interpolated_data['keypoints'].tolist()
        num_frames = len(keypoints_list)
        num_joints = 25  
        joint_positions = np.array(keypoints_list).reshape(num_frames, num_joints, 3)[:, :, :2]

        orientation_angles = calculate_orientation_angles(joint_positions)
        fft_features_orientation = fft_orientation_angles_new(orientation_angles)
        # print("FFT features for joint orientations:", fft_features_orientation)

        x, y = joint_positions[:, 0, 0], joint_positions[:, 0, 1]
        curvature = calculate_curvature(x, y)
        autocorr = autocorrelation(x) 

        # 创建特征向量
        feature_vector = create_feature_vector(curvature, autocorr)
        # print("Feature vector:", feature_vector)

        print("Shape of angle_displacement_feature:", angle_displacement_feature.shape)
        print("Shape of relative_direction_feature:", relative_direction_feature.shape)
        print("Shape of fft_features:", fft_features.shape)
        print("Shape of feature_vector:", feature_vector.shape)
        print("Shape of fft_features_orientation:", fft_features_orientation.shape)

        scaler1 = StandardScaler()
        angle_displacement_feature_scaled = scaler1.fit_transform(angle_displacement_feature.reshape(-1, 1))

        scaler2 = StandardScaler()
        relative_direction_feature_scaled = scaler2.fit_transform(relative_direction_feature.reshape(-1, 1))

        scaler3 = StandardScaler()
        fft_features_scaled = scaler3.fit_transform(fft_features.reshape(-1, 1))

        scaler4 = StandardScaler()
        feature_vector_scaled = scaler4.fit_transform(feature_vector.reshape(-1, 1))

        scaler5 = StandardScaler()
        fft_features_orientation_scaled = scaler5.fit_transform(fft_features_orientation.reshape(-1, 1))

        angle_displacement_feature_scaled = angle_displacement_feature_scaled.reshape(1, -1)
        relative_direction_feature_scaled = relative_direction_feature_scaled.reshape(1, -1)
        fft_features_scaled = fft_features_scaled.reshape(1, -1)
        feature_vector_scaled = feature_vector_scaled.reshape(1, -1)
        fft_features_orientation_scaled = fft_features_orientation_scaled.reshape(1, -1)


        # 堆叠
        combined_feature_vector = np.hstack([angle_displacement_feature_scaled, relative_direction_feature_scaled, fft_features_scaled, feature_vector_scaled, fft_features_orientation_scaled])
        features_list.append(combined_feature_vector)
    
    X = np.vstack(features_list)
    print("Shape of X:", X.shape)
    print("First 5 rows of X:", X[:5])

    return X

def predict(model, X):
    return model.predict(X)

def predict_probabilities(model, X):
    """
    使用模型对特征数据 X 进行概率预测。
    """
    return model.predict_proba(X)  # 返回概率

def main_predict():
    model = joblib.load('model.pkl')  # 加载模型
    filepath = "output_pose_2.npy"
    X = preprocess_and_extract_features(filepath)
    predictions = predict(model, X)
    print("Predictions:", predictions)
    probabilities = predict_probabilities(model, X)
    print("Probabilities:", probabilities)

    predictions = predictions.tolist()
    probabilities = probabilities.tolist()

    result = {'label':predictions, 'probability':probabilities}

    json.dump(result, open('predictions.json', 'w'))  # 保存预测结果

if __name__ == "__main__":
    main_predict()