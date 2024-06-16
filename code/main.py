import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.data_loader import load_keypoints_with_frame_info
from src.preprocessing.interpolation import linear_interpolate_keypoints
from src.preprocessing.abnormal_keypoints_interpolation import interpolate_abnormal_keypoints
from src.preprocessing.noise_suppression import suppress_noise_with_filters
from src.preprocessing.normalize_rotate_keypoints import normalize_and_rotate_keypoints
from src.feature_extraction.angle_displacement import build_angle_displacement_histogram
from src.feature_extraction.relative_direction import build_relative_direction_histogram
from src.feature_extraction.fft_joint_orientation import calculate_orientation_angles, fft_orientation_angles, fft_orientation_angles_new
from src.preprocessing.motion_trajectory_analysis import autocorrelation, calculate_curvature, create_feature_vector
from src.feature_extraction.fft_joint_displacement import calculate_displacement_signal, extract_fft_features
from model.stacking_model import evaluate_stacking_model, train_stacking_model
import joblib
from sklearn.preprocessing import StandardScaler

# from .src.models import calculate_joint_orientation_angles_for_limb, extract_fft_features_for_limb_orientation


# 定义时间间隔Δt和直方图的bin数量
delta_t = 1  # 示例：每隔1帧计算角位移
num_bins = 20  # 示例：将0到180度的角位移分成10个bin

# 假设 labels_array 是包含所有视频标签的数组，其中0表示无脑性瘫痪，1表示有脑瘫
labels_array = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1]])
# array([[0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# 初始化列表以存储分割后的数据
features_list = []
labels_list = []



# 其他预处理、特征提取和模型训练的导入语句

def main():
    # 数据目录
    data_directory = "C:\\Users\\caifengze\\Desktop\\CPDS_PATENE\\data\\00_25J_MRGB\\MRGBD"      #MRGBD
    # data_directory = "C:\\Users\\caifengze\\Desktop\\CPDS_PATENE\\data\\00_25J_RVI_38_Full\\25J_RVI38_Full_Processed"      #RVI_38_Full_Processed
    # data_directory = "C:\\Users\\caifengze\\Desktop\\CPDS_PATENE\\data\\00_25J_RVI_38_Full_mini\\25J_RVI38_Full_Processed"      #RVI_38_Full_Processed

    # 步骤1: 加载数据
    print("Loading data...")
    keypoints_data = load_keypoints_with_frame_info(data_directory)
    # print(keypoints_data)
    print(np.shape(keypoints_data))

    # if not isinstance(keypoints_data, pd.DataFrame):
    #     keypoints_data = pd.DataFrame(keypoints_data)
    
    # keypoints_data.to_csv("keypoints_data.csv", index=False)

    if not isinstance(keypoints_data, pd.DataFrame):
        keypoints_data = pd.DataFrame(keypoints_data)
    
    # nan_counts = keypoints_data['keypoints'].apply(lambda keypoints: np.isnan(np.array(keypoints)).sum()).sum()
    # print("Total NaN count in keypoints:", nan_counts)
    # Total NaN count in keypoints: 0
    
    # 步骤2: 数据预处理
    # 例如，线性插值处理缺失数据点
    print("Preprocessing data...")
    interpolated_data = linear_interpolate_keypoints(keypoints_data)
    # print(interpolated_data.head())
    # print("Total NaN count in keypoints:", nan_counts)
    # ! pass

    # nan_counts = interpolated_data['keypoints'].apply(lambda keypoints: np.isnan(np.array(keypoints)).sum()).sum()
    # 对异常的关节点进行插值处理
    interpolated_data = interpolate_abnormal_keypoints(keypoints_data)
    print(interpolated_data.head())

    # nan_counts = interpolated_data['keypoints'].apply(lambda keypoints: np.isnan(np.array(keypoints)).sum()).sum()
    # print("Total NaN count in keypoints:", nan_counts)
    # Total NaN count in keypoints: 13206
    # ! pass

    # 对坐标点序列应用中值滤波器和高斯平滑处理以抑制噪声
    for index, row in interpolated_data.iterrows():
        keypoints = np.array(row['keypoints']).reshape(-1, 3)  # 假设关键点数据是扁平化的
        x_coordinates = keypoints[:, 0]
        y_coordinates = keypoints[:, 1]
        
        # 应用噪声抑制
        x_filtered = suppress_noise_with_filters(x_coordinates.reshape(-1, 1))
        y_filtered = suppress_noise_with_filters(y_coordinates.reshape(-1, 1))
        # 更新DataFrame
        interpolated_data.at[index, 'keypoints'] = np.column_stack((x_filtered, y_filtered, keypoints[:, 2])).flatten().tolist()
    # print(x_coordinates)
    # print(y_coordinates)
    # print(interpolated_data.head())
    # Gussi 滤波后数据差的比较大
    
    # 其他预处理步骤，如坐标预处理、姿态归一化等
        
    # nan_counts = interpolated_data['keypoints'].apply(lambda keypoints: np.isnan(np.array(keypoints)).sum()).sum()
    # print("Total NaN count in keypoints:", nan_counts)
    # ! pass
        
    keypoints_array = np.array(interpolated_data['keypoints'].tolist())  # 假设每行的关键点数据已经是完整的[x, y, confidence]列表
    
    print("Processed keypoints:", keypoints_array)
    print("Processed keypoints shape:", np.shape(keypoints_array))

    processed_keypoints = normalize_and_rotate_keypoints(keypoints_array)
    interpolated_data['keypoints'] = processed_keypoints.tolist()
    # interpolated_data.to_csv("C:/Users/caifengze/Desktop/CPDS_PATENE/code/tmp/interpolated_keypoints.csv", index=False)
    print(interpolated_data.head())

    # 步骤3: 特征提取
    print("Extracting features...")
    for video_id, group in keypoints_data.groupby('video_id'):
        group.to_csv(f"group_{video_id}.csv", index=False)
        # 提取这个组的关键点列表
        keypoints_list = group['keypoints'].tolist()
        print("keypoints_list",keypoints_list[:10])
        # 将列表转换为NumPy数组以供特征提取函数使用
        # keypoints_data_new = np.array(keypoints_list)
    
        # 特征提取的代码逻辑

        # 计算角位移直方图特征
        # keypoints_list = interpolated_data['keypoints'].tolist()  # 将DataFrame列转换为列表
        keypoints_array = np.array(keypoints_list)  # 将列表转换为NumPy数组
        angle_displacement_feature = build_angle_displacement_histogram(keypoints_array, delta_t, num_bins)
        print("Extracted angle displacement feature:", angle_displacement_feature)

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
        num_joints = 25  # 根据实际的关键点数量更新
        joint_positions = np.array(keypoints_list).reshape(num_frames, num_joints, 3)[:, :, :2]  # 假设最后一个维度是[x, y, confidence]，这里我们只取[x, y]

        orientation_angles = calculate_orientation_angles(joint_positions)
        fft_features_orientation = fft_orientation_angles_new(orientation_angles)
        # print("FFT features for joint orientations:", fft_features_orientation)

        # num_frames = len(keypoints_list)
        # # num_joints = len(keypoints_list[0]) // 2  # 假设每个关键点有x和y两个坐标值
        # num_joints = 37
        # # 将列表转换为[num_frames, num_joints, 2]形状的NumPy数组
        # joint_positions = np.array(keypoints_list).reshape(num_frames, num_joints, 2)
        # limb_joints_all = [
        #     (17,15), 
        #     (15, 0), 
        #     ( 0,16),
        #     (16,18),
        #     ( 0, 1),
        #     ( 1, 2),
        #     ( 1, 5),
        #     ( 2, 3),
        #     ( 3, 4),
        #     ( 5, 6),
        #     ( 6, 7),
        #     ( 1, 8),
        #     ( 8, 9),
        #     ( 9,10),
        #     (10,11),
        #     (11,24),
        #     (11,22),
        #     (22,23),
        #     ( 8,12),
        #     (12,13),
        #     (13,14),
        #     (14,21),
        #     (14,19),
        #     (19,20)
        #     ]
        # orientation_angles = calculate_joint_orientation_angles_for_limb(joint_positions, limb_joints_all)
        # fft_features = extract_fft_features_for_limb_orientation(orientation_angles, num_bins)
        # print("Extracted FFT features for limb orientation:", fft_features)

        # 自相关特征的代码逻辑
        x, y = joint_positions[:, 0, 0], joint_positions[:, 0, 1]  # 第一个关节的x和y轨迹
        curvature = calculate_curvature(x, y)
        autocorr = autocorrelation(x)  # 示例：只对x轨迹计算自相关

        # 创建特征向量
        feature_vector = create_feature_vector(curvature, autocorr)
        print("Feature vector:", feature_vector)

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

        # print("angle_displacement_feature_scaled",angle_displacement_feature_scaled)
        # print("relative_direction_feature_scaled",relative_direction_feature_scaled)
        # print("fft_features_scaled",fft_features_scaled)
        # print("feature_vector_scaled",feature_vector_scaled)
        # print("fft_features_orientation_scaled",fft_features_orientation_scaled)

        # 堆叠
        combined_feature_vector = np.hstack([angle_displacement_feature_scaled, relative_direction_feature_scaled, fft_features_scaled, feature_vector_scaled, fft_features_orientation_scaled])
        features_list.append(combined_feature_vector)

    X = np.vstack(features_list)  # 特征矩阵
    y = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1])
    # y = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # 将X转换为DataFrame
    features_df = pd.DataFrame(X)
    # 将y添加为新列
    features_df['Label'] = y
    features_df.to_csv('features_labels.csv', index=False)

    # 标签数组
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 步骤4: 模型训练
    print("Training model...")
    model = train_stacking_model(X_train, y_train)
    # 模型训练的代码逻辑

    # 步骤5: 模型评估
    print("Evaluating model...")
    
    # 模型评估的代码逻辑
    evaluate_stacking_model(model, X_test, y_test)

    # 步骤6: 结果可视化
    print("Visualizing results...")
    # 结果可视化的代码逻辑

    joblib.dump(model, 'model.pkl')  # 保存模型

if __name__ == "__main__":
    main()
