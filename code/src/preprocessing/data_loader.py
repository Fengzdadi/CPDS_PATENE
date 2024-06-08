import os
import json
import pandas as pd

def load_keypoints_with_frame_info(directory):
    all_keypoints = []
    for filename in os.listdir(directory):
        if filename.endswith('_keypoints.json'):
            # 解析视频编号和帧编号
            parts = filename.split('_')
            video_id = parts[2]  # 假设视频编号总是在第三个位置
            frame_id = parts[3]  # 假设帧编号总是在第四个位置

            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                for person in data['people']:
                    keypoints = person['pose_keypoints_2d']
                    # 将视频编号和帧编号添加到关键点数据中
                    keypoints_data = {'video_id': video_id, 'frame_id': frame_id, 'keypoints': keypoints}
                    all_keypoints.append(keypoints_data)
    return all_keypoints

# 你的数据目录
directory = "C:\\Users\\caifengze\\Desktop\\CPDS_PATENE\\data\\00_25J_MRGB\\MRGBD"      #MRGBD

keypoints_data = load_keypoints_with_frame_info(directory)

# 将数据转换为Pandas DataFrame进行进一步处理
df_keypoints = pd.DataFrame(keypoints_data)

# 查看DataFrame的前几行以验证数据加载是否正确
# print(df_keypoints.head())
