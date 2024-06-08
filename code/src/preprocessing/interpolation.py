import pandas as pd
import numpy as np

def linear_interpolate_keypoints(keypoints_data):
    df_keypoints = pd.DataFrame(keypoints_data)
    interpolated_data = []

    for video_id, group in df_keypoints.groupby('video_id'):
        sorted_group = group.sort_values(by='frame_id')
        keypoints = np.array(sorted_group['keypoints'].tolist())

        num_frames, num_keypoints = keypoints.shape
        for j in range(0, num_keypoints, 3):  # 每3个数处理一次 (x, y, 置信度)
            x = keypoints[:, j]
            y = keypoints[:, j+1]
            confidence = keypoints[:, j+2]
            
            # 对x, y坐标进行线性插值，仅当置信度为0时
            valid = confidence > 0
            invalid = ~valid

            # 使用numpy的interp函数进行线性插值，为无效点填充值
            if np.any(invalid) and np.sum(valid) > 1:  # 至少需要两个有效点进行插值
                x[invalid] = np.interp(np.flatnonzero(invalid), np.flatnonzero(valid), x[valid])
                y[invalid] = np.interp(np.flatnonzero(invalid), np.flatnonzero(valid), y[valid])
                # 对于置信度，可能需要一个合理的策略来插值或赋值，这里我们简单地将插值后的点置信度设为相邻点置信度的平均值
                valid_confidences = confidence[valid]
                if len(valid_confidences) > 0:
                    average_confidence = np.mean(valid_confidences)
                else:
                    average_confidence = 0  # 或者选择一个合理的默认值
                confidence[invalid] = average_confidence
            # 更新关键点数据
            keypoints[:, j] = x
            keypoints[:, j+1] = y
            keypoints[:, j+2] = confidence

        # 将处理后的数据加回到结果列表中
        for i, row in enumerate(sorted_group.to_dict('records')):
            row['keypoints'] = keypoints[i].tolist()
            interpolated_data.append(row)

    return pd.DataFrame(interpolated_data)

